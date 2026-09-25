"""
quant_model_macro_3.py

Step 3 of the macro-factor model: standard Fama-French 5-factor regressions,
but now the dependent variables are the Yahoo Finance asset-class monthly
excess returns (Data/asset_class_monthly_returns.csv) instead of macro
factors. This is the mirror image of quant_model_macro_2.py: there the FF
factors were the endogenous variable and macro factors were the regressors;
here each asset class is the dependent variable and the FF factors are the
regressors.

For every asset class this fits two things on the same aligned sample:
  1. Multivariate: excess_return ~ const + Mkt-RF + SMB + HML + RMW + CMA
     (all five FF factors together) -> the asset's factor loadings/alpha
     net of the other factors, i.e. what a standard FF5 regression reports.
  2. Univariate: excess_return ~ const + <one FF factor at a time>
     -> a simple-baseline comparison against the multivariate betas: a
     factor that's significant alone but loses significance once the other
     factors are included is probably just correlated with another factor
     that's doing the real work, and vice versa (same logic as step 2).

Each asset class has its own start date (many only have history from the
2000s/2010s onward), so alignment/dropna and the HAC maxlags are computed
per asset, not once globally.

Inputs:
  - Data/asset_class_monthly_returns.csv  (Yahoo Finance monthly total returns, %)
  - Data/FF_plus_macro_workable.csv       (for the 5 FF factors + RF)

Outputs (Data/ folder):
  - asset_ff_multivariate_betas_tstats_long.csv  every (asset, term incl. Alpha) row: beta, t-stat, R2, N_Obs
  - asset_ff_multivariate_betas_wide.csv         same, pivoted: rows=asset, cols=term, values=beta
  - asset_ff_multivariate_tstats_wide.csv        same, pivoted, values=t-stat
  - asset_ff_univariate_betas_tstats_long.csv    every (asset, FF factor) pair: beta, t-stat, R2, N_Obs
  - asset_ff_univariate_betas_wide.csv           same, pivoted: rows=asset, cols=FF factor, values=beta
  - asset_ff_univariate_tstats_wide.csv          same, pivoted, values=t-stat

Both univariate and multivariate regressions use HAC (Newey-West) standard
errors, per-asset maxlags via the standard rule of thumb
floor(4*(T/100)^(2/9)), since monthly return series are typically serially
correlated.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

ASSET_PATH = r"C:\Users\ak\OneDrive\Desktop\Stuff\Python projects\Data\asset_class_monthly_returns.csv"
FF_PATH = r"C:\Users\ak\OneDrive\Desktop\Stuff\Python projects\Data\FF_plus_macro_workable.csv"
OUT_DIR = r"C:\Users\ak\OneDrive\Desktop\Stuff\Python projects\Data"

FF_FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]  # RF excluded here too, it's used to build excess returns
MIN_OBS = 24  # skip assets with too short a history for a meaningful regression


def hac_maxlags(n_obs):
    return int(np.floor(4 * (n_obs / 100) ** (2 / 9)))


if __name__ == "__main__":
    # --- Load ---
    assets = pd.read_csv(ASSET_PATH, parse_dates=["Date"]).set_index("Date")
    ff = pd.read_csv(FF_PATH, parse_dates=["Date"]).set_index("Date")[FF_FACTOR_COLS + ["RF"]]

    asset_cols = list(assets.columns)
    print(f"Asset classes: {len(asset_cols)}")
    print(f"FF factors (regressors): {FF_FACTOR_COLS}\n")

    multi_rows = []
    uni_rows = []
    skipped = []

    for asset_col in asset_cols:
        # Each asset has its own history; align + drop NaNs per asset so the
        # sample is as long as the data allows, not clipped to the shortest asset.
        data = ff.join(assets[[asset_col]], how="inner").dropna()
        n_obs = len(data)
        if n_obs < MIN_OBS:
            skipped.append((asset_col, n_obs))
            continue

        excess = data[asset_col] - data["RF"]
        maxlags = hac_maxlags(n_obs)

        # --- Multivariate: all 5 FF factors together ---
        x_multi = sm.add_constant(data[FF_FACTOR_COLS])
        multi_result = sm.OLS(excess, x_multi).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        for term in ["const"] + FF_FACTOR_COLS:
            multi_rows.append({
                "Asset": asset_col,
                "Term": "Alpha" if term == "const" else term,
                "Beta": multi_result.params[term],
                "T_Stat": multi_result.tvalues[term],
                "R2": multi_result.rsquared,
                "N_Obs": n_obs,
            })

        # --- Univariate: one FF factor at a time ---
        for ff_col in FF_FACTOR_COLS:
            x_uni = sm.add_constant(data[ff_col])
            uni_result = sm.OLS(excess, x_uni).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
            uni_rows.append({
                "Asset": asset_col,
                "FF_Factor": ff_col,
                "Beta": uni_result.params[ff_col],
                "T_Stat": uni_result.tvalues[ff_col],
                "R2": uni_result.rsquared,
                "N_Obs": n_obs,
            })

    if skipped:
        print("Skipped (fewer than MIN_OBS={} aligned observations):".format(MIN_OBS))
        for asset_col, n_obs in skipped:
            print(f"  {asset_col}: {n_obs} obs")
        print()

    # --- Multivariate outputs ---
    TERM_ORDER = ["Alpha"] + FF_FACTOR_COLS
    multi_long = pd.DataFrame(multi_rows)
    multi_beta_wide = multi_long.pivot(index="Asset", columns="Term", values="Beta")[TERM_ORDER]
    multi_t_wide = multi_long.pivot(index="Asset", columns="Term", values="T_Stat")[TERM_ORDER]

    print("=== Multivariate (FF5) betas ===")
    print(multi_beta_wide.round(3))
    print("\n=== Multivariate (FF5) t-stats ===")
    print(multi_t_wide.round(2))

    multi_long.to_csv(rf"{OUT_DIR}\asset_ff_multivariate_betas_tstats_long.csv", index=False)
    multi_beta_wide.to_csv(rf"{OUT_DIR}\asset_ff_multivariate_betas_wide.csv")
    multi_t_wide.to_csv(rf"{OUT_DIR}\asset_ff_multivariate_tstats_wide.csv")

    # --- Univariate outputs ---
    uni_long = pd.DataFrame(uni_rows)
    uni_beta_wide = uni_long.pivot(index="Asset", columns="FF_Factor", values="Beta")[FF_FACTOR_COLS]
    uni_t_wide = uni_long.pivot(index="Asset", columns="FF_Factor", values="T_Stat")[FF_FACTOR_COLS]

    print("\n=== Univariate betas ===")
    print(uni_beta_wide.round(3))
    print("\n=== Univariate t-stats ===")
    print(uni_t_wide.round(2))

    uni_long.to_csv(rf"{OUT_DIR}\asset_ff_univariate_betas_tstats_long.csv", index=False)
    uni_beta_wide.to_csv(rf"{OUT_DIR}\asset_ff_univariate_betas_wide.csv")
    uni_t_wide.to_csv(rf"{OUT_DIR}\asset_ff_univariate_tstats_wide.csv")

    print(f"\nSaved:")
    print(f"  {OUT_DIR}\\asset_ff_multivariate_betas_tstats_long.csv")
    print(f"  {OUT_DIR}\\asset_ff_multivariate_betas_wide.csv")
    print(f"  {OUT_DIR}\\asset_ff_multivariate_tstats_wide.csv")
    print(f"  {OUT_DIR}\\asset_ff_univariate_betas_tstats_long.csv")
    print(f"  {OUT_DIR}\\asset_ff_univariate_betas_wide.csv")
    print(f"  {OUT_DIR}\\asset_ff_univariate_tstats_wide.csv")
