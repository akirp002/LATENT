"""
quant_model_macro_4.py

Step 4: extend quant_model_macro_3.py's multivariate regression from 5 FF
factors to the full 14-factor set (5 FF + 9 macro PCA factors from
quant_model_macro_1.py), and for every asset class report BOTH plain-OLS
and HAC (Newey-West) t-stats side by side.

Why: quant_model_macro_3.py already used HAC SEs, but only regressed on
the 5 FF factors. The point of this script is narrower -- isolate how much
of the "poor t-stats despite low VIF" picture is explained by
autocorrelation/heteroskedasticity in monthly returns (which plain OLS SEs
ignore) versus how much is just genuine estimation noise/weak signal that
HAC correction doesn't fix. VIF already ruled out collinearity as the cause
(quant_model_macro_1.py's factors all had VIF < 2.2), so this isolates the
next candidate explanation.

Inputs:
  - Data/asset_class_monthly_returns.csv
  - Data/FF_plus_macro_workable.csv   (5 FF factors + RF)
  - Data/macro_pca_factors.csv        (9 macro PCA factors)

Output (Data/ folder):
  - asset_full14_betas_tstats_long.csv   every (asset, term) row:
      beta, t_stat_ols, t_stat_hac, r2, n_obs
  - asset_full14_tstats_wide.csv         pivoted, HAC t-stats, rows=asset cols=term
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

ASSET_PATH = r"C:\Users\ak\OneDrive\Desktop\Stuff\Python projects\Data\asset_class_monthly_returns.csv"
FF_PATH = r"C:\Users\ak\OneDrive\Desktop\Stuff\Python projects\Data\FF_plus_macro_workable.csv"
MACRO_PCA_PATH = r"C:\Users\ak\OneDrive\Desktop\Stuff\Python projects\Data\macro_pca_factors.csv"
OUT_DIR = r"C:\Users\ak\OneDrive\Desktop\Stuff\Python projects\Data"

FF_FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
MIN_OBS = 40  # need more slack than the FF5-only script: 14 regressors now, not 5


def hac_maxlags(n_obs):
    return int(np.floor(4 * (n_obs / 100) ** (2 / 9)))


if __name__ == "__main__":
    assets = pd.read_csv(ASSET_PATH, parse_dates=["Date"]).set_index("Date")
    ff = pd.read_csv(FF_PATH, parse_dates=["Date"]).set_index("Date")[FF_FACTOR_COLS + ["RF"]]
    macro = pd.read_csv(MACRO_PCA_PATH, parse_dates=["Date"]).set_index("Date")

    factors = ff.join(macro, how="inner")
    FACTOR_COLS = FF_FACTOR_COLS + list(macro.columns)

    asset_cols = list(assets.columns)
    print(f"Asset classes: {len(asset_cols)}")
    print(f"Regressors ({len(FACTOR_COLS)}): {FACTOR_COLS}\n")

    rows = []
    skipped = []

    for asset_col in asset_cols:
        data = factors.join(assets[[asset_col]], how="inner").dropna()
        n_obs = len(data)
        if n_obs < MIN_OBS:
            skipped.append((asset_col, n_obs))
            continue

        excess = data[asset_col] - data["RF"]
        maxlags = hac_maxlags(n_obs)

        x = sm.add_constant(data[FACTOR_COLS])
        ols_result = sm.OLS(excess, x).fit()  # plain (non-robust) SEs
        hac_result = sm.OLS(excess, x).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

        for term in ["const"] + FACTOR_COLS:
            rows.append({
                "Asset": asset_col,
                "Term": "Alpha" if term == "const" else term,
                "Beta": hac_result.params[term],
                "T_Stat_OLS": ols_result.tvalues[term],
                "T_Stat_HAC": hac_result.tvalues[term],
                "R2": hac_result.rsquared,
                "N_Obs": n_obs,
            })

    if skipped:
        print(f"Skipped (fewer than MIN_OBS={MIN_OBS} aligned observations):")
        for asset_col, n_obs in skipped:
            print(f"  {asset_col}: {n_obs} obs")
        print()

    long = pd.DataFrame(rows)
    TERM_ORDER = ["Alpha"] + FACTOR_COLS

    t_wide_hac = long.pivot(index="Asset", columns="Term", values="T_Stat_HAC")[TERM_ORDER]
    t_wide_ols = long.pivot(index="Asset", columns="Term", values="T_Stat_OLS")[TERM_ORDER]

    long.to_csv(rf"{OUT_DIR}\asset_full14_betas_tstats_long.csv", index=False)
    t_wide_hac.to_csv(rf"{OUT_DIR}\asset_full14_tstats_wide.csv")

    # --- Summary: how many |t| > 2 under OLS vs HAC, per factor ---
    print("=== Count of assets with |t| > 2, OLS vs HAC (per factor) ===\n")
    summary = pd.DataFrame({
        "Sig_OLS": (t_wide_ols.abs() > 2).sum(),
        "Sig_HAC": (t_wide_hac.abs() > 2).sum(),
        "N_Assets": t_wide_ols.notna().sum(),
    })
    print(summary)

    # --- Flag cases where OLS said significant but HAC disagrees, or vice versa ---
    flip_to_insig = ((t_wide_ols.abs() > 2) & (t_wide_hac.abs() <= 2))
    flip_to_sig = ((t_wide_ols.abs() <= 2) & (t_wide_hac.abs() > 2))
    n_flip_to_insig = flip_to_insig.sum().sum()
    n_flip_to_sig = flip_to_sig.sum().sum()
    print(f"\nOLS significant -> HAC insignificant: {n_flip_to_insig} (asset, factor) pairs")
    print(f"OLS insignificant -> HAC significant: {n_flip_to_sig} (asset, factor) pairs")

    print(f"\nSaved:")
    print(f"  {OUT_DIR}\\asset_full14_betas_tstats_long.csv")
    print(f"  {OUT_DIR}\\asset_full14_tstats_wide.csv")
