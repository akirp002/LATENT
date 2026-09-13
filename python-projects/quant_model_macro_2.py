"""
quant_model_macro_2.py

Step 2 of the macro-factor model: a VAR regressing the Fama-French 5 factor
returns (endogenous, with their own lags) on the category-level macro PCA
factors from quant_model_macro_1.py (exogenous regressors).

This is a VARX(p): each FF factor is modeled as a function of its own and
the other FF factors' lagged values, plus the macro factors (contemporaneous
+ implicitly whatever lag structure you choose to feed in). Lag order p is
chosen automatically via AIC over a max lag search.

Inputs:
  - Data/FF_plus_macro_workable.csv  (for the 5 FF factor return columns)
  - Data/macro_pca_factors.csv       (9 macro PCA factors from step 1)

Also runs a series of univariate OLS regressions — each FF factor on one
macro factor at a time (no lags, no other regressors) — as a simple-baseline
comparison against the multivariate VAR betas/t-stats: a macro factor that's
significant on its own but loses significance in the VAR is probably just
correlated with another macro factor that's doing the real work, and vice
versa.

Outputs (Data/ folder):
  - var_model_summary.txt          full statsmodels VAR summary (readable)
  - var_coefficients.csv           every coefficient (both endog-lag and exog terms)
  - var_fitted_values.csv          in-sample fitted values per FF factor
  - var_residuals.csv              in-sample residuals per FF factor
  - var_model.pkl                  pickled fitted VARResultsWrapper for reuse
  - univariate_betas_tstats_long.csv  every (macro factor, FF factor) pair: beta, t-stat, R2
  - univariate_betas_wide.csv         same, pivoted: rows=macro factor, cols=FF factor, values=beta
  - univariate_tstats_wide.csv        same, pivoted, values=t-stat
"""

import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

FF_PATH = r"C:\Users\ak\OneDrive\Desktop\Stuff\Python projects\Data\FF_plus_macro_workable.csv"
MACRO_PATH = r"C:\Users\ak\OneDrive\Desktop\Stuff\Python projects\Data\macro_pca_factors.csv"
OUT_DIR = r"C:\Users\ak\OneDrive\Desktop\Stuff\Python projects\Data"

FF_FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]  # RF excluded, it's the risk-free rate, not a factor spread
MAX_LAGS = 6


if __name__ == "__main__":
    # --- Load and align ---
    ff = pd.read_csv(FF_PATH, parse_dates=["Date"]).set_index("Date")[FF_FACTOR_COLS]
    macro = pd.read_csv(MACRO_PATH, parse_dates=["Date"]).set_index("Date")

    data = ff.join(macro, how="inner").dropna()
    print(f"Aligned sample: {data.index.min().date()} -> {data.index.max().date()} "
          f"({len(data)} monthly observations)")

    endog = data[FF_FACTOR_COLS]
    exog = data.drop(columns=FF_FACTOR_COLS)
    print(f"Endogenous (FF factors): {list(endog.columns)}")
    print(f"Exogenous (macro factors): {list(exog.columns)}\n")

    # --- Select lag order by AIC ---
    model = VAR(endog=endog, exog=exog)
    lag_selection = model.select_order(maxlags=MAX_LAGS)
    print(lag_selection.summary())
    chosen_lag = lag_selection.aic
    if chosen_lag == 0:
        chosen_lag = 1  # VAR needs at least 1 lag to be a VAR rather than plain OLS
    print(f"\nChosen lag order (AIC): {chosen_lag}\n")

    # --- Fit ---
    results = model.fit(chosen_lag)
    print(results.summary())

    # --- Save outputs ---
    with open(rf"{OUT_DIR}\var_model_summary.txt", "w") as f:
        f.write(str(results.summary()))

    # Coefficients: statsmodels stores as a DataFrame (rows = regressors incl.
    # lags/exog/const, columns = each endogenous FF factor)
    results.params.to_csv(rf"{OUT_DIR}\var_coefficients.csv")

    fitted = pd.DataFrame(results.fittedvalues, columns=FF_FACTOR_COLS)
    fitted.to_csv(rf"{OUT_DIR}\var_fitted_values.csv")

    resid = pd.DataFrame(results.resid, columns=FF_FACTOR_COLS)
    resid.to_csv(rf"{OUT_DIR}\var_residuals.csv")

    with open(rf"{OUT_DIR}\var_model.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved:")
    print(f"  {OUT_DIR}\\var_model_summary.txt")
    print(f"  {OUT_DIR}\\var_coefficients.csv")
    print(f"  {OUT_DIR}\\var_fitted_values.csv")
    print(f"  {OUT_DIR}\\var_residuals.csv")
    print(f"  {OUT_DIR}\\var_model.pkl  (pickled fitted model, reload with pickle.load)")

    # -------------------------------------------------------------------
    # Univariate regressions: for every (macro factor, FF factor) pair,
    # fit a simple contemporaneous OLS with just that one regressor + a
    # constant. Same aligned sample as the VAR, so results are comparable.
    # SEs are HAC (Newey-West) since monthly macro/factor series are
    # typically serially correlated; maxlags via the standard NW rule of
    # thumb floor(4*(T/100)^(2/9)).
    # -------------------------------------------------------------------
    print("\n=== Univariate regressions (one macro factor at a time) ===\n")

    hac_maxlags = int(np.floor(4 * (len(data) / 100) ** (2 / 9)))
    print(f"HAC (Newey-West) maxlags: {hac_maxlags}")

    macro_cols = list(exog.columns)
    rows = []
    for macro_col in macro_cols:
        x = sm.add_constant(data[macro_col])
        for ff_col in FF_FACTOR_COLS:
            y = data[ff_col]
            uni_result = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": hac_maxlags})
            rows.append({
                "Macro_Factor": macro_col,
                "FF_Factor": ff_col,
                "Beta": uni_result.params[macro_col],
                "T_Stat": uni_result.tvalues[macro_col],
                "R2": uni_result.rsquared,
            })

    uni_long = pd.DataFrame(rows)
    uni_beta_wide = uni_long.pivot(index="Macro_Factor", columns="FF_Factor", values="Beta")[FF_FACTOR_COLS]
    uni_t_wide = uni_long.pivot(index="Macro_Factor", columns="FF_Factor", values="T_Stat")[FF_FACTOR_COLS]
    uni_r2_wide = uni_long.pivot(index="Macro_Factor", columns="FF_Factor", values="R2")[FF_FACTOR_COLS]

    print("Univariate betas:")
    print(uni_beta_wide.round(3))
    print("\nUnivariate t-stats:")
    print(uni_t_wide.round(2))
    print("\nUnivariate R2:")
    print(uni_r2_wide.round(3))

    uni_long.to_csv(rf"{OUT_DIR}\univariate_betas_tstats_long.csv", index=False)
    uni_beta_wide.to_csv(rf"{OUT_DIR}\univariate_betas_wide.csv")
    uni_t_wide.to_csv(rf"{OUT_DIR}\univariate_tstats_wide.csv")

    print(f"\nSaved:")
    print(f"  {OUT_DIR}\\univariate_betas_tstats_long.csv")
    print(f"  {OUT_DIR}\\univariate_betas_wide.csv")
    print(f"  {OUT_DIR}\\univariate_tstats_wide.csv")
