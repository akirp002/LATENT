"""
vif_check.py

Compute Variance Inflation Factors (VIF) across the full 14-factor set:
5 Fama-French factors (Mkt-RF, SMB, HML, RMW, CMA) + 9 macro PCA factors
(one per economic category from quant_model_macro_1.py).

For each factor, VIF = 1 / (1 - R^2) where R^2 comes from regressing that
factor on the other 13. High VIF (>5, or >10 stricter) flags a factor that's
largely redundant given the rest of the set -- a candidate for the
second-stage PCA / combination noted in quant_model_macro_1.py's docstring.
"""

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

FF_PATH = r"C:\Users\ak\OneDrive\Desktop\Stuff\Python projects\Data\FF_plus_macro_workable.csv"
MACRO_PCA_PATH = r"C:\Users\ak\OneDrive\Desktop\Stuff\Python projects\Data\macro_pca_factors.csv"

FF_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

VIF_THRESHOLD_YELLOW = 5.0
VIF_THRESHOLD_RED = 10.0


def load_factor_panel() -> pd.DataFrame:
    ff = pd.read_csv(FF_PATH, parse_dates=["Date"]).set_index("Date")[FF_FACTORS]
    macro = pd.read_csv(MACRO_PCA_PATH, parse_dates=["Date"]).set_index("Date")

    panel = ff.join(macro, how="inner")
    panel = panel.dropna()  # VIF requires a complete design matrix
    return panel


def compute_vif(panel: pd.DataFrame) -> pd.DataFrame:
    # add_constant so the auxiliary regressions include an intercept --
    # standard practice for VIF, otherwise VIFs are inflated/misleading.
    X = add_constant(panel)

    rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        vif = variance_inflation_factor(X.values, i)
        rows.append({"factor": col, "VIF": vif})

    vif_df = pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)
    return vif_df


def flag(vif: float) -> str:
    if vif > VIF_THRESHOLD_RED:
        return "RED (>10)"
    elif vif > VIF_THRESHOLD_YELLOW:
        return "yellow (>5)"
    return ""


if __name__ == "__main__":
    panel = load_factor_panel()
    print(f"Panel shape after aligning FF + macro PCA factors and dropping NaNs: {panel.shape}")
    print(f"Date range: {panel.index.min().date()} to {panel.index.max().date()}\n")

    vif_df = compute_vif(panel)

    print("=== VIF across all 14 factors ===\n")
    for _, row in vif_df.iterrows():
        print(f"  {row['factor']:<28} VIF = {row['VIF']:6.2f}   {flag(row['VIF'])}")

    n_red = (vif_df["VIF"] > VIF_THRESHOLD_RED).sum()
    n_yellow = ((vif_df["VIF"] > VIF_THRESHOLD_YELLOW) & (vif_df["VIF"] <= VIF_THRESHOLD_RED)).sum()
    print(f"\n{n_red} factor(s) above VIF 10, {n_yellow} factor(s) between 5 and 10.")

    print("\n=== Correlation matrix (for identifying WHICH factors collide) ===")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(panel.corr().round(2))
