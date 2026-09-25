"""
quant_model_macro_1.py

Step 1 of the macro-factor model: categorize the FRED series we pulled in
fetch_macro_fred.py into economic channels, transform each series to a
stationary form, then run PCA *within each category* to collapse the
correlated raw series (growth cluster, inflation cluster, etc.) into a
small number of orthogonal macro factors.

Why PCA per category instead of one PCA across all 27 series: the whole
panel mixes growth, inflation, policy, currency, credit and commodity
signals. A single PCA would just find "the biggest common cyclical wobble"
(which drowns out inflation- or credit-specific variation). Running PCA
within each economically-meaningful group first gives you one clean
"Growth factor," one "Inflation factor," etc. — the reduced set an
explicit FAVAR-style setup wants, per the multicollinearity note from
earlier. A second-stage PCA across the per-category factors (or just using
them directly as regressors) is the natural next script.
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA_PATH = r"C:\Users\ak\OneDrive\Desktop\Stuff\Python projects\Data\FF_plus_macro_workable.csv"
OUT_PATH = r"C:\Users\ak\OneDrive\Desktop\Stuff\Python projects\Data\macro_pca_factors.csv"

# ---------------------------------------------------------------------------
# 1. Categorize the macro columns into economic channels.
# ---------------------------------------------------------------------------
CATEGORIES = {
    # Split US vs Global: US_INDPRO/OECD_CLI_US moved together in the first
    # run (49.3% PC1) while EA/China loaded weaker or negative — that's two
    # distinct growth cycles, not one.
    "US_Growth": [
        "US_INDPRO",
        "OECD_CLI_US",
    ],
    "Global_Growth": [
        "EA_INDPRO",
        "CHINA_INDPRO",
    ],
    "Inflation": [
        "US_CPI",
        "US_CORE_CPI",
        "EA_HICP",
        "CHINA_CPI",
    ],
    "Rates_Yields": [
        "US_10Y",
        "DE_10Y",
        "JP_10Y",
        "US_10Y_REAL",
        "US_BREAKEVEN_5Y5Y",
        "US_BREAKEVEN_10Y",
    ],
    # Split US vs Global: policy rates (Fed/ECB/BOJ) clustered fine in the
    # first run but M2 growth didn't join them (31.7% PC1 for the blended
    # group) — separating rates-setters by region keeps each factor coherent.
    "US_Monetary_Policy": [
        "FED_FUNDS",
        "FED_BALANCE_SHEET",
        "US_M2",
    ],
    "Global_Monetary_Policy": [
        "ECB_DEPO_RATE",
        "JP_DISCOUNT_RATE",
        "EA_M2",
        "CHINA_M2",
    ],
    "Currency": [
        "DXY_BROAD",
    ],
    # NOTE: this is a US-only risk-premia factor (BAA spread + VIX are both
    # US series). We don't have a clean non-US equivalent — the Euro HY OAS
    # series is capped to ~3 years by FRED's API (same licensing issue noted
    # in fetch_macro_fred.py), and there's no free long-history EU/global
    # vol index (VSTOXX/MOVE aren't on FRED). Not split for that reason.
    "Credit_Risk": [
        "US_CREDIT_SPREAD_BAA10Y",
        "VIX",
    ],
    "Commodities": [
        "BRENT_OIL",
        "COPPER",
    ],
}

# ---------------------------------------------------------------------------
# 2. How to transform each column before PCA. Raw index/price levels are
#    non-stationary (trend-dominated) and would swamp a PCA on levels, so
#    those go in as month-over-month % change. Rates, spreads, and yields
#    are already in stationary percentage-point units and are left as
#    levels (PCA runs on their z-scored level).
# ---------------------------------------------------------------------------
PCT_CHANGE_COLS = {
    "US_INDPRO", "OECD_CLI_US", "CHINA_INDPRO",
    "US_CPI", "US_CORE_CPI", "EA_HICP", "CHINA_CPI",
    "FED_BALANCE_SHEET", "US_M2", "EA_M2", "CHINA_M2",
    "DXY_BROAD", "BRENT_OIL", "COPPER",
}
LEVEL_COLS = {
    "US_10Y", "DE_10Y", "JP_10Y", "US_10Y_REAL",
    "US_BREAKEVEN_5Y5Y", "US_BREAKEVEN_10Y",
    "FED_FUNDS", "ECB_DEPO_RATE", "JP_DISCOUNT_RATE",
    "US_CREDIT_SPREAD_BAA10Y", "VIX",
    "EA_INDPRO",  # already a YoY growth rate from FRED, not a level index
}


def transform_column(df: pd.DataFrame, col: str) -> pd.Series:
    if col in PCT_CHANGE_COLS:
        return df[col].pct_change() * 100
    elif col in LEVEL_COLS:
        return df[col]
    else:
        raise ValueError(f"Column {col} not classified as pct_change or level")


def run_category_pca(df: pd.DataFrame, cols: list[str], n_components: int = 1):
    """Standardize + PCA a single category's columns. Returns (scores_df,
    loadings_df, explained_variance_ratio)."""
    transformed = pd.DataFrame({c: transform_column(df, c) for c in cols}, index=df.index)
    transformed = transformed.dropna()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(transformed)

    n_components = min(n_components, transformed.shape[1])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(scaled)

    scores_df = pd.DataFrame(
        scores,
        index=transformed.index,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=cols,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    return scores_df, loadings_df, pca.explained_variance_ratio_


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).set_index("Date")

    all_scores = {}
    print("=== Category PCA summary ===\n")

    for category, cols in CATEGORIES.items():
        cols_available = [c for c in cols if c in df.columns]
        if len(cols_available) < 2:
            print(f"[{category}] skipped PCA (only {len(cols_available)} series: {cols_available}); "
                  f"using it directly as the factor.\n")
            if cols_available:
                col = cols_available[0]
                factor_name = f"{category}_Factor"
                all_scores[factor_name] = transform_column(df, col)
            continue

        scores_df, loadings_df, explained_var = run_category_pca(df, cols_available, n_components=1)
        factor_name = f"{category}_Factor"
        all_scores[factor_name] = scores_df["PC1"]

        print(f"[{category}] series: {cols_available}")
        print(f"  PC1 explains {explained_var[0]:.1%} of variance in this group")
        print(f"  Loadings (sign/magnitude shows how each series drives the factor):")
        for series_name, loading in loadings_df["PC1"].items():
            print(f"    {series_name:<28} {loading:+.3f}")
        print()

    macro_factors = pd.DataFrame(all_scores)
    macro_factors.index.name = "Date"
    macro_factors = macro_factors.dropna(how="all")
    macro_factors.to_csv(OUT_PATH)

    print(f"Saved category-level macro PCA factors -> {OUT_PATH}")
    print(f"Shape: {macro_factors.shape}")
    print(macro_factors.head())
    print(macro_factors.tail())
