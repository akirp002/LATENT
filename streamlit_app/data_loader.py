"""
Shared data-loading helpers for the Streamlit app.

Every function here just reads a CSV/JSON/TXT that one of this repo's
pipeline scripts already wrote to Data/ -- nothing is re-derived except the
tau-hat shrinkage calc and VIF calc, which are copied verbatim from
scripts/generate_showcase_charts.py and factor-research/vif_check.py so the
numbers match the published showcase charts exactly.

All loaders return None (or an empty DataFrame) if the source file is
missing, so pages can show a "not available" message instead of crashing.
"""

import os
import re

import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FR_DATA = os.path.join(ROOT, "factor-research", "Data")
PB_DATA = os.path.join(ROOT, "portfolio-benchmarking", "Data")
COLLAR_DIR = os.path.join(ROOT, "portfolio-benchmarking", "demos", "collar-strategy")


def _read_csv(path, **kwargs):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        st.warning(f"Found `{os.path.relpath(path, ROOT)}` but could not parse it: {e}")
        return None


def _read_text(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        st.warning(f"Found `{os.path.relpath(path, ROOT)}` but could not read it: {e}")
        return None


def _read_json(path):
    if not os.path.exists(path):
        return None
    import json

    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Found `{os.path.relpath(path, ROOT)}` but could not parse it: {e}")
        return None


# ---------------------------------------------------------------------------
# factor-research/Data
# ---------------------------------------------------------------------------

@st.cache_data
def load_macro_pca_factors():
    return _read_csv(os.path.join(FR_DATA, "macro_pca_factors.csv"), parse_dates=["Date"])


@st.cache_data
def load_asset_full14_betas_tstats():
    return _read_csv(os.path.join(FR_DATA, "asset_full14_betas_tstats_long.csv"))


@st.cache_data
def load_ff_plus_macro_workable():
    return _read_csv(os.path.join(FR_DATA, "FF_plus_macro_workable.csv"), parse_dates=["Date"])


@st.cache_data
def load_ff_plus_macro_full():
    """Same Fama-French columns as FF_plus_macro_workable.csv, but back to
    1963 instead of being pre-trimmed to the macro PCA factors' 2006+
    range. Used so a Fama-French factor's own rolling-return history isn't
    truncated just because it's being charted alongside macro factors that
    don't exist before 2006."""
    return _read_csv(os.path.join(FR_DATA, "FF_plus_macro_full.csv"), parse_dates=["Date"])


ROLLING_WINDOWS = {
    "3 month": 3,
    "6 month": 6,
    "1 year": 12,
    "2 year": 24,
    "3 year": 36,
}


def _rolling_growth(df, cols, window_months, date_col="Date", pct_units=True):
    """Rolling `window_months`-month compounded growth factor, e.g. 1.05 for
    a +5% cumulative move. `pct_units=True` treats each value as a monthly
    percentage return (divides by 100 first); pass False for values already
    expressed as decimal fractions."""
    d = df.set_index(date_col)[cols]
    if pct_units:
        d = d / 100.0
    return (1 + d).rolling(window_months, min_periods=window_months).apply(np.prod, raw=True)


@st.cache_data
def compute_cumulative_growth(df, cols, date_col="Date", pct_units=True):
    """Growth of $1, compounding monthly values in `cols` since the start
    of each column's OWN history (each value treated as a monthly return).
    `fillna(0)` patches sporadic single-month gaps as "no change" so cumprod
    doesn't go NaN forever after one missing print, but it must not run
    before a column's first real observation -- e.g. a macro factor that
    only starts in 2006 must not show a fabricated flat $1.00 line back to
    1963 just because it's sharing a frame with Fama-French columns that do
    go back that far. So growth before each column's own first valid date
    is explicitly reset to NaN after the fact."""
    d = df.set_index(date_col)[cols]
    if pct_units:
        d = d / 100.0
    growth = (1 + d.fillna(0)).cumprod()
    for col in d.columns:
        first_valid = d[col].first_valid_index()
        if first_valid is None:
            growth[col] = np.nan
        else:
            growth.loc[growth.index < first_valid, col] = np.nan
    return growth


@st.cache_data
def compute_rolling_cumulative_return(df, cols, window_months, date_col="Date", pct_units=True):
    """Rolling `window_months`-month growth of $1 invested at the start of
    each trailing window, NOT annualized -- e.g. 1.05 means a dollar
    invested `window_months` months ago is worth $1.05 today; 0.92 means
    it's worth $0.92. Same recursive month-by-month compounding as
    compute_cumulative_growth, except it resets to $1 at the start of each
    window instead of running from the start of the whole series. Requires
    a full window of non-null data, so the first `window_months - 1` rows
    come back as NaN."""
    return _rolling_growth(df, cols, window_months, date_col, pct_units)


@st.cache_data
def compute_rolling_annualized_return(df, cols, window_months, date_col="Date", pct_units=True):
    """Rolling `window_months`-month cumulative return, annualized to a
    12-month rate: (cumulative growth) ** (12 / window_months) - 1. Returns
    percent units. Requires a full window of non-null data, so the first
    `window_months - 1` rows come back as NaN."""
    growth = _rolling_growth(df, cols, window_months, date_col, pct_units)
    annualized = growth ** (12.0 / window_months) - 1.0
    return annualized * 100.0


@st.cache_data
def compute_vif_table():
    """Reproduces vif_check.py's compute_vif() on this repo's own committed
    CSVs (the original script hard-codes a Windows path that doesn't exist
    here, so this rebuilds the same factor panel from the relative repo
    paths and calls the identical VIF formula: VIF = 1/(1-R^2))."""
    ff_macro = load_ff_plus_macro_workable()
    macro_pca = load_macro_pca_factors()
    if ff_macro is None or macro_pca is None:
        return None
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant

        FF_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        ff = ff_macro.set_index("Date")[FF_FACTORS]
        macro = macro_pca.set_index("Date")
        panel = ff.join(macro, how="inner").dropna()

        rows = []
        X_full = add_constant(panel)
        for i, col in enumerate(panel.columns):
            vif = variance_inflation_factor(X_full.values, i + 1)
            rows.append({"factor": col, "VIF": vif})
        vif_df = pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)
        return vif_df
    except Exception as e:
        st.warning(f"Could not compute VIF table: {e}")
        return None


@st.cache_data
def compute_tau_shrinkage():
    """Reproduces generate_showcase_charts.py's chart_tau_shrinkage() data
    prep exactly (same normal-normal shrinkage formula), returned as a
    DataFrame instead of a PNG."""
    df = load_asset_full14_betas_tstats()
    if df is None:
        return None
    df = df[df["Term"] != "Alpha"].copy()
    df["SE"] = (df["Beta"] / df["T_Stat_HAC"]).abs()

    rows = []
    for factor, g in df.groupby("Term"):
        beta_var = g["Beta"].var(ddof=1)
        mean_se2 = (g["SE"] ** 2).mean()
        tau2 = max(0.0, beta_var - mean_se2)
        rows.append({"Factor": factor, "tau_hat": np.sqrt(tau2), "is_noise": np.sqrt(tau2) < 1e-6})
    return pd.DataFrame(rows).sort_values("tau_hat", ascending=False).reset_index(drop=True)


def significant_factor_label(name, tau_df):
    """'🚩 US_Growth_Factor'-style label for macro-factor pickers: flags a
    factor whose cross-sectional beta spread is real signal rather than
    noise, per this page's own τ̂ shrinkage result (is_noise == False)."""
    if tau_df is None:
        return name
    row = tau_df[tau_df["Factor"] == name]
    if not row.empty and not bool(row.iloc[0]["is_noise"]):
        return f"🚩 {name}"
    return name


# ---------------------------------------------------------------------------
# portfolio-benchmarking/Data
# ---------------------------------------------------------------------------

@st.cache_data
def load_benchmark_wealth_curve():
    return _read_csv(os.path.join(PB_DATA, "benchmark_wealth_curve.csv"), parse_dates=["Date"])


@st.cache_data
def load_portfolio_active_returns():
    return _read_csv(os.path.join(PB_DATA, "portfolio_active_returns.csv"), parse_dates=["Date"])


@st.cache_data
def load_portfolio_factor_exposures():
    return _read_csv(os.path.join(PB_DATA, "portfolio_factor_exposures.csv"), index_col=0)


@st.cache_data
def load_tracking_error_vs_benchmark_txt():
    return _read_text(os.path.join(PB_DATA, "tracking_error_vs_benchmark.txt"))


@st.cache_data
def load_tracking_error_summary_txt():
    return _read_text(os.path.join(PB_DATA, "tracking_error_summary.txt"))


@st.cache_data
def load_benchmark_mvo_summary_txt():
    return _read_text(os.path.join(PB_DATA, "benchmark_mvo_summary.txt"))


@st.cache_data
def load_optimizer_summary_txt():
    return _read_text(os.path.join(PB_DATA, "optimizer_summary.txt"))


@st.cache_data
def load_path_metrics_json():
    return _read_json(os.path.join(PB_DATA, "path_metrics.json"))


@st.cache_data
def load_holdings_household_clean():
    return _read_csv(os.path.join(PB_DATA, "holdings_household_clean.csv"))


@st.cache_data
def load_trade_proposals(path_letter):
    fname = f"trade_proposals_path{path_letter}.csv"
    return _read_csv(os.path.join(PB_DATA, fname))


# ---------------------------------------------------------------------------
# portfolio-benchmarking/demos/collar-strategy
# ---------------------------------------------------------------------------

@st.cache_data
def load_collar_cvar_comparison():
    return _read_csv(os.path.join(COLLAR_DIR, "collar_cvar_comparison.csv"))


@st.cache_data
def load_collar_scenarios_comparison():
    return _read_csv(os.path.join(COLLAR_DIR, "collar_scenarios_comparison.csv"))


@st.cache_data
def load_saa_benchmark_te_comparison():
    return _read_csv(os.path.join(COLLAR_DIR, "saa_benchmark_te_comparison.csv"))


def parse_optimizer_summary_narrative(text):
    """optimizer_summary.txt embeds a monospace ASCII trade table after each
    path's headline metrics -- the same trades the page already renders as a
    proper st.dataframe further down, so showing both is a wall of
    duplicated text. This strips each path's embedded table (everything from
    its "Account ... Ticker ... Direction ..." header line onward) and keeps
    just the narrative/metrics lines."""
    if text is None:
        return None
    blocks = re.split(r"\n(?=Path [A-Za-z0-9]+ --)", text)
    cleaned = []
    for block in blocks:
        out_lines = []
        for line in block.split("\n"):
            stripped = line.strip()
            if stripped.startswith("Account") and "Ticker" in stripped and "Direction" in stripped:
                break
            out_lines.append(line)
        cleaned.append("\n".join(out_lines).rstrip())
    return "\n\n".join(cleaned)


def missing(label, relpath):
    st.info(f"**Not available:** `{relpath}` was not found in this checkout, so {label} cannot be shown.")
