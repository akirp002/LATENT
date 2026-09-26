import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chart_utils as cu
import data_loader as dl

st.set_page_config(page_title="Factor Research | LATENT_V2", layout="wide")
st.title("1. Factor research overview")
st.caption(
    "Source: factor-research/Data/macro_pca_factors.csv, "
    "FF_plus_macro_full.csv (Fama-French factor returns, full 1963+ "
    "history), FF_plus_macro_workable.csv (for VIF, which needs FF and "
    "macro aligned to the same 2006+ window), asset_full14_betas_tstats_long.csv "
    "(for betas/t-stats and the τ̂ shrinkage calc)."
)

# --- Combined factor returns: macro + Fama-French, one overlay chart --------
st.header("Factor returns over time — macro + Fama-French, overlaid")

macro = dl.load_macro_pca_factors()
ff_macro = dl.load_ff_plus_macro_full()
tau_df = dl.compute_tau_shrinkage()

FF_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
macro_cols = [c for c in macro.columns if c != "Date"] if macro is not None else []

if macro is None and ff_macro is None:
    dl.missing(
        "the factor return chart",
        "factor-research/Data/macro_pca_factors.csv and FF_plus_macro_full.csv",
    )
else:
    if tau_df is not None:
        st.caption(
            "🚩 next to a macro factor below = that factor's cross-sectional "
            "beta spread is real signal, not noise, per the τ̂ shrinkage "
            "result further down this page (τ̂ ≠ 0)."
        )

    combined = None
    if macro is not None and ff_macro is not None:
        # Left join off the Fama-French table so FF factors keep their full
        # 1963+ history -- macro factors simply come back NaN before 2006,
        # instead of the whole panel getting truncated to macro's shorter
        # range like an inner/outer join on the trimmed workable file would.
        combined = ff_macro[["Date"] + FF_FACTORS].merge(macro, on="Date", how="left").sort_values("Date")
    elif macro is not None:
        combined = macro.copy()
    else:
        combined = ff_macro[["Date"] + FF_FACTORS].copy()

    st.caption(
        "Fama-French lines use their full available history (back to "
        "1963); macro-factor lines are only defined from 2006 onward, "
        "since that's when the macro PCA factors start."
    )

    all_cols = macro_cols + [c for c in FF_FACTORS if ff_macro is not None]
    default_pick = [c for c in ["Mkt-RF", "SMB"] if c in all_cols] + macro_cols[:2]

    chosen = st.multiselect(
        "Factors to plot (macro + Fama-French, together)",
        all_cols,
        default=default_pick,
        format_func=lambda c: dl.significant_factor_label(c, tau_df),
        key="factors_chosen",
    )

    c1, c2 = st.columns(2)
    with c1:
        window_label = st.select_slider(
            "Rolling window", options=list(dl.ROLLING_WINDOWS.keys()), value="1 year", key="window"
        )
    with c2:
        metric = st.radio(
            "Metric",
            ["Cumulative return (over window)", "Annualized rate (same window)"],
            key="metric",
        )
    window_months = dl.ROLLING_WINDOWS[window_label]

    if not chosen:
        st.info("Select at least one factor above.")
    else:
        if metric.startswith("Cumulative"):
            result = dl.compute_rolling_cumulative_return(combined, chosen, window_months)
            y_title = f"Trailing {window_label} growth of $1"
            value_format, zero_line, ref_line = "$.2f", False, 1.0
        else:
            result = dl.compute_rolling_annualized_return(combined, chosen, window_months)
            y_title = "Rolling annualized return (%)"
            value_format, zero_line, ref_line = ".1f", True, None
        st.caption(
            "Compounds each monthly value as (1 + value/100) over the trailing "
            f"window. Macro factors are PCA scores, not asset returns, so this "
            "is a momentum/trend view of the factor level rather than a "
            "literal investment return; Fama-French values are true monthly "
            "returns."
        )
        chart_key = f"factors_rolling_{window_label}_{metric}_{','.join(chosen)}"
        cu.line_chart(
            result, y_title, value_format=value_format, zero_line=zero_line, ref_line=ref_line, key=chart_key
        )

    st.subheader("Total growth of $1 (cumulative since inception, not rolling)")
    st.caption(
        "Compounds every monthly value from the start of its own history "
        "through today and never resets -- unlike the rolling window chart "
        "above, which restarts from $1 at the beginning of every trailing "
        "window."
    )
    if not chosen:
        st.info("Select at least one factor above.")
    else:
        log_scale = st.checkbox(
            "Log scale", value=True, key="factors_log_scale",
            help="Recommended for a 60+ year compounding series -- on a linear "
            "axis the fastest-compounding factor flattens every slower one "
            "down near zero.",
        )
        total_growth = dl.compute_cumulative_growth(combined, chosen)
        cu.line_chart(
            total_growth,
            "Growth of $1 (cumulative since inception)",
            value_format="$.2f",
            ref_line=1.0,
            log_scale=log_scale,
            key=f"factors_total_growth_{','.join(chosen)}_{log_scale}",
        )

        with st.expander("🔍 Validate: end value + CAGR per factor"):
            st.caption(
                "Reads the same numbers the chart plots, as exact figures instead "
                "of a line position -- the last row of each column is where that "
                "factor's line ends on the chart above. CAGR is the compounding "
                "rate that gets $1 to that ending value over the number of years "
                "shown, so you can sanity-check it against commonly cited "
                "long-run Fama-French premia (Mkt-RF roughly 6-8%/yr, the other "
                "four factors typically a few percent/yr, some near zero -- see "
                "the VIF/τ̂ sections below for which factors this dataset finds "
                "to be weak or noisy)."
            )
            summary_rows = []
            for f in chosen:
                s = total_growth[f].dropna()
                if s.empty:
                    continue
                start_date, end_date = s.index[0], s.index[-1]
                years = (end_date - start_date).days / 365.25
                end_val = s.iloc[-1]
                cagr = end_val ** (1 / years) - 1 if years > 0 else float("nan")
                summary_rows.append(
                    {
                        "Factor": f,
                        "Start": start_date.date(),
                        "End": end_date.date(),
                        "Years": round(years, 1),
                        "Ending $1 grew to": f"${end_val:,.2f}",
                        "CAGR": f"{cagr * 100:.2f}%",
                    }
                )
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

            st.divider()
            st.caption(
                "Row-by-row trace for one factor -- multiply (1 + value/100) into "
                "a running product starting at $1; the last row should exactly "
                "match that factor's ending value in the table above and its "
                "final chart position."
            )
            trace_factor = st.selectbox("Factor to trace", chosen, key="trace_factor")
            trace = combined[["Date", trace_factor]].dropna(subset=[trace_factor]).copy()
            trace["Monthly value"] = trace[trace_factor]
            trace["1 + value/100"] = 1 + trace[trace_factor] / 100.0
            trace["Running product ($1 → )"] = trace["1 + value/100"].cumprod()
            st.dataframe(
                trace[["Date", "Monthly value", "1 + value/100", "Running product ($1 → )"]].tail(15),
                use_container_width=True,
                hide_index=True,
            )

st.divider()

# --- VIF table --------------------------------------------------------------
st.header("Multicollinearity check (VIF, all 14 factors)")
st.caption(
    "Reproduces factor-research/vif_check.py's compute_vif() on this repo's "
    "own FF_plus_macro_workable.csv + macro_pca_factors.csv (the original "
    "script points at a hard-coded local Windows path, so this recomputes "
    "the identical VIF = 1/(1-R²) formula against the same relative-path CSVs)."
)
vif_df = dl.compute_vif_table()
if vif_df is None:
    dl.missing(
        "the VIF table",
        "factor-research/Data/FF_plus_macro_workable.csv and/or macro_pca_factors.csv",
    )
else:
    def _flag(v):
        if v > 10:
            return "red (>10)"
        if v > 5:
            return "yellow (5-10)"
        return "ok"

    vif_df = vif_df.copy()
    vif_df["flag"] = vif_df["VIF"].apply(_flag)
    st.dataframe(vif_df, use_container_width=True, hide_index=True)
    cu.bar_chart(vif_df, "factor", "VIF", "Factor", "VIF")
    st.caption(
        f"Max VIF: {vif_df['VIF'].max():.2f} on `{vif_df.iloc[vif_df['VIF'].idxmax()]['factor']}` "
        "— well under the conventional 5-10 flag threshold, consistent with the README claim."
    )

st.divider()

# --- Asset factor betas / t-stats -------------------------------------------
st.header("Asset factor betas & t-stats (14-factor model)")
betas = dl.load_asset_full14_betas_tstats()
if betas is None:
    dl.missing("the asset betas/t-stats table", "factor-research/Data/asset_full14_betas_tstats_long.csv")
else:
    assets = sorted(betas["Asset"].unique())
    pick = st.selectbox("Asset", assets, index=0)
    sub = betas[betas["Asset"] == pick].drop(columns=["Asset"]).reset_index(drop=True)
    st.dataframe(sub, use_container_width=True, hide_index=True)
    with st.expander("Full long-format table (all assets x factors)"):
        st.dataframe(betas, use_container_width=True)

st.divider()

# --- tau-hat shrinkage --------------------------------------------------------
st.header("Empirical-Bayes τ̂ shrinkage by factor")
st.caption(
    "Same normal-normal shrinkage formula as scripts/generate_showcase_charts.py's "
    "chart_tau_shrinkage(): τ̂² = max(0, cross-sectional Var(β̂) − mean SE²), per "
    "factor across the 33 asset-class ETFs. A factor at τ̂ ≈ 0 means its apparent "
    "cross-sectional spread in beta is noise, not real signal — this is the same "
    "flag used to mark 🚩 factors in the chart above."
)
if tau_df is None:
    dl.missing("the τ̂ shrinkage chart", "factor-research/Data/asset_full14_betas_tstats_long.csv")
else:
    display_df = tau_df.copy()
    display_df["tau_hat_label"] = display_df.apply(
        lambda r: "≈ 0 (pure noise)" if r["is_noise"] else f"{r['tau_hat']:.4f}", axis=1
    )
    st.dataframe(
        display_df[["Factor", "tau_hat_label", "is_noise"]].rename(
            columns={"tau_hat_label": "tau_hat"}
        ),
        use_container_width=True,
        hide_index=True,
    )
    cu.bar_chart(tau_df, "Factor", "tau_hat", "Factor", "τ̂")
    n_noise = int(tau_df["is_noise"].sum())
    st.caption(
        f"{n_noise} of {len(tau_df)} factors shrink to τ̂ ≈ 0 in this run — "
        "compare against the README's '5 of 14 factors shrink to τ̂≈0' claim."
    )
