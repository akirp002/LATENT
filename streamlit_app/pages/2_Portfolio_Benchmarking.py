import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chart_utils as cu
import data_loader as dl

st.set_page_config(page_title="Portfolio Benchmarking | LATENT_V2", layout="wide")
st.title("2. Portfolio benchmarking")
st.caption(
    "Source: portfolio-benchmarking/Data/benchmark_wealth_curve.csv, "
    "portfolio_active_returns.csv, portfolio_factor_exposures.csv."
)
st.info(
    "All holdings behind this page are the **fabricated sample portfolio** "
    "described in the repo README (two made-up accounts, one concentrated "
    "PLTR position) — not a real brokerage account."
)

# --- Benchmark wealth curve ---------------------------------------------------
st.header("Benchmark wealth curve (buy-and-hold)")
wealth = dl.load_benchmark_wealth_curve()
if wealth is None:
    dl.missing("the benchmark wealth curve", "portfolio-benchmarking/Data/benchmark_wealth_curve.csv")
else:
    cu.line_chart(
        wealth.set_index("Date")[["Benchmark_Wealth"]],
        "Wealth index ($1 start)",
        value_format="$.2f",
    )
    with st.expander("Raw benchmark_wealth_curve.csv"):
        st.dataframe(wealth, use_container_width=True)

st.divider()

# --- Portfolio vs. benchmark: rolling cumulative / annualized returns -------
st.header("Portfolio vs. benchmark: rolling return")
active = dl.load_portfolio_active_returns()
if active is None:
    dl.missing(
        "the portfolio-vs-benchmark return comparison",
        "portfolio-benchmarking/Data/portfolio_active_returns.csv",
    )
else:
    series_cols = [c for c in ["Portfolio_Full", "Portfolio_ExConcentrated", "Benchmark"] if c in active.columns]
    picked = st.multiselect(
        "Series to plot", series_cols, default=series_cols, key="pb_series"
    )
    c1, c2 = st.columns(2)
    with c1:
        window_label = st.select_slider(
            "Rolling window", options=list(dl.ROLLING_WINDOWS.keys()), value="1 year", key="pb_window"
        )
    with c2:
        metric = st.radio(
            "Metric",
            ["Cumulative return (over window)", "Annualized rate (same window)"],
            key="pb_metric",
        )
    window_months = dl.ROLLING_WINDOWS[window_label]

    if not picked:
        st.info("Select at least one series above.")
    else:
        if metric.startswith("Cumulative"):
            result = dl.compute_rolling_cumulative_return(
                active, picked, window_months, pct_units=False
            )
            y_title = f"Trailing {window_label} growth of $1"
            value_format, zero_line, ref_line = "$.2f", False, 1.0
        else:
            result = dl.compute_rolling_annualized_return(
                active, picked, window_months, pct_units=False
            )
            y_title = "Rolling annualized return (%)"
            value_format, zero_line, ref_line = ".1f", True, None
        st.caption(
            "Portfolio_Full/Portfolio_ExConcentrated/Benchmark are monthly "
            "returns (decimal fractions in the source CSV); compounded over "
            "the trailing window and, for the annualized view, scaled to a "
            "12-month rate."
        )
        chart_key = f"pb_rolling_{window_label}_{metric}_{','.join(picked)}"
        cu.line_chart(
            result, y_title, value_format=value_format, zero_line=zero_line, ref_line=ref_line, key=chart_key
        )

        st.subheader("Total growth of $1 (cumulative since inception, not rolling)")
        st.caption(
            "Compounds every monthly return from the start of the sample "
            "through today and never resets -- unlike the rolling window "
            "chart above."
        )
        pb_log_scale = st.checkbox("Log scale", value=False, key="pb_log_scale")
        total_growth = dl.compute_cumulative_growth(active, picked, pct_units=False)
        cu.line_chart(
            total_growth,
            "Growth of $1 (cumulative since inception)",
            value_format="$.2f",
            ref_line=1.0,
            log_scale=pb_log_scale,
            key=f"pb_total_growth_{','.join(picked)}_{pb_log_scale}",
        )
    with st.expander("Raw portfolio_active_returns.csv"):
        st.dataframe(active, use_container_width=True)

st.divider()

# --- Factor exposure comparison ---------------------------------------------
st.header("Factor exposure: portfolio vs. benchmark")
exposures = dl.load_portfolio_factor_exposures()
if exposures is None:
    dl.missing("the factor exposure comparison", "portfolio-benchmarking/Data/portfolio_factor_exposures.csv")
else:
    st.dataframe(exposures.round(3), use_container_width=True)
    plot_cols = [c for c in exposures.columns if c.startswith("Portfolio") or c == "Benchmark"]
    if plot_cols:
        exposures_long = (
            exposures[plot_cols]
            .reset_index()
            .rename(columns={"index": "Factor"})
            .melt(id_vars="Factor", var_name="Series", value_name="Exposure")
        )
        cu.grouped_bar_chart(
            exposures_long,
            "Factor",
            "Exposure",
            "Series",
            "Factor",
            "Factor exposure (beta)",
            value_format=".2f",
        )

benchmark_summary = dl.load_benchmark_mvo_summary_txt()
with st.expander("Benchmark construction summary (benchmark_mvo_summary.txt)"):
    if benchmark_summary is None:
        dl.missing("the benchmark construction summary", "portfolio-benchmarking/Data/benchmark_mvo_summary.txt")
    else:
        st.text(benchmark_summary)
