import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chart_utils as cu
import data_loader as dl

st.set_page_config(page_title="Household Optimizer | LATENT_V2", layout="wide")
st.title("3. Household CVaR/TE optimizer")

st.warning(
    "**Fabricated sample household — not a real client or brokerage account.** "
    "Every number on this page comes from running the real "
    "`optimize_tax_cvar_mvo.py` optimizer on a made-up two-account household "
    "built by `generate_sample_real_holdings.py`. There is no real household, "
    "brokerage workbook, or dollar figure behind any number here — see the "
    "repo README ('Household tax-aware CVaR optimizer (fabricated sample "
    "household, real pipeline)') and docs/HOW_IT_WORKS.md for how the "
    "fabrication is structured."
)
st.caption(
    "Source: portfolio-benchmarking/Data/path_metrics.json and optimizer_summary.txt "
    "(trade_proposals_pathC.csv / pathD.csv for the trade-level detail)."
)

metrics = dl.load_path_metrics_json()

st.header("Path C / D trade-off")
if metrics is None:
    dl.missing("the path trade-off table", "portfolio-benchmarking/Data/path_metrics.json")
else:
    paths = metrics.get("paths", {})
    rows = []
    for name in ["before", "C", "D"]:
        p = paths.get(name)
        if p is None:
            continue
        rows.append(
            {
                "Path": name,
                "TE": p.get("te"),
                "TE (factor-only)": p.get("te_factor_only"),
                "Active CVaR(5%)": p.get("cvar"),
                "Tax cost (ex-MSTR)": p.get("tax_cost"),
                "MSTR tax cost (excluded from budget)": p.get("tax_cost_snow"),
                "Total realized tax cost": p.get("tax_cost_total"),
                "Turnover": p.get("turnover"),
                "Sell cap on concentrated position": p.get("sell_cap"),
            }
        )
    df = pd.DataFrame(rows).set_index("Path")
    pct_cols = ["TE", "TE (factor-only)", "Active CVaR(5%)", "Turnover", "Sell cap on concentrated position"]
    fmt = {c: "{:.2%}" for c in pct_cols if c in df.columns}
    fmt.update({c: "${:,.0f}" for c in df.columns if "tax cost" in c.lower() or "Total realized" in c})
    st.dataframe(cu.format_table(df, fmt, na_rep="—"), use_container_width=True)

    total_value = metrics.get("total_household_value")
    if total_value:
        st.caption(f"Household total market value (fabricated): ${total_value:,.2f}")

    chart_df = df[["TE", "Active CVaR(5%)"]].abs()
    st.bar_chart(chart_df)

    with st.expander("Full path_metrics.json (all paths, incl. SAA variants)"):
        st.json(metrics)

st.divider()

st.header("Optimizer run detail (optimizer_summary.txt)")
summary = dl.load_optimizer_summary_txt()
if summary is None:
    dl.missing("the optimizer run detail", "portfolio-benchmarking/Data/optimizer_summary.txt")
else:
    st.text(dl.parse_optimizer_summary_narrative(summary))
    with st.expander("Full optimizer_summary.txt (includes the raw trade table already shown as a dataframe below)"):
        st.text(summary)

st.divider()

st.header("Trade-level detail")
tab_c, tab_d = st.tabs(["Path C (sell cap 70%)", "Path D (sell cap 30%)"])
with tab_c:
    trades_c = dl.load_trade_proposals("C")
    if trades_c is None:
        dl.missing("Path C trade proposals", "portfolio-benchmarking/Data/trade_proposals_pathC.csv")
    else:
        st.dataframe(trades_c, use_container_width=True, hide_index=True)

with tab_d:
    trades_d = dl.load_trade_proposals("D")
    if trades_d is None:
        dl.missing("Path D trade proposals", "portfolio-benchmarking/Data/trade_proposals_pathD.csv")
    else:
        st.dataframe(trades_d, use_container_width=True, hide_index=True)

st.divider()

st.header("Household holdings (fabricated sample)")
holdings = dl.load_holdings_household_clean()
if holdings is None:
    dl.missing("the household holdings table", "portfolio-benchmarking/Data/holdings_household_clean.csv")
else:
    st.dataframe(holdings, use_container_width=True, hide_index=True)
