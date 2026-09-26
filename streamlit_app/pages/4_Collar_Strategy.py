import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chart_utils as cu
import data_loader as dl

st.set_page_config(page_title="Collar Strategy | LATENT_V2", layout="wide")
st.title("4. Collar strategy what-if")

st.warning(
    "**Demo/what-if branch, not the canonical production pipeline.** Per "
    "the repo's `docs/HOW_IT_WORKS.md`, `portfolio-benchmarking/demos/"
    "collar-strategy/` is an isolated what-if analysis on top of the same "
    "fabricated sample household used on the Household Optimizer page — it "
    "is not an input to the real optimizer."
)
st.caption(
    "Source: portfolio-benchmarking/demos/collar-strategy/collar_cvar_comparison.csv, "
    "collar_scenarios_comparison.csv, and saa_benchmark_te_comparison.csv."
)

st.header("CVaR before/after a protective collar, by path")
collar = dl.load_collar_cvar_comparison()
if collar is None:
    dl.missing("the collar CVaR comparison", "portfolio-benchmarking/demos/collar-strategy/collar_cvar_comparison.csv")
else:
    st.dataframe(
        cu.format_table(
            collar,
            {
                "snow_weight": "{:.2%}",
                "snow_notional": "${:,.0f}",
                "cvar_uncollared": "{:.2%}",
                "cvar_collared": "{:.2%}",
                "improvement_pp": "{:.2f} pp",
            },
        ),
        use_container_width=True,
        hide_index=True,
    )
    chart_df = (
        collar.set_index("path")[["cvar_uncollared", "cvar_collared"]]
        .abs()
        .reset_index()
        .melt(id_vars="path", var_name="Series", value_name="abs_cvar")
    )
    cu.grouped_bar_chart(
        chart_df, "path", "abs_cvar", "Series", "Path", "|CVaR(5%)|", value_format=".1%"
    )
    st.caption(
        "Bars show |CVaR(5%)| by path — uncollared vs. collared on the "
        "same fabricated concentrated position (MSTR) across Before / "
        "Path C / Path D / SAA / SAA_3pct."
    )

st.divider()

st.header("CVaR & tracking error vs. collar coverage")
st.caption(
    "How much CVaR improves, and how much tracking error you take on, as "
    "more of the concentrated position is put under the collar (0% = no "
    "collar, 100% = the whole position) — already computed per path in "
    "collar_scenarios_comparison.csv."
)
scenarios = dl.load_collar_scenarios_comparison()
if scenarios is None:
    dl.missing("the collar scenario detail", "portfolio-benchmarking/demos/collar-strategy/collar_scenarios_comparison.csv")
else:
    scen = scenarios.copy()
    scen["abs_cvar"] = scen["cvar"].abs()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("CVaR vs. collar coverage")
        cu.scenario_line_chart(
            scen,
            "coverage",
            "abs_cvar",
            "path_label",
            "Collar coverage (% of position)",
            "|CVaR(5%)|",
        )
    with c2:
        st.subheader("Tracking error vs. collar coverage")
        cu.scenario_line_chart(
            scen,
            "coverage",
            "scenario_te",
            "path_label",
            "Collar coverage (% of position)",
            "Scenario tracking error",
        )
    with st.expander("Raw collar_scenarios_comparison.csv"):
        st.dataframe(scenarios, use_container_width=True, hide_index=True)

st.divider()

st.header("Collar exposure levels — 0% / 30% / 50% / 100%, by path")
st.caption(
    "Same collar_scenarios_comparison.csv, filtered to the collar coverage "
    "levels this demo actually solves (0% = no collar baseline, 30%, 50%, "
    "100% of the concentrated position). Broken out by path so you can see "
    "the actual dollar/share/contract exposure under the collar next to the "
    "CVaR and tracking-error trade-off it buys."
)
if scenarios is None:
    dl.missing(
        "the collar exposure breakdown",
        "portfolio-benchmarking/demos/collar-strategy/collar_scenarios_comparison.csv",
    )
else:
    levels_all = scenarios[scenarios["coverage"].isin([0.0, 0.3, 0.5, 1.0])].copy()
    levels_all["abs_cvar"] = levels_all["cvar"].abs()
    # 0% collar has zero exposure by construction, so it's excluded from the
    # notional/exposure table and chart below (nothing to show there) but
    # kept in for the CVaR-by-coverage-level comparison.
    levels = levels_all[levels_all["coverage"] > 0].copy()

    st.subheader("Exposure under the collar (unpacked)")
    exposure_table = levels[
        [
            "path_label",
            "coverage_label",
            "snow_weight",
            "covered_notional",
            "covered_shares",
            "covered_contracts",
        ]
    ].rename(
        columns={
            "path_label": "Path",
            "coverage_label": "Coverage",
            "snow_weight": "Concentrated position weight",
            "covered_notional": "Covered notional ($)",
            "covered_shares": "Covered shares",
            "covered_contracts": "Covered option contracts",
        }
    )
    st.dataframe(
        cu.format_table(
            exposure_table,
            {
                "Concentrated position weight": "{:.2%}",
                "Covered notional ($)": "${:,.0f}",
                "Covered shares": "{:,.1f}",
                "Covered option contracts": "{:,.2f}",
            },
        ),
        use_container_width=True,
        hide_index=True,
    )
    cu.grouped_bar_chart(
        levels,
        "path_label",
        "covered_notional",
        "coverage_label",
        "Path",
        "Covered notional ($)",
        value_format="$,.0f",
    )

    st.subheader("CVaR by path and coverage level (0% / 30% / 50% / 100%)")
    cu.grouped_bar_chart(
        levels_all, "path_label", "abs_cvar", "coverage_label", "Path", "|CVaR(5%)|", value_format=".1%"
    )
    st.caption(
        "Each bar is colored by collar coverage level, so you can see |CVaR(5%)| "
        "shrink from the uncovered 0% baseline through 30%, 50%, and full 100% "
        "coverage, path by path."
    )

    st.subheader("Tracking error by path and coverage level")
    cu.grouped_bar_chart(
        levels, "path_label", "scenario_te", "coverage_label", "Path", "Scenario tracking error", value_format=".1%"
    )

st.divider()

st.header("Tracking error: SAA benchmark, with vs. without proxy-tracking term")
saa_te = dl.load_saa_benchmark_te_comparison()
if saa_te is None:
    dl.missing("the SAA benchmark TE comparison", "portfolio-benchmarking/demos/collar-strategy/saa_benchmark_te_comparison.csv")
else:
    st.dataframe(
        cu.format_table(saa_te, {"saa_te": "{:.2%}", "saa_te_sleeve_only": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )
    saa_long = saa_te.melt(id_vars="path", var_name="Series", value_name="TE")
    cu.grouped_bar_chart(
        saa_long, "path", "TE", "Series", "Path", "Tracking error", value_format=".1%"
    )
