import os
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chart_utils as cu
import optimizer_live as live

st.set_page_config(page_title="Optimizer Lab | LATENT_V2", layout="wide")
st.title("5. Optimizer Lab — live parameter sweeps")

st.warning(
    "**Fabricated sample household — not a real client or brokerage account.** "
    "Unlike every other page in this app, this page RE-RUNS the real "
    "`optimize_tax_cvar_mvo.py` solve() live, in-process, with parameters you "
    "choose below — it is not reading a pre-computed CSV/JSON. The math "
    "(objective, constraints, CVaR/TE definitions) is untouched; only the "
    "handful of tunable constants (MSTR sell cap, tax budget, turnover "
    "penalty, idio shrinkage, CVaR floor) are made adjustable. See the repo "
    "README and `optimize_tax_cvar_mvo.py`'s module docstring for the full "
    "problem statement."
)

DEFAULT_CONFIGS = [
    dict(name="Path C (70% MSTR cap)", mstr_sell_cap=0.70, tax_budget=0.02, turnover_lambda=0.02,
         idio_shrinkage=0.4, enforce_cvar=False, cvar_floor=-0.15, min_ticker_weight=0.003),
    dict(name="Path D (30% MSTR cap)", mstr_sell_cap=0.30, tax_budget=0.02, turnover_lambda=0.02,
         idio_shrinkage=0.4, enforce_cvar=False, cvar_floor=-0.15, min_ticker_weight=0.003),
    dict(name="Aggressive (100% cap, looser turnover)", mstr_sell_cap=1.00, tax_budget=0.03,
         turnover_lambda=0.005, idio_shrinkage=0.4, enforce_cvar=False, cvar_floor=-0.15,
         min_ticker_weight=0.003),
]

st.header("Configurations to compare")
n_cfg = st.number_input("Number of configurations", min_value=1, max_value=4, value=3, step=1)

with st.form("optimizer_configs"):
    cfgs = []
    cols = st.columns(int(n_cfg))
    for i, col in enumerate(cols):
        defaults = DEFAULT_CONFIGS[i] if i < len(DEFAULT_CONFIGS) else DEFAULT_CONFIGS[-1]
        with col:
            st.subheader(f"Config {i + 1}")
            name = st.text_input("Label", value=defaults["name"], key=f"name_{i}")
            mstr_cap = st.slider("MSTR sell cap (% of current position)", 0, 100,
                                  int(defaults["mstr_sell_cap"] * 100), 5, key=f"mstr_{i}") / 100.0
            tax_budget = st.slider("Annual tax budget (% of household)", 0.0, 5.0,
                                    defaults["tax_budget"] * 100, 0.25, key=f"tax_{i}") / 100.0
            turnover_lambda = st.select_slider(
                "Turnover penalty (lambda)", options=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
                value=defaults["turnover_lambda"], key=f"turn_{i}",
            )
            idio_shrinkage = st.slider("Idio covariance shrinkage (toward diagonal)", 0.0, 1.0,
                                        defaults["idio_shrinkage"], 0.1, key=f"shrink_{i}")
            enforce_cvar = st.checkbox("Enforce CVaR floor constraint", value=defaults["enforce_cvar"],
                                        key=f"envcvar_{i}")
            cvar_floor = st.slider("CVaR floor (worst-5% active return, annual)", -30, 0,
                                    int(defaults["cvar_floor"] * 100), 1, key=f"floor_{i}",
                                    disabled=not enforce_cvar) / 100.0
            min_ticker_weight = st.slider("Min per-ticker floor (%)", 0.0, 2.0,
                                           defaults["min_ticker_weight"] * 100, 0.1, key=f"minw_{i}") / 100.0
            cfgs.append(live.OptConfig(
                label=name, mstr_sell_cap=mstr_cap, tax_budget=tax_budget,
                turnover_lambda=turnover_lambda, idio_shrinkage=idio_shrinkage,
                enforce_cvar=enforce_cvar, cvar_floor=cvar_floor,
                min_ticker_weight=min_ticker_weight,
            ))
    n_boot = st.slider("TE bootstrap draws (block bootstrap of factor covariance)", 50, 500, 150, 50)
    submitted = st.form_submit_button("Run optimizer", type="primary")

if not submitted:
    st.info("Adjust parameters above and click **Run optimizer**.")
    st.stop()

results = []
with st.spinner("Solving each configuration..."):
    for cfg in cfgs:
        try:
            results.append(live.run_config(cfg))
        except Exception as e:
            results.append({"label": cfg.label, "config": cfg, "feasible": False, "message": str(e)})

infeasible = [r for r in results if not r["feasible"]]
for r in infeasible:
    st.error(f"**{r['label']}**: infeasible — {r['message']}")

feasible_results = [r for r in results if r["feasible"]]
if not feasible_results:
    st.stop()

st.divider()

# --- Headline metrics table --------------------------------------------------
st.header("Headline metrics per configuration")
summary_rows = []
for r in feasible_results:
    summary_rows.append({
        "Config": r["label"],
        "TE (factor-only)": r["te_factor"],
        "TE (idio, correlated)": r["te_idio"],
        "TE (total)": r["te_total"],
        "Active CVaR(5%)": r["cvar"],
        "Tax cost": r["tax_cost"] / live.load_base_data()["total_household_value"],
        "Turnover": r["turnover"],
    })
summary_df = pd.DataFrame(summary_rows).set_index("Config")
pct_cols = ["TE (factor-only)", "TE (idio, correlated)", "TE (total)", "Active CVaR(5%)", "Tax cost", "Turnover"]
st.dataframe(cu.format_table(summary_df, {c: "{:.2%}" for c in pct_cols}), use_container_width=True)

# --- PM trade-off view: risk vs. tax cost vs. turnover, one glance --------------
st.subheader("Trade-off view: risk reduction vs. tax cost vs. turnover")
st.caption(
    "Bubble size = turnover (one-way $ moved, as % of household). The ideal "
    "corner is bottom-left (low TE, low tax cost) with a small bubble (low turnover) — "
    "every config here is a different point on that same trade-off, not a strictly-better one."
)
tradeoff_df = pd.DataFrame([{
    "Config": r["label"], "TE (total)": r["te_total"],
    "Tax cost (% of household)": r["tax_cost"] / live.load_base_data()["total_household_value"],
    "Turnover": r["turnover"], "Active CVaR(5%)": r["cvar"],
} for r in feasible_results])
tradeoff_chart = alt.Chart(tradeoff_df).mark_circle(opacity=0.8).encode(
    x=alt.X("TE (total):Q", axis=alt.Axis(format="%")),
    y=alt.Y("Tax cost (% of household):Q", axis=alt.Axis(format="%")),
    size=alt.Size("Turnover:Q", legend=alt.Legend(format="%")),
    color=alt.Color("Config:N", legend=alt.Legend(title="Config")),
    tooltip=["Config", alt.Tooltip("TE (total):Q", format=".2%"),
             alt.Tooltip("Tax cost (% of household):Q", format=".2%"),
             alt.Tooltip("Turnover:Q", format=".2%"),
             alt.Tooltip("Active CVaR(5%):Q", format=".2%")],
).properties(height=380)
st.altair_chart(tradeoff_chart, use_container_width=True)

st.divider()

# --- Asset allocation by class + concentration -----------------------------------
st.header("Portfolio construction: allocation by asset class & concentration")
base = live.load_base_data()
class_of = base["class_of"]
tickers_row0 = feasible_results[0]["tickers_row"]  # same holdings universe for every config
x0 = feasible_results[0]["x0"]

alloc_rows = []
current_by_class = live.weight_by_class(x0, tickers_row0, class_of)
for cls, w in current_by_class.items():
    alloc_rows.append({"Config": "Current (before any trades)", "Asset class": cls, "Weight": float(w)})
for r in feasible_results:
    by_class = live.weight_by_class(r["x"], r["tickers_row"], class_of)
    for cls, w in by_class.items():
        alloc_rows.append({"Config": r["label"], "Asset class": cls, "Weight": float(w)})
alloc_df = pd.DataFrame(alloc_rows)

alloc_chart = alt.Chart(alloc_df).mark_bar().encode(
    x=alt.X("Config:N", title=None, sort=["Current (before any trades)"] + [r["label"] for r in feasible_results]),
    y=alt.Y("Weight:Q", axis=alt.Axis(format="%"), stack="zero"),
    color=alt.Color("Asset class:N", legend=alt.Legend(columns=1)),
    order=alt.Order("Asset class:N"),
    tooltip=["Config", "Asset class", alt.Tooltip("Weight:Q", format=".2%")],
).properties(height=460)
st.altair_chart(alloc_chart, use_container_width=True)

hhi_rows = [{"Config": "Current (before any trades)",
             "HHI (ticker-level)": live.herfindahl_index(x0),
             "MSTR weight": float(pd.Series(x0, index=tickers_row0).groupby(level=0).sum().get("MSTR", 0.0))}]
for r in feasible_results:
    hhi_rows.append({
        "Config": r["label"],
        "HHI (ticker-level)": live.herfindahl_index(r["x"]),
        "MSTR weight": float(pd.Series(r["x"], index=r["tickers_row"]).groupby(level=0).sum().get("MSTR", 0.0)),
    })
hhi_df = pd.DataFrame(hhi_rows).set_index("Config")
st.caption(
    "Herfindahl-Hirschman Index (sum of squared position weights) — higher means more "
    "concentrated. A 16-ticker equal-weighted book would sit around 6.25%; this fabricated "
    "household's concentration is almost entirely the MSTR position, shown alongside for context."
)
st.dataframe(
    cu.format_table(hhi_df, {"HHI (ticker-level)": "{:.2%}", "MSTR weight": "{:.2%}"}),
    use_container_width=True,
)

st.divider()

# --- CVaR box plot -------------------------------------------------------------
st.header("Active-return scenario distribution & CVaR by configuration")
st.caption(
    "Each box is the distribution of overlapping 12-month active-return scenarios "
    "(same scenarios `optimize_tax_cvar_mvo.py`'s CVaR constraint is built on). "
    "The red diamond marks Active CVaR(5%) — the mean of the worst 5% tail, not the box-plot minimum."
)
scen_rows = []
for r in feasible_results:
    for v in r["active_scenarios"]:
        scen_rows.append({"Config": r["label"], "Active return (annual scenario)": float(v)})
scen_df = pd.DataFrame(scen_rows)

box = alt.Chart(scen_df).mark_boxplot(extent="min-max", size=40).encode(
    x=alt.X("Config:N", title=None),
    y=alt.Y("Active return (annual scenario):Q", axis=alt.Axis(format="%")),
    color=alt.Color("Config:N", legend=None),
)
cvar_points = pd.DataFrame({"Config": [r["label"] for r in feasible_results], "cvar": [r["cvar"] for r in feasible_results]})
cvar_marks = alt.Chart(cvar_points).mark_point(shape="diamond", size=140, color="crimson", filled=True).encode(
    x="Config:N", y=alt.Y("cvar:Q", axis=alt.Axis(format="%")),
)
st.altair_chart((box + cvar_marks).properties(height=420), use_container_width=True)

st.divider()

# --- TE box plot (bootstrap) ---------------------------------------------------
st.header("Tracking-error estimation uncertainty by configuration")
st.caption(
    "Each box is a block-bootstrap (12-month blocks) resample of the monthly FF5 factor "
    "panel, rebuilding the blended long-run + EWMA covariance and re-scoring each "
    "configuration's OWN optimized weights against it — i.e. \"how much would this path's "
    "TE move if the recent factor-covariance regime had played out slightly differently.\" "
    "The idiosyncratic term is held at its point estimate; only the factor-covariance term is resampled."
)
with st.spinner("Bootstrapping TE distributions..."):
    te_rows = []
    for r in feasible_results:
        te_boot = live.bootstrap_te_distribution(r, n_boot=n_boot)
        for v in te_boot:
            te_rows.append({"Config": r["label"], "Bootstrapped TE": float(v)})
te_df = pd.DataFrame(te_rows)

te_box = alt.Chart(te_df).mark_boxplot(extent="min-max", size=40).encode(
    x=alt.X("Config:N", title=None),
    y=alt.Y("Bootstrapped TE:Q", axis=alt.Axis(format="%")),
    color=alt.Color("Config:N", legend=None),
)
te_points = pd.DataFrame({"Config": [r["label"] for r in feasible_results], "te": [r["te_total"] for r in feasible_results]})
te_marks = alt.Chart(te_points).mark_point(shape="diamond", size=140, color="crimson", filled=True).encode(
    x="Config:N", y=alt.Y("te:Q", axis=alt.Axis(format="%")),
)
st.altair_chart((te_box + te_marks).properties(height=420), use_container_width=True)

st.divider()

# --- TE risk contribution: factor + idiosyncratic -------------------------------
st.header("TE risk contribution: factors vs. idiosyncratic risk")
st.caption(
    "Euler (exact, additive) decomposition of total tracking error, same formulas as "
    "`portfolio_risk_attribution.py`: each FF5 factor's contribution can be NEGATIVE "
    "(a hedging exposure), and idiosyncratic risk is broken out by asset class. "
    "Bars sum exactly to each configuration's TE (total)."
)
contrib_rows = []
for r in feasible_results:
    for factor, v in r["te_contrib_factor"].items():
        contrib_rows.append({"Config": r["label"], "Source": factor, "Group": "Factor", "TE contribution": float(v)})
    for cls, v in r["te_idio_contrib_by_class"].items():
        if abs(v) < 1e-6:
            continue
        contrib_rows.append({"Config": r["label"], "Source": f"Idio: {cls}", "Group": "Idiosyncratic", "TE contribution": float(v)})
contrib_df = pd.DataFrame(contrib_rows)

contrib_chart = alt.Chart(contrib_df).mark_bar().encode(
    x=alt.X("Config:N", title=None),
    y=alt.Y("TE contribution:Q", axis=alt.Axis(format="%"), stack="zero"),
    color=alt.Color("Source:N", legend=alt.Legend(title="Risk source", columns=2)),
    order=alt.Order("Group:N"),
    tooltip=["Config", "Source", alt.Tooltip("TE contribution:Q", format=".2%")],
)
st.altair_chart(contrib_chart.properties(height=460), use_container_width=True)

with st.expander("Active CVaR tail-conditional factor attribution (informational — not a constraint on either path above unless 'Enforce CVaR floor' was checked)"):
    st.caption(
        "Same tail scenarios as the CVaR box plot above, but averaging each factor's OWN "
        "return over just the worst-5% scenarios instead of the whole distribution — a factor "
        "can be a big TE contributor and a small CVaR contributor if it didn't move against the "
        "book specifically when the book was losing. The residual is what the 5 factors don't "
        "explain about the tail (overwhelmingly MSTR idiosyncratic risk in this book)."
    )
    cvar_contrib_rows = []
    for r in feasible_results:
        for factor, v in r["cvar_contrib_factor"].items():
            cvar_contrib_rows.append({"Config": r["label"], "Source": factor, "CVaR contribution": float(v)})
        cvar_contrib_rows.append({"Config": r["label"], "Source": "Residual (idiosyncratic)", "CVaR contribution": float(r["cvar_residual"])})
    cvar_contrib_df = pd.DataFrame(cvar_contrib_rows)
    cvar_contrib_chart = alt.Chart(cvar_contrib_df).mark_bar().encode(
        x=alt.X("Config:N", title=None),
        y=alt.Y("CVaR contribution:Q", axis=alt.Axis(format="%"), stack="zero"),
        color=alt.Color("Source:N", legend=alt.Legend(title="Tail risk source", columns=2)),
        tooltip=["Config", "Source", alt.Tooltip("CVaR contribution:Q", format=".2%")],
    )
    st.altair_chart(cvar_contrib_chart.properties(height=420), use_container_width=True)

st.divider()

# --- Macro regime tie-in ---------------------------------------------------------
st.header("Tying the portfolio to macro factor data")
st.caption(
    "The optimizer above minimizes tracking error against 5 Fama-French style factors "
    "(Mkt-RF/SMB/HML/RMW/CMA). Those don't speak directly to a macro view. This section "
    "regresses each HELD TICKER's own monthly return (not a generic asset-class proxy) on "
    "the 9 macro PCA factors already built in this repo "
    "(`factor-research/Data/macro_pca_factors.csv`, from `quant_model_macro_*.py`), then "
    "aggregates by weight — an exact, additive decomposition — so you can read off which "
    "asset classes are driving the book's exposure to growth, inflation, rates, policy, "
    "currency, credit, and commodity factors, before and after each proposed rebalance."
)
with st.spinner("Regressing household tickers on macro PCA factors..."):
    macro_pca = live.load_macro_pca_factors()
    macro_beta_wide, macro_r2 = live.regress_tickers_macro(base["returns_wide"], macro_pca, base["usable_tickers"])

macro_rows = []
current_macro = live.macro_exposure_by_class(x0, tickers_row0, macro_beta_wide, class_of)
for cls in current_macro.index:
    for factor in live.MACRO_FACTOR_COLS:
        macro_rows.append({"Config": "Current (before any trades)", "Macro factor": factor,
                            "Asset class": cls, "Weighted beta": float(current_macro.loc[cls, factor])})
for r in feasible_results:
    m = live.macro_exposure_by_class(r["x"], r["tickers_row"], macro_beta_wide, class_of)
    for cls in m.index:
        for factor in live.MACRO_FACTOR_COLS:
            macro_rows.append({"Config": r["label"], "Macro factor": factor,
                                "Asset class": cls, "Weighted beta": float(m.loc[cls, factor])})
macro_df = pd.DataFrame(macro_rows)

config_pick = st.selectbox(
    "Configuration to show macro-factor exposure for",
    ["Current (before any trades)"] + [r["label"] for r in feasible_results],
)
macro_chart = alt.Chart(macro_df[macro_df["Config"] == config_pick]).mark_bar().encode(
    x=alt.X("Macro factor:N", title=None),
    y=alt.Y("Weighted beta:Q", stack="zero", title="Portfolio-weighted beta"),
    color=alt.Color("Asset class:N", legend=alt.Legend(columns=1)),
    order=alt.Order("Asset class:N"),
    tooltip=["Macro factor", "Asset class", alt.Tooltip("Weighted beta:Q", format=".3f")],
).properties(height=460)
st.altair_chart(macro_chart, use_container_width=True)

macro_playbook_notes = {
    "Inflation_Factor": "Theme 2 (Structural Inflation) in the macro playbook — positive beta benefits if greenflation/deglobalization/wage pressure persists; negative beta is exposed to a disinflation surprise.",
    "Rates_Yields_Factor": "Theme 1 (Fiscal Dominance) — term-premium and bear-steepening risk; large positive beta is exposed to a failed-auction-style yield spike.",
    "US_Monetary_Policy_Factor": "Fed reaction-function risk — how much the book moves with policy-rate surprises specifically, distinct from the broader yield-curve factor above.",
    "Global_Monetary_Policy_Factor": "Non-US central bank policy risk — relevant if reading this alongside the playbook's fiscal-monetary collision scenario abroad, not just in the US.",
    "Currency_Factor": "Theme 3 (Geopolitical Fragmentation) — USD debasement / reserve-diversification exposure.",
    "Credit_Risk_Factor": "Section 2, Themes 1 & 2 (Private Markets/Public Illiquidity + Systemic Leverage & Basis Trades) — spread-widening sensitivity; a stress-test analog for the playbook's private-credit gating cascade.",
    "Commodities_Factor": "Theme 3 — real-asset/commodity-supply-shock sensitivity (chokepoint disruption, re-shoring capex).",
    "US_Growth_Factor": "Domestic growth-cycle beta — the soft-landing vs. hard-landing axis in the Section 4 scenario matrix.",
    "Global_Growth_Factor": "Ex-US growth-cycle beta — relevant to the EM/international sleeves specifically.",
}
with st.expander("How to read each macro factor against the playbook's themes/scenarios"):
    for factor in live.MACRO_FACTOR_COLS:
        st.markdown(f"**{factor.replace('_', ' ')}** — {macro_playbook_notes.get(factor, '')}")
    st.caption(
        "This is interpretive framing on top of the regression betas above, not a separate "
        "computed number — it borrows the theme names from the attached macro playbook "
        "(`docs/multi_asset_macro_microstructure_playbook.md`) to help read the chart; it "
        "does not imply the playbook's narrative is realized or forecast."
    )

with st.expander("Macro-factor beta regression quality (R², by ticker)"):
    st.dataframe(
        cu.format_table(macro_r2.rename("R^2").to_frame(), {"R^2": "{:.2%}"}), use_container_width=True
    )

st.divider()

st.header("Trade-level detail per configuration")
tabs = st.tabs([r["label"] for r in feasible_results])
for tab, r in zip(tabs, feasible_results):
    with tab:
        total_value = live.load_base_data()["total_household_value"]
        opt_mod, _ = live.get_optimizer_modules()
        trades = opt_mod.build_trade_table(r["rows"], r["x0"], r["x"], r["tax_vec_full"], total_value)
        st.dataframe(trades, use_container_width=True, hide_index=True)
