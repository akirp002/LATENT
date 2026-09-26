"""
LATENT_V2 -- interactive results dashboard.

Entry point: `streamlit run streamlit_app/app.py`

This app is mostly a thin, read-only viewer over data this repo's pipeline
scripts have already computed and written to `Data/` under
factor-research/ and portfolio-benchmarking/ -- it does not call yfinance
or FRED. The one exception is page 5, Optimizer Lab, which DOES re-run the
real optimizer live with parameters you choose -- see
streamlit_app/README.md for exactly which file (or live call) feeds which
page. Use the sidebar to move between sections; Streamlit auto-discovers
them from streamlit_app/pages/.
"""

import streamlit as st

st.set_page_config(page_title="LATENT_V2 Dashboard", layout="wide")

st.title("LATENT_V2 — interactive results dashboard")

st.markdown(
    """
This app is an interactive companion to the static results published at
[akirp002.github.io/LATENT_V2](https://akirp002.github.io/LATENT_V2/). It
reads the exact same CSV/JSON/TXT files this repo's pipelines already
wrote to `Data/` — nothing is re-fetched, re-optimized, or invented here.

Use the sidebar to open a section:

1. **Factor research overview** — macro PCA factors, VIF collinearity check,
   asset factor betas/t-stats, and the empirical-Bayes τ̂ shrinkage result.
2. **Portfolio benchmarking** — benchmark wealth curve vs. the sample
   portfolio, tracking-error numbers, factor exposure comparison.
3. **Household CVaR/TE optimizer** — the Path C/D trade-off on the
   **fabricated sample household** (see banner on that page).
4. **Collar strategy what-if** — CVaR before/after a protective collar on
   the concentrated position, across paths.
5. **Optimizer Lab** — live parameter sweeps of the real optimizer: adjust
   the MSTR sell cap, tax budget, turnover penalty, idio shrinkage, and
   CVaR floor across multiple configurations at once, and compare them with
   CVaR/TE box plots and a factor + idiosyncratic TE risk-contribution
   breakdown.

All portfolio/household numbers in this app are demo data built by
`generate_sample_real_holdings.py` and friends — see the repo's top-level
`README.md` and `docs/HOW_IT_WORKS.md` for exactly how the fabrication is
structured. Factor-research numbers (VIF, τ̂, betas) run on real FRED/
Fama-French/yfinance history.
"""
)

st.caption(
    "Source of truth for every number shown here: the CSV/JSON/TXT files under "
    "`factor-research/Data/` and `portfolio-benchmarking/Data/` in this repo. "
    "See streamlit_app/README.md for the exact file -> page mapping."
)
