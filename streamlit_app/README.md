# LATENT_V2 Streamlit dashboard

Interactive companion to the static results at
[akirp002.github.io/LATENT_V2](https://akirp002.github.io/LATENT_V2/). Every
page except **5. Optimizer Lab** reads this repo's own already-computed
CSV/JSON/TXT outputs — it does not call yfinance/FRED and does not re-run
any optimizer. The Optimizer Lab page is the one exception: it imports and
calls the real `optimize_tax_cvar_mvo.py` `solve()` live, in-process, with
parameters you choose in the UI (MSTR sell cap, tax budget, turnover
penalty, idio shrinkage, CVaR floor), so you can sweep configurations
beyond the two fixed Path C / Path D runs already baked into
`path_metrics.json`. See `streamlit_app/optimizer_live.py`'s docstring.

## Run it

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

## Pages and their data sources

| Page | Reads |
|---|---|
| `app.py` (landing page) | no data, just links to the four sections |
| `pages/1_Factor_Research.py` | `factor-research/Data/macro_pca_factors.csv`; `FF_plus_macro_workable.csv` + `macro_pca_factors.csv` (VIF, recomputed via the same formula as `vif_check.py`); `asset_full14_betas_tstats_long.csv` (betas/t-stats table and the τ̂ shrinkage calc, same formula as `scripts/generate_showcase_charts.py`'s `chart_tau_shrinkage()`) |
| `pages/2_Portfolio_Benchmarking.py` | `portfolio-benchmarking/Data/benchmark_wealth_curve.csv`, `portfolio_active_returns.csv`, `tracking_error_vs_benchmark.txt`, `tracking_error_summary.txt`, `portfolio_factor_exposures.csv`, `benchmark_mvo_summary.txt` |
| `pages/3_Household_Optimizer.py` | `portfolio-benchmarking/Data/path_metrics.json`, `optimizer_summary.txt`, `trade_proposals_pathC.csv`, `trade_proposals_pathD.csv`, `holdings_household_clean.csv` — all on the **fabricated** sample household from `generate_sample_real_holdings.py` |
| `pages/4_Collar_Strategy.py` | `portfolio-benchmarking/demos/collar-strategy/collar_cvar_comparison.csv`, `saa_benchmark_te_comparison.csv`, `collar_scenarios_comparison.csv` — demo/what-if branch, not the canonical production pipeline |
| `pages/5_Optimizer_Lab.py` | **Live**: calls `portfolio-benchmarking/optimize_tax_cvar_mvo.py`'s `solve()` directly (via `streamlit_app/optimizer_live.py`), for up to 4 user-parameterized configurations at once. Shows CVaR box plots (the scenario distribution each config's Active CVaR is the tail mean of), TE box plots (block-bootstrap of the factor covariance), and the Euler TE risk decomposition into factor + idiosyncratic-by-asset-class contributions, reusing `portfolio_risk_attribution.py`'s formulas |

`streamlit_app/data_loader.py` centralizes every read; if a source file is
missing in a given checkout, the corresponding widget shows a plain "not
available" message instead of crashing. The only computation done outside
of loading a file is the VIF table and the τ̂ shrinkage table, both
recomputed with the exact same formulas the source scripts use (see the
docstrings in `data_loader.py`) so the numbers match the published
showcase charts — no new modeling or numbers are introduced.

All household/portfolio holdings shown anywhere in this app are the
fabricated sample data described in the top-level `README.md` and
`docs/HOW_IT_WORKS.md`, not a real account.
