# Project Architecture Map

How `python-projects/` and `portfolio-mgmt-demo/` relate, what depends on
what, and how to actually run them. Written as a companion to
`quant_agent_guidelines.md` (that file is the coding-style/judgment rules;
this file is the concrete map of scripts, data files, and run order).

There is no orchestrator (no Makefile, Airflow DAG, or run-all script) —
everything below is run manually, in order, script by script. That's
consistent with `quant_agent_guidelines.md` §1: this is the **research
track**, and premature orchestration tooling would be exactly the kind of
complexity that guidance tells you to avoid until there's a real reason
(a schedule, another consumer, a live deliverable) to justify it.

---

## The two projects, in one sentence each

- **`python-projects/`** — the *research engine*: builds a general-purpose
  macro + Fama-French factor panel and fits exploratory models against it
  (PCA, VAR, factor regressions, VIF/HAC diagnostics, and a Bayesian
  shrinkage / time-varying-parameter track).
- **`portfolio-mgmt-demo/`** — a *portfolio benchmarking and tracking-error
  pipeline*, demonstrated on a fabricated sample portfolio. It **consumes**
  the factor panel `python-projects/` produces rather than rebuilding it.

So the relationship is one-directional: `portfolio-mgmt-demo` imports data
from `python-projects/Data/`, nothing flows the other way.

---

## `python-projects/` — pipeline stages

```
fetch_macro_fred.py ─┐
                      ├─> transform_data_ff.py ─> Data\FF_plus_macro_workable.csv
pull_yfinance_data.py ┘         (+ raw FF zips)         │
                                                          │
                            Data\asset_class_monthly_returns.csv
                            (independent Yahoo pull, 34 asset-class ETFs)
                                                          │
        ┌─────────────────────────────────────────────────┤
        v                                                  v
quant_model_macro_1.py                          quant_model_macro_3.py
  category PCA on 25 macro series                 asset ETFs ~ FF5 (OLS + HAC)
  -> Data\macro_pca_factors.csv                    -> asset_ff_*_betas/tstats*.csv
        │                                                  │
        v                                                  v
quant_model_macro_2.py                          quant_model_macro_4.py
  VARX: FF5 ~ lags + 9 macro factors               asset ETFs ~ FF5 + 9 macro (14 total)
  -> var_*.csv, var_model.pkl                       OLS + HAC side by side
        │                                            -> asset_full14_*.csv
        v
vif_check.py
  VIF across the same 14 factors (diagnostic only, no saved output beyond
  console — multicollinearity check before trusting quant_model_macro_4's
  betas)

plot_quant_pca_factors.py, plot_quant_ff_factors.py
  -> Data\macro_pca_factors.png, Data\ff_factors.png  (visual QA, leaf nodes)

tvp_ffbs_gibbs.py
  Standalone module, NOT wired into the pipeline yet. Carter & Kohn FFBS +
  Metropolis-within-Gibbs time-varying-beta sampler, under review before
  running on real data. Intended eventual consumer: per-asset hierarchical
  posteriors from the quant_model_macro_4.py betas would seed this model's
  priors — that hierarchical-fitting step itself isn't a saved script yet.
```

**Run order, if starting from scratch:**
1. `fetch_macro_fred.py` (needs a `FRED_API_KEY` environment variable — see
   its docstring), `pull_yfinance_data.py`, `transform_data_ff.py` — build
   the base panel.
2. `quant_model_macro_1.py` — macro PCA (must run before step 3/4, they
   depend on `macro_pca_factors.csv`).
3. `quant_model_macro_2.py` — VAR (depends on step 2's output).
4. `quant_model_macro_3.py` then `quant_model_macro_4.py` — asset
   regressions (4 supersedes 3's factor set, but 3's HAC-only-on-FF5 output
   is still referenced/used as the baseline comparison, so both are kept).
5. `vif_check.py` — diagnostic, run after step 4 (or before, it only needs
   the two factor CSVs) whenever you want to re-check collinearity, e.g.
   after changing `CATEGORIES` in step 1.
6. Plot scripts — anytime after the CSV they read exists.

**Files not part of the pipeline:**
- **`Portfolio Optimization.py`** — despite the name, this is a `cupy`/GPU
  Monte-Carlo MLE scratch script (ARCH-model-flavored: `scipy.stats.invgamma`,
  `rosen`/`rosen_der` test functions, `sympy.Matrix`). It doesn't read or
  write anything in `Data\` and isn't called by anything else.

---

## `portfolio-mgmt-demo/` — pipeline stages

**All data in this folder is fabricated** — `generate_sample_holdings.py`
builds a synthetic two-account portfolio (made-up tickers, weights, and
dollar values) with one large recent-IPO "concentrated position" (PLTR),
used to exercise the rest of the pipeline end-to-end.

Two **separate, non-interacting** sub-pipelines:

### A. Sample-holdings pipeline (active)

```
generate_sample_holdings.py
  fabricated two-account portfolio
  -> Data\holdings_clean.csv, Data\holdings_tickers.txt
        │
        v
fetch_holdings_prices.py
  -> Data\holdings_monthly_returns.csv
        │
        v
construct_benchmark_mvo.py  <───────── ..\python-projects\Data\FF_plus_macro_workable.csv (FF5 + RF)
  max-Sharpe long-only weights over the 5 FF factors (60-70% Mkt-RF constraint)
  -> Data\benchmark_mvo_weights.csv, Data\benchmark_wealth_curve.csv,
     Data\benchmark_effective_weights.csv, Data\benchmark_mvo_summary.txt
        │
        v
portfolio_factor_exposures.py  <────── ..\python-projects\Data\FF_plus_macro_workable.csv (FF5 + RF)
  per-holding FF5 regressions (HAC) -> portfolio-level exposure (full &
  ex-concentrated-position) -> benchmark exposure -> active exposure ->
  ex-ante TE (analytic) and ex-post TE (realized wealth-curve comparison)
  -> Data\holding_betas.csv, Data\portfolio_factor_exposures.csv,
     Data\portfolio_active_returns.csv, Data\tracking_error_vs_benchmark.txt
```

### B. Synthetic-portfolio tracking-error model (separate prototype)

```
benchmark_tracking_error_model.py  <── ..\python-projects\Data\FF_plus_macro_workable.csv
                                    <── ..\python-projects\Data\macro_pca_factors.csv
  *** uses its OWN separately-fabricated synthetic portfolio
      (build_synthetic_portfolio), independent of pipeline A's sample
      holdings *** -- exercises the rolling-beta / Ledoit-Wolf /
      tracking-error machinery with a portfolio that has deliberate style
      drift built in, useful for testing that machinery on something with
      known, controlled dynamics.
  -> Data\synthetic_portfolio_returns.csv, Data\rolling_betas.csv,
     Data\ledoit_wolf_covariance.csv, Data\rolling_realized_te.csv,
     Data\benchmark_static_betas.csv, Data\benchmark_fitted_vs_active.csv,
     Data\benchmark_regression_summary.txt, Data\tracking_error_summary.txt
```

**These two do not feed each other.** Pipeline A demonstrates the
sample-holdings-to-tracking-error flow end to end; pipeline B is a separate
scaffold exercising the rolling-beta/Ledoit-Wolf machinery specifically. As
the docstring in `benchmark_tracking_error_model.py` says: swap
`load_portfolio_returns()` to point at a real weighted return series and
this becomes a second, more sophisticated (rolling-beta, Ledoit-Wolf) TE
model — right now both A and B run on fabricated data by design.

**Run order for pipeline A:** `generate_sample_holdings.py` →
`fetch_holdings_prices.py` → `construct_benchmark_mvo.py` →
`portfolio_factor_exposures.py`. Steps 3 and 4 both independently read the
FF5 columns straight from `..\python-projects\Data\FF_plus_macro_workable.csv`
— that file has to exist (i.e., `transform_data_ff.py` must have been run in
the other project) before either will run.

---

## Cross-project dependency summary

| Consumer (portfolio-mgmt-demo) | Depends on (python-projects/Data/) |
|---|---|
| `construct_benchmark_mvo.py` | `FF_plus_macro_workable.csv` (FF5 + RF only) |
| `portfolio_factor_exposures.py` | `FF_plus_macro_workable.csv` (FF5 + RF only) |
| `benchmark_tracking_error_model.py` | `FF_plus_macro_workable.csv` + `macro_pca_factors.csv` (full 14-factor set) |
| `fetch_holdings_prices.py` | conceptually mirrors `pull_yfinance_data.py`'s conventions, no file dependency |

The dependency is entirely on **one file** in practice for the active
pipeline (A): `FF_plus_macro_workable.csv`. Nothing in `portfolio-mgmt-demo`
reads the macro PCA factors, the VAR output, the asset-class regressions
(`quant_model_macro_3/4.py`), or `vif_check.py`/`tvp_ffbs_gibbs.py` — those
stay purely inside the research project, at least for now.

**Practical implication**: if you change how `FF_plus_macro_workable.csv` is
built (e.g. a new FF vintage, a different `transform_data_ff.py` cleaning
step), pipeline A's benchmark and tracking-error numbers silently change too
on the next run — there's no version pin or snapshot between the two
projects, just a live path reference
(`PROJECTS_DATA_DIR = ../python-projects/Data`). Worth being deliberate
about re-running pipeline A after any upstream change, rather than assuming
it's insulated.

---

## `docs/` — published results

A static results/showcase page built from this repo's own committed data:
`docs/index.html` (chart gallery), `docs/factor-pipeline-math.html` (the
interactive methodology reference, MathJax-rendered), and
`docs/factor-pipeline-math.pdf` (the same content as a whitepaper).
`scripts/generate_showcase_charts.py` regenerates the PNGs in `docs/assets/`
directly from the CSVs both pipelines above produce — rerun it after any
upstream change to keep the showcase in sync. Served via GitHub Pages from
this branch's `/docs` folder.

---

## Should you actually orchestrate this?

Per `quant_agent_guidelines.md` §1 and §6: not yet, on the research side —
this is still exploratory, run-on-demand work, and a Makefile/Airflow DAG
here would be exactly the premature machinery that guidance warns against.

**Where I'd draw the line differently**: if `portfolio-mgmt-demo`'s pipeline
A ever gets pointed at a real, live portfolio, it's worth treating as closer
to the **production track** — real account-level data and a number someone
relies on deserve two things this repo doesn't yet have: (1) a single
`run_all.py` (or even a short shell script) so five ordered scripts don't
have to be remembered and run by hand, and (2) the same discipline
`benchmark_tracking_error_model.py`'s docstring already calls out —
synthetic-data outputs should never quietly get treated as real ones.
