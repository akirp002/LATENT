# Project Architecture Map

How `Python projects\` and `Portfolio Mgmt Ajit\` relate, what depends on
what, and how to actually run them. Written as a companion to
`quant_agent_guidelines.md` (that file is the coding-style/judgment rules;
this file is the concrete map of scripts, data files, and run order).

There is no orchestrator (no Makefile, Airflow DAG, or run-all script) —
everything below is run manually, in order, script by script. That's
consistent with `quant_agent_guidelines.md` §1: this is the **research
track**, and premature orchestration tooling would be exactly the kind of
complexity that guidance tells you to avoid until there's a real reason
(a schedule, another consumer, client-facing delivery) to justify it. See
"Should you actually orchestrate this?" at the bottom.

---

## The two projects, in one sentence each

- **`Python projects\`** — the *research engine*: builds a general-purpose
  macro + Fama-French factor panel and fits exploratory models against it
  (PCA, VAR, factor regressions, VIF/HAC diagnostics, and now a TVP/Bayesian
  track). Nothing here is about Ajit's actual money.
- **`Portfolio Mgmt Ajit\`** — a *specific client deliverable*: parses Ajit &
  Apurva's real brokerage holdings, builds a strategic benchmark, and
  measures tracking error / factor exposure for that real portfolio. It
  **consumes** the factor panel `Python projects\` produces rather than
  rebuilding it.

So the relationship is one-directional: `Portfolio Mgmt Ajit` imports data
from `Python projects\Data\`, nothing flows the other way.

---

## `Python projects\` — pipeline stages

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
  Metropolis-within-Gibbs time-varying-beta sampler, currently under review
  (not executed on real data). Intended eventual consumer: the per-asset
  static hierarchical posteriors from the quant_model_macro_4.py betas would
  seed this model's priors, but that hierarchical fitting step itself hasn't
  been built as a script yet — right now it only exists as ad hoc analysis
  in-session (the tau^2 empirical-Bayes computation), not a saved artifact.
```

**Run order, if starting from scratch:**
1. `fetch_macro_fred.py`, `pull_yfinance_data.py`, `transform_data_ff.py` — build the base panel.
2. `quant_model_macro_1.py` — macro PCA (must run before step 3/4, they depend on `macro_pca_factors.csv`).
3. `quant_model_macro_2.py` — VAR (depends on step 2's output).
4. `quant_model_macro_3.py` then `quant_model_macro_4.py` — asset regressions (4 supersedes 3's factor set, but 3's HAC-only-on-FF5 output is still referenced/used as the baseline comparison, so both are kept).
5. `vif_check.py` — diagnostic, run after step 4 (or before, it only needs the two factor CSVs) whenever you want to re-check collinearity, e.g. after changing `CATEGORIES` in step 1.
6. Plot scripts — anytime after the CSV they read exists.

**Files not part of the pipeline (orphaned / needs your attention):**
- **`Portfolio Optimization.py`** — despite the name, this is a `cupy`/GPU Monte-Carlo MLE scratch script (ARCH-model-flavored: `scipy.stats.invgamma`, `rosen`/`rosen_der` test functions, `sympy.Matrix`). It doesn't read or write anything in `Data\` and isn't called by anything else. Worth either renaming to reflect what it actually is, or deleting if it was superseded.
- **`Data\arch_mle_mc.cpp (bxo9gm).cpp (bxo9gm).cpp (bxo9gm)`** — a stray file with a mangled triple-duplicated name (looks like an editor autosave/conflict artifact), sitting in `Data\` where every other file is a pipeline output. Not referenced by any script. Worth deleting or moving out of `Data\`.
- **`claude_api_key.txt`** — a bare secret sitting in the project root. Flagging since it's the kind of file that's easy to accidentally commit/sync/share; worth moving to an env var or a git-ignored/OneDrive-excluded location if this folder is ever versioned or shared.

---

## `Portfolio Mgmt Ajit\` — pipeline stages

Two **separate, non-interacting** sub-pipelines live in this folder:

### A. Real-holdings pipeline (current, active)

```
"Ajit & Apurva's Financials.xlsx"
        │
        v
load_holdings.py
  -> Data\holdings_clean.csv, Data\holdings_tickers.txt
        │
        v
fetch_holdings_prices.py
  -> Data\holdings_monthly_returns.csv
        │
        v
construct_benchmark_mvo.py  <───────── ..\Python projects\Data\FF_plus_macro_workable.csv (FF5 + RF)
  max-Sharpe long-only weights over the 5 FF factors (60-70% Mkt-RF constraint)
  -> Data\benchmark_mvo_weights.csv, Data\benchmark_wealth_curve.csv,
     Data\benchmark_effective_weights.csv, Data\benchmark_mvo_summary.txt
        │
        v
portfolio_factor_exposures.py  <────── ..\Python projects\Data\FF_plus_macro_workable.csv (FF5 + RF)
  per-holding FF5 regressions (HAC) -> portfolio-level exposure (full & ex-SNOW)
  -> benchmark exposure -> active exposure -> ex-ante TE (analytic) and
  ex-post TE (realized wealth-curve comparison)
  -> Data\holding_betas.csv, Data\portfolio_factor_exposures.csv,
     Data\portfolio_active_returns.csv, Data\tracking_error_vs_benchmark.txt
```

### B. Synthetic-portfolio tracking-error model (older, placeholder)

```
benchmark_tracking_error_model.py  <── ..\Python projects\Data\FF_plus_macro_workable.csv
                                    <── ..\Python projects\Data\macro_pca_factors.csv
  *** uses a SYNTHETIC portfolio (build_synthetic_portfolio), not real
      holdings *** -- built before load_holdings.py existed, as a way to
      exercise the rolling-beta / Ledoit-Wolf / tracking-error machinery
      before real data was available.
  -> Data\synthetic_portfolio_returns.csv, Data\rolling_betas.csv,
     Data\ledoit_wolf_covariance.csv, Data\rolling_realized_te.csv,
     Data\benchmark_static_betas.csv, Data\benchmark_fitted_vs_active.csv,
     Data\benchmark_regression_summary.txt, Data\tracking_error_summary.txt
```

**These two do not feed each other.** Pipeline A is the real deliverable;
pipeline B is a scaffold/prototype that predates real data. As the docstring
in `benchmark_tracking_error_model.py` itself says: swap
`load_portfolio_returns()` to point at `holdings_monthly_returns.csv` (or a
real weighted return series built from it) and this becomes a second, more
sophisticated (rolling-beta, Ledoit-Wolf) TE model for the real portfolio —
right now it's still running on fake data, which is easy to forget since
the script otherwise looks fully production-shaped.

**Run order for pipeline A:** `load_holdings.py` → `fetch_holdings_prices.py`
→ `construct_benchmark_mvo.py` → `portfolio_factor_exposures.py`. Steps 3
and 4 both independently read the FF5 columns straight from
`..\Python projects\Data\FF_plus_macro_workable.csv` — that file has to
exist (i.e., `transform_data_ff.py` must have been run in the other project)
before either will run.

---

## Cross-project dependency summary

| Consumer (Portfolio Mgmt Ajit) | Depends on (Python projects\Data\) |
|---|---|
| `construct_benchmark_mvo.py` | `FF_plus_macro_workable.csv` (FF5 + RF only) |
| `portfolio_factor_exposures.py` | `FF_plus_macro_workable.csv` (FF5 + RF only) |
| `benchmark_tracking_error_model.py` | `FF_plus_macro_workable.csv` + `macro_pca_factors.csv` (full 14-factor set) |
| `fetch_holdings_prices.py` | conceptually mirrors `pull_yfinance_data.py`'s conventions, no file dependency |

The dependency is entirely on **one file** in practice for the active
pipeline (A): `FF_plus_macro_workable.csv`. Nothing in `Portfolio Mgmt Ajit`
reads the macro PCA factors, the VAR output, the asset-class regressions
(`quant_model_macro_3/4.py`), or `vif_check.py`/`tvp_ffbs_gibbs.py` — those
stay purely inside the research project, at least for now.

**Practical implication**: if you change how `FF_plus_macro_workable.csv` is
built (e.g. a new FF vintage, a different `transform_data_ff.py` cleaning
step), pipeline A's benchmark and tracking-error numbers for Ajit's real
portfolio silently change too on the next run — there's no version pin or
snapshot between the two projects, just a live path reference
(`PROJECTS_DATA_DIR = r"...\Python projects\Data"`). Worth being deliberate
about re-running pipeline A after any upstream change, rather than assuming
it's insulated.

---

## Should you actually orchestrate this?

Per `quant_agent_guidelines.md` §1 and §6: not yet, on the research side —
this is still exploratory, single-user, run-on-demand work, and a
Makefile/Airflow DAG here would be exactly the premature machinery that
guidance warns against.

**Where I'd draw the line differently**: `Portfolio Mgmt Ajit`'s pipeline A
is closer to the **production track** than it currently looks — it touches
real account-level data (`Ajit & Apurva's Financials.xlsx`) and its output
(`tracking_error_vs_benchmark.txt`) is a number someone (Ajit) presumably
looks at as if it's a real, current risk figure. Two things worth doing
before this becomes a habit you rely on:

1. **A single `run_all.py` (or even a short PowerShell script) for pipeline
   A specifically** — five scripts in a fixed order, each depending on the
   last one's output, with no automated re-run today. That's the exact
   "reruns should be repeatable without hidden manual steps" bar from
   guidelines §1, and it's currently not met — you'd have to remember the
   order and re-run each file by hand.
2. **Flag/resolve the synthetic-data trap in `benchmark_tracking_error_model.py`**
   before anyone treats its output as real — right now it's the only script
   in either project whose output *looks* like a real risk number but is
   built on fabricated data, which is the one thing in this whole map worth
   fixing first.
