# LATENT

**Results page (charts, methodology, architecture map): [akirp002.github.io/LATENT](https://akirp002.github.io/LATENT/)**

## The problem

Most factor-based portfolio tools stop at the standard 5 Fama-French factors
(market, size, value, profitability, investment). That's a reasonable
baseline, but it leaves an obvious question unanswered: **how much of an
asset's or portfolio's return is actually explained by macro conditions
— growth, inflation, rates, monetary policy, currency, credit, commodities —
that FF5 doesn't capture at all?**

This repo answers that in two stages:

1. **Build the macro factor set and stress-test it.** Before trusting any
   macro-augmented beta, check the obvious failure modes first: is the
   macro panel redundant with itself (collinearity), is any factor's
   cross-sectional signal distinguishable from noise (weak-factor
   shrinkage), and does correcting for return autocorrelation change the
   answer (HAC vs. OLS)? Get those diagnostics right before reading
   anything into the betas themselves.
2. **Use the resulting factor set for something concrete: benchmarking a
   real portfolio.** Construct a strategic benchmark, measure a sample
   portfolio's factor exposure against it, and decompose tracking error
   into "systematic style drift" vs. "idiosyncratic single-position risk."

`factor-research/` does stage 1 (the general-purpose research engine);
`portfolio-benchmarking/` does stage 2, consuming stage 1's output rather
than rebuilding it. See [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md) for
a module-by-module walkthrough of what each script does and why,
[`docs/project-architecture-map.md`](docs/project-architecture-map.md) for
the dependency diagram and exact run order, and
[`docs/factor-pipeline-math.html`](docs/factor-pipeline-math.html) for the
math behind each stage.

## Key results so far

All numbers below are read directly off this repo's committed output files
— nothing here is illustrative. Full charts and captions:
[docs/index.html](https://akirp002.github.io/LATENT/).

**Is the 14-factor macro panel (5 FF + 9 macro PCA) trustworthy?**

| Diagnostic | Result | Verdict |
|---|---|---|
| Multicollinearity (VIF, all 14 factors) | max VIF 2.17 (Mkt-RF) | Well under the 5–10 conventional flag — collinearity is not inflating any factor's beta. |
| Weak-factor shrinkage (empirical-Bayes τ̂, 33 asset-class ETFs) | 5 of 14 factors (CMA, Credit_Risk, Rates_Yields, US_Monetary_Policy, Global_Monetary_Policy) shrink to τ̂≈0 | Their apparent cross-sectional spread in beta is noise, not real signal — don't read economic meaning into those betas. |
| Autocorrelation correction (HAC/Newey-West vs. OLS significance) | 16 of 462 asset–factor pairs flip significant one way, 10 the other | Autocorrelation isn't the reason some factors look weak; the shrinkage result above is the real explanation. |

**Sample-portfolio benchmark (fabricated demo data, real pipeline)**

Strategic benchmark: long-only max-Sharpe over FF5, 60–70% Mkt-RF
constraint → converged to 60% Mkt-RF / 40% RMW. Buy-and-hold, 2006–2026
(246 months):

| Metric | Value |
|---|---|
| Realized CAGR | 9.30% (vs. 8.92% expected static) |
| Annualized volatility | 9.95% |
| Sharpe ratio | 0.78 |
| Max drawdown | -19.88% |
| Gut-check vs. 8–12% expected CAGR | passes |

Tracking a demo two-account portfolio with one concentrated recent-IPO
position (PLTR) against that benchmark:

| | Full portfolio (incl. concentrated position) | Ex-concentrated |
|---|---|---|
| Ex-ante TE (analytic, FF5 beta delta) | 15.85% | 3.37% |
| Ex-post TE (realized, 69-month overlap) | 36.24% | 3.77% |

The ~9x gap between the two portfolios' realized TE — and the fact that
ex-ante TE undershoots the full portfolio's realized number — makes the
same point concretely: a single concentrated position, not systematic
factor drift, is what's driving this portfolio away from its benchmark.
The ex-concentrated version tracks tightly on both measures.

A separate rolling-beta / Ledoit-Wolf tracking-error prototype
(`benchmark_tracking_error_model.py`, deliberately-synthetic portfolio with
built-in style drift) shows the shrinkage estimator pulling ex-ante TE from
a 3.40% sample-covariance estimate to 3.76% (shrinkage intensity 0.063),
against a realized 9.12% — consistent with that portfolio's engineered
drift not being fully priced by a static covariance estimate.

## Repository structure

```
LATENT/
├── factor-research/        Stage 1: macro + FF5 factor panel, VAR, per-asset
│                            regressions (OLS/HAC), VIF, empirical-Bayes
│                            shrinkage, and a Bayesian time-varying-beta
│                            model (in progress).
├── portfolio-benchmarking/  Stage 2: benchmark construction + tracking-error
│                            pipeline, demonstrated on a fabricated sample
│                            portfolio. All data in this folder is fabricated.
├── docs/                    Published results: docs/index.html (chart
│                            gallery), factor-pipeline-math.html/.pdf
│                            (methodology), project-architecture-map.md
│                            (script-by-script dependency map). Served via
│                            GitHub Pages from this branch's /docs folder —
│                            this is the single source for both docs; the
│                            research/benchmarking folders link here rather
│                            than keeping their own copies.
└── scripts/                 generate_showcase_charts.py regenerates
                             docs/assets/*.png from the CSVs the two
                             pipelines above produce.
```

## Running it

```
pip install pandas numpy scipy statsmodels scikit-learn yfinance openpyxl matplotlib pymc pytensor
```

There is no orchestrator (no Makefile / Airflow DAG / run-all script) — see
[`docs/project-architecture-map.md`](docs/project-architecture-map.md) for
why, and for the exact run order. Short version:

- **`factor-research/`**: `fetch_macro_fred.py` (needs a `FRED_API_KEY`) +
  `pull_yfinance_data.py` + `transform_data_ff.py` build the base panel,
  then `quant_model_macro_1.py` → `2.py` → `3.py`/`4.py` → `vif_check.py`
  in that order (each depends on the previous step's output).
- **`portfolio-benchmarking/`**: `generate_sample_holdings.py` →
  `fetch_holdings_prices.py` → `construct_benchmark_mvo.py` →
  `portfolio_factor_exposures.py`. This reads `factor-research/`'s FF5
  output, so that project's data-build step must run first (its output is
  already committed in this repo, so a fresh clone can run
  `portfolio-benchmarking/` immediately without rebuilding stage 1).

Each script is a standalone `if __name__ == "__main__":` entry point — run
individually, in the order above.

**Known limitation:** several `factor-research/` scripts write output to a
hardcoded absolute Windows path rather than a path relative to the repo —
see each script's `DATA_PATH`/`OUT_PATH` constants before running them
somewhere other than the original machine.
