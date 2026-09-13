# LATENT

A macro-factor research engine (`python-projects/`) and a factor-based
portfolio benchmarking / tracking-error pipeline (`portfolio-mgmt-demo/`).
See [`python-projects/project_architecture_map.md`](python-projects/project_architecture_map.md)
for how the two relate and how each script's outputs feed the next, and
[`python-projects/factor_pipeline_math.html`](python-projects/factor_pipeline_math.html)
for the math behind each stage.

## `python-projects/`

Builds a monthly panel of the 5 Fama-French factors plus 9 macro-driven
factors (category-level PCA over ~25 FRED series: growth, inflation, rates,
monetary policy, currency, credit, commodities), then fits a battery of
exploratory models against it: a macro-augmented VAR, per-asset FF5/14-factor
regressions with HAC (Newey-West) standard errors, VIF collinearity checks,
empirical-Bayes/ridge shrinkage on weak factor betas, and (in progress) a
time-varying-parameter factor model estimated via Carter & Kohn (1994)
forward-filtering/backward-sampling with a Metropolis-within-Gibbs step for
the non-conjugate hyperparameters.

## `portfolio-mgmt-demo/`

**All data in this folder is fabricated.** `generate_sample_holdings.py`
builds a synthetic two-account portfolio (made-up tickers, weights, and
dollar values) with one large recent-IPO "concentrated position" (PLTR), used
to exercise the rest of the pipeline: fetch real market prices for those
sample tickers, construct a max-Sharpe strategic benchmark over the 5 FF
factors, regress each holding on FF5, and compute both ex-ante (analytic) and
ex-post (realized) tracking error against the benchmark. `benchmark_tracking_error_model.py`
is a separate, still-synthetic prototype exercising a Ledoit-Wolf-shrunk,
rolling-beta version of the same tracking-error idea.

Run order: `generate_sample_holdings.py` → `fetch_holdings_prices.py` →
`construct_benchmark_mvo.py` → `portfolio_factor_exposures.py`. Both
`construct_benchmark_mvo.py` and `portfolio_factor_exposures.py` read the FF5
factor file from `../python-projects/Data/`, so that project's data-build
step needs to have run first (its Data files are included in this repo).

## Setup

```
pip install pandas numpy scipy statsmodels scikit-learn yfinance openpyxl matplotlib pymc pytensor
```

Each script is a standalone `if __name__ == "__main__":` entry point — run
them individually in the order noted above/in `project_architecture_map.md`.
There is no orchestrator; see that file's closing section for why.
