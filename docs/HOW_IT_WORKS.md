# How the code works

This is a walkthrough of the two pipelines, module by module, in the order
data actually flows through each. See
[`project-architecture-map.md`](project-architecture-map.md) for the
dependency diagram and exact run order; this file is the "what each step
actually does and why" companion to that map.

## Part 1 — `factor-research/`: build the macro factor set and stress-test it

### 1. `fetch_macro_fred.py`, `pull_yfinance_data.py`, `transform_data_ff.py` — pull the raw data

- **`fetch_macro_fred.py`** pulls ~25 macro series from FRED (needs a
  `FRED_API_KEY`) across growth, inflation, rates, monetary policy,
  currency, credit, and commodities, resamples everything to month-start,
  and merges it onto the Fama-French factor file. Series that exist on
  the wishlist but aren't available on FRED (proprietary/other-vendor
  data) are documented at the bottom of the file rather than silently
  dropped.
- **`pull_yfinance_data.py`** downloads monthly total returns for a broad
  lineup of ~34 asset-class ETFs (US/international/EM equity by size and
  style, fixed income, real assets) via yfinance and saves one wide
  `Date x Ticker` CSV — this becomes the dependent-variable panel for the
  factor regressions in steps 3–4 below.
- **`transform_data_ff.py`** reads the raw Fama-French 5-factor CSV,
  strips the annual-data section the file appends after the monthly rows,
  and parses the dates.

Output: `Data/FF_plus_macro_workable.csv` (5 FF factors + RF + ~25 raw
macro series) and `Data/asset_class_monthly_returns.csv`.

### 2. `quant_model_macro_1.py` — category PCA on the macro panel

A single PCA across all ~25 macro series would just find "the biggest
common cyclical wobble" and drown out inflation- or credit-specific
variation. Instead, the series are grouped into 9 economically-meaningful
categories first (e.g. splitting `US_Growth` from `Global_Growth`, since
US industrial production and OECD's leading indicator moved together in an
early run while EA/China loaded differently — two distinct growth cycles,
not one) and PCA is run **within each category**, collapsing correlated
raw series into one clean factor per category.

Output: `Data/macro_pca_factors.csv` — the 9 macro PCA factors.

### 3. `quant_model_macro_2.py` — VARX: FF5 ~ its own lags + macro factors

A VAR where each of the 5 FF factors is modeled on its own and the other
FF factors' lagged values, plus the 9 macro PCA factors as exogenous
regressors (lag order chosen via AIC search). Also runs a parallel set of
univariate OLS baselines (one macro factor at a time, no lags) — a macro
factor that's significant alone but loses significance in the full VAR is
probably just correlated with another factor doing the real work, and vice
versa.

Output: `var_model_summary.txt`, `var_coefficients.csv`,
`var_fitted_values.csv`, `var_residuals.csv`, `var_model.pkl`, plus the
univariate comparison CSVs.

### 4. `quant_model_macro_3.py` then `quant_model_macro_4.py` — asset regressions

The mirror image of step 3: now each of the ~34 asset-class ETFs from
`pull_yfinance_data.py` is the dependent variable.

- **`quant_model_macro_3.py`**: standard FF5 regression per asset
  (multivariate, all 5 factors together) plus the same univariate
  one-factor-at-a-time baseline, both with HAC (Newey-West) standard
  errors since monthly returns are typically serially correlated.
- **`quant_model_macro_4.py`**: extends the same multivariate regression
  to the full 14-factor set (5 FF + 9 macro), reporting plain-OLS and HAC
  t-stats side by side. The point is narrower than step 3 — isolate how
  much of any "weak t-stat" picture is autocorrelation (which OLS SEs
  ignore) versus genuine estimation noise, now that VIF (next step) has
  already ruled out collinearity as a cause.

Output: `asset_ff_*` CSVs (step 3) and `asset_full14_*` CSVs (step 4) —
betas, t-stats (OLS and HAC), R², N_Obs per asset/factor.

### 5. `vif_check.py` — collinearity diagnostic

Computes Variance Inflation Factor for each of the 14 factors (regress
each on the other 13; VIF = 1/(1-R²)). This is what rules collinearity out
as the explanation for weak-looking betas elsewhere — the highest VIF
comes back at 2.17 (Mkt-RF), well under the conventional 5–10 flag.

### 6. `plot_quant_pca_factors.py`, `plot_quant_ff_factors.py` — visual QA

Leaf-node plotting scripts; no downstream consumer, run anytime after
their input CSV exists.

### 7. `tvp_ffbs_gibbs.py` — time-varying-beta model (not wired in yet)

A standalone Carter & Kohn (1994) Forward-Filtering Backward-Sampling
model with a Metropolis-within-Gibbs step for non-conjugate
hyperparameters: betas follow a random walk (`beta_t = beta_{t-1} + eta_t`),
observations are `y_t = X_t'beta_t + eps_t`. Priors: `beta_0` seeded from
the static hierarchical posterior, per-factor `tau_j ~ HalfStudentT` (needs
the MH step since it's non-conjugate), `sigma^2 ~ InverseGamma` (conjugate,
direct Gibbs draw). This is under review before running on real data —
intended eventual input is a per-asset hierarchical posterior seeded from
`quant_model_macro_4.py`'s betas (that hierarchical-fitting step itself
isn't a saved script yet).

### Not part of the pipeline

`Portfolio Optimization.py` — despite the name, a `cupy`/GPU Monte-Carlo
MLE scratch script (ARCH-model-flavored). Doesn't read or write anything
in `Data/` and isn't called by anything else.

---

## Part 2 — `portfolio-benchmarking/`: use the factor set to benchmark a portfolio

**All data in this folder is fabricated.** `generate_sample_holdings.py`
stands in for a private "load real brokerage export" step that isn't
included here.

### 1. `generate_sample_holdings.py` — the fabricated portfolio

Two made-up accounts, one large single-stock "concentrated position"
(PLTR, chosen specifically as a recent IPO — its short trading history is
a deliberate stress-test for the history-alignment logic downstream), full
weights and ex-concentrated-position weights (remaining holdings
renormalized to 100%).

Output: `Data/holdings_clean.csv`, `Data/holdings_tickers.txt`.

### 2. `fetch_holdings_prices.py` — price history

Downloads monthly adjusted-close prices for every holding via yfinance,
converts to monthly percent returns (same percent-not-decimal convention
as `factor-research/pull_yfinance_data.py`, so it lines up with the FF
factor file's scale before the regression step divides by 100).

Output: `Data/holdings_monthly_returns.csv`.

### 3. `construct_benchmark_mvo.py` — the strategic benchmark

Constructs a single, fixed (never rebalanced), long-only weight vector
over the 5 FF factor return streams via mean-variance (max-Sharpe)
optimization: weights sum to 1, `0.60 ≤ w_Mkt-RF ≤ 0.70`, full-sample
(unshrunk) covariance. Only the 5 FF factors are eligible — they're the
only factors in this project that are actual investable long-short spread
portfolios; the 9 macro PCA factors are standardized z-scores of indicator
levels with no security that pays "the Inflation Factor return," so they
stay useful for *explaining* moves but aren't held. The resulting weight
vector (60% Mkt-RF / 40% RMW) is applied as true buy-and-hold: each
factor's $1 wealth index compounds independently and the benchmark's
aggregate wealth is their weighted sum, so effective weights drift over
time exactly as a real unrebalanced portfolio's would.

Output: `Data/benchmark_mvo_weights.csv`, `benchmark_wealth_curve.csv`,
`benchmark_effective_weights.csv`, `benchmark_mvo_summary.txt` (CAGR, vol,
Sharpe, max drawdown, and a gut-check against an 8–12% expected CAGR
range).

### 4. `portfolio_factor_exposures.py` — the core analysis

1. **Per-holding regression**: each ticker's excess return on the 5 FF
   factors, HAC standard errors, over whatever history yfinance returned
   for it.
2. **Portfolio-level exposure**: weighted sum of per-holding betas, both
   full-portfolio and ex-concentrated-position.
3. **Benchmark exposure**: the fixed 60/0/0/40/0 vector from step 3 —
   constructed, not estimated, so no regression needed there.
4. **Active exposure**: `db = beta_portfolio - beta_benchmark`.
5. **Ex-ante tracking error**: `TE = sqrt(db' Sigma db)`, annualized, same
   full-sample covariance convention as the benchmark.
6. **Ex-post tracking error**: reconstruct what the current weights would
   actually have earned each month (common-history overlap across
   holdings), compare to the benchmark's realized return series, take the
   annualized std of the difference — a real check on the regression-based
   ex-ante number, not just a restatement of it.

Output: `Data/holding_betas.csv`, `portfolio_factor_exposures.csv`,
`portfolio_active_returns.csv`, `tracking_error_vs_benchmark.txt`.

### 5. `benchmark_tracking_error_model.py` — separate rolling-beta / Ledoit-Wolf prototype

**Uses its own, separately-fabricated synthetic portfolio** with
deliberate style drift built in (rising HML and a macro-factor tilt over
time) — it does not consume pipeline A's sample holdings. Exercises
machinery pipeline A doesn't:

1. **Style-regression benchmark**: full-sample OLS (HAC) of the portfolio
   on the factor set; the fitted, alpha-stripped part is the "systematic
   benchmark," and `Active_t = Portfolio_t - Benchmark_t` is the ex-post
   tracking-difference series.
2. **Rolling betas**: the same regression re-fit on a rolling 36-month
   window, restricted to just the 5 FF factors — the 9 macro factors are
   deliberately dropped here because 36–60 monthly observations can't
   identify 14 collinear regressors per window (verified empirically: the
   "current vs. policy" delta becomes dominated by estimation noise, not
   real drift, once all 14 are forced into a short window).
3. **Ledoit-Wolf covariance**: shrinkage-estimated covariance of the FF5
   returns, used as the risk model for ex-ante TE — the one place in
   either pipeline that uses a shrunk (rather than full-sample) covariance
   matrix, since this script is specifically exercising machinery meant
   for a live/rolling portfolio.

Swapping `load_portfolio_returns()` for a real weighted return series
(see the commented-out alternative inside that function) turns this into
a second, more sophisticated TE model — nothing else in the script needs
to change.

Output: `Data/synthetic_portfolio_returns.csv`, `rolling_betas.csv`,
`ledoit_wolf_covariance.csv`, `rolling_realized_te.csv`,
`benchmark_static_betas.csv`, `benchmark_fitted_vs_active.csv`,
`benchmark_regression_summary.txt`, `tracking_error_summary.txt`.

---

## `scripts/generate_showcase_charts.py` — turning both pipelines' output into the results page

Reads the CSVs both pipelines above produce and regenerates the PNGs in
`docs/assets/` — no new modeling happens here, just presentation. Two
charts are computed inline rather than read from a saved CSV:

- **VIF by factor** — reads `vif_check.py`'s inputs directly and replots them.
- **Empirical-Bayes τ̂ by factor** — a normal-normal shrinkage estimate
  computed on the fly from `asset_full14_betas_tstats_long.csv`:
  `τ̂² = max(0, cross-sectional Var(β̂) − mean SE²)` per factor across the
  33 assets. A factor whose cross-sectional spread in beta is no bigger
  than its own estimation-noise variance shrinks to `τ̂ ≈ 0` — meaning its
  apparent variation across assets is noise, not real signal. This is the
  basis for the "5 of 14 factors are pure noise" result in the README.

Rerun this script after any upstream change to keep the showcase in sync
with the underlying data.
