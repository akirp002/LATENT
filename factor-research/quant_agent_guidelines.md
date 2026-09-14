# Agent Guidelines: Quant Modeling & Data Engineering

Context file for agentic coding sessions (Claude Code, Cowork, etc.) on quant/data
engineering work. Pairs with the general CLAUDE.md behavioral rules (think-before-coding,
simplicity-first, surgical edits, goal-driven execution) — this file adds domain-specific
judgment calls for finance/econometrics code.

## 0. Prime directive: Occam's Razor

For any modeling or engineering choice, default to the simplest method that is
**statistically/numerically adequate** for the question being asked — not the most
sophisticated method available.

- Before reaching for a Kalman filter, MCMC sampler, or particle filter, ask: does OLS,
  a rolling window, or a closed-form estimator already answer the question? Add
  state-space/Bayesian machinery only when there's a specific reason simpler estimators
  fail (latent state, missing data, regime-dependent dynamics, nonlinearity that a linear
  filter can't capture).
- Flag it explicitly when a request seems to reach for something advanced (particle filter,
  ensemble methods, deep learning) where a simpler model would likely perform comparably —
  this is a known bias risk for a quant background, not just a coding-style nitpick.
  Say so, propose the simpler baseline first, and let the more complex approach be justified
  by a documented gap in the baseline (residual autocorrelation, poor OOS fit, known
  nonlinearity), not by default.
- Corollary: build the baseline first, always. A naive/simple model (random walk, equal
  weight, single-factor OLS) is the benchmark every "smarter" model must beat before it's
  worth the added complexity and fragility.

## 1. Two tracks: research vs. production

Most of AJ's work splits into two modes with different bars for code quality. Identify
which track a task is on before writing code, and say which one you're assuming.

**Research / exploratory track** (model development, backtests, one-off analysis,
dissertation-style estimation): optimize for iteration speed and correctness of the math.
Notebook-style is fine. Global state and hardcoded indices are fine *if* the alternative
is premature abstraction that slows down exploration. Still keep: clear variable names for
anything that isn't standard math notation, a short docstring or comment block stating what
the estimator/model is and its assumptions, and a way to rerun the whole thing end-to-end
without hidden manual steps.

**Production / pipeline track** (anything feeding a dashboard, a scheduled job, another
team, or a live book): held to the CLAUDE.md standard in full — tests, error handling for
real failure modes only, idempotent runs, logging at pipeline boundaries, no notebook-style
cell dependencies. Don't let production code inherit research-code habits (magic indices,
mutable globals, silent NaN handling) just because a prototype worked that way.

If a task starts in research mode and is about to get reused (someone else runs it, it
schedules on a cadence, it touches account-level or client-facing data), flag the track
change explicitly before continuing — that's a different quality bar, not just "more code."

## 2. Data engineering

- Validate schema and ranges at ingestion boundaries (nulls, dtypes, date ranges,
  duplicate keys), not deep inside business logic — fail fast and close to the source.
- Prefer vectorized pandas/numpy/Spark operations over row-wise loops; if a loop over rows
  or dates is unavoidable, say why (path-dependent state, e.g. a filter recursion) rather
  than defaulting to it.
- Idempotency for anything scheduled (Airflow, batch jobs): reruns on the same input should
  produce the same output, and partial failures shouldn't leave partial writes.
- Joins/reconciliation logic: state the grain (one row = what?) before writing the join, and
  assert the row count or key uniqueness after it. Silent fan-out from a join is the most
  common source of reconciliation bugs.
- Spark/Scala vs. pandas: default to pandas/numpy for anything that fits in memory on a
  single node. Reach for Spark only when the data genuinely doesn't fit, or the job already
  lives in a Spark pipeline — don't introduce distributed-computing complexity for
  desk-sized data.

## 3. Quant modeling

**Filtering / state-space work** (Kalman, ensemble Kalman, particle filters):
- Separate the transition equation, observation equation, and update step into named
  functions/blocks even in research code — this is the single biggest readability win for
  filter code and catches sign/dimension errors early.
- Use library implementations (`statsmodels`, `filterpy`, `pykalman`) unless the model
  structure genuinely requires a custom filter (nonlinear, regime-switching, non-Gaussian).
  Custom filters are where subtle bugs hide longest — don't hand-roll one to save an import.
- Validate against a simulated-data ground truth or a closed-form special case before
  trusting output on real data.
- Avoid manual matrix inversion (`inv(X) @ Y`) where a solve or Cholesky-based approach is
  numerically better-conditioned (`solve`, `lstsq`, `cho_solve`) — flag this even in research
  code if the matrices are ill-conditioned or the model will get reused.

**Factor models / forecasting (VAR, SARIMAX, GARCH, factor risk models):**
- State the estimation window and rebalance/refresh frequency explicitly; this is usually
  the single biggest driver of results, more than model choice.
- Backtest hygiene: no lookahead — features and fitted parameters at time *t* must only use
  information available at *t*. Walk-forward or expanding-window validation, not a single
  in-sample fit evaluated on the same data.
- Report an out-of-sample baseline comparison (equal-weight, random walk, prior-period
  persistence) alongside any model result — a Sharpe/Sortino number in isolation is not
  evidence the model adds value.

**Portfolio optimization / simulation:**
- Constraints (long-only, box constraints, turnover limits) belong in the optimizer
  formulation, not as post-hoc clipping/renormalization of weights — clipping after the
  fact silently breaks whatever objective was being optimized.
- Keep the objective function and the constraint set easy to read independently of the
  solver call, so the economic assumptions are auditable without tracing solver internals.

## 4. Library selection (default choices, override only with a reason)

- Optimization: `cvxpy` for convex portfolio problems, `scipy.optimize` for general
  nonlinear/MLE problems.
- Time series: `statsmodels` (VAR, SARIMAX, state-space) before custom implementations;
  `arch` for GARCH family.
- Filtering: `filterpy`/`pykalman` before hand-rolled Kalman/particle filters.
- Covariance estimation: `sklearn.covariance` (Ledoit-Wolf, OAS) rather than manual
  shrinkage formulas, unless the shrinkage target is nonstandard.
- GPU (`cupy`, etc.): only when profiling shows CPU is the actual bottleneck and the
  problem is large enough (big Monte Carlo draws, large matrix chains) to justify it —
  default to numpy/CPU and add GPU as an optimization, not a starting point.

## 5. Verification bar

Before calling modeling code "done":
- A known-answer or simulated-data test that the estimator recovers true parameters
  within a reasonable tolerance.
- An out-of-sample check, not just in-sample fit.
- For anything with financial output (returns, weights, risk numbers): a sanity check
  against a naive benchmark and against economic intuition (does a rate-hike shock
  actually move duration risk the right direction?).

## 6. Versatility note

Match tooling to the actual constraint of the task — dissertation-style state-space
econometrics, production Spark pipelines, and quick single-file portfolio scripts are all
in scope, and the right answer for one is often the wrong answer for another. When unsure
which mode fits, ask or state the assumption, per the general think-before-coding rule —
don't default to the most familiar tool if a simpler one fits the actual problem better.
