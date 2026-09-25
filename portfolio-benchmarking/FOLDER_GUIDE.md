# Folder Guide — LATENT_V2

What's actually here, what it's for, and what's live vs. dormant. Written for
managing the household portfolio over a 20-25 year horizon — the short
version is: **one folder (`portfolio-benchmarking/`) does the real work,
`factor-research/` feeds it one file, and most individual scripts inside
`portfolio-benchmarking/` are either a one-time setup step, a demo you can
ignore, or a step in the real pipeline.**

---

## Top-level folders

```
LATENT_V2/
├── factor-research/          Stage 1: general-purpose macro + Fama-French
│                              factor research engine. Feeds one CSV into
│                              portfolio-benchmarking; nothing flows back.
├── portfolio-benchmarking/    Stage 2: THE ACTIVE WORK. Real household
│                              holdings -> benchmark -> tax/CVaR-aware trade
│                              proposals -> dashboard. Also contains an
│                              older fabricated-data demo, kept for reference.
├── docs/                      Published GitHub Pages site (charts,
│                              methodology, architecture map) for the
│                              factor-research side of the project.
└── scripts/                   One script that regenerates docs/'s chart PNGs
                                from committed CSVs. Not part of the real
                                pipeline.
```

**Only one dependency crosses folders**: `portfolio-benchmarking/` reads
`factor-research/Data/FF_plus_macro_workable.csv` (the 5 Fama-French factors
+ risk-free rate). Everything else in `factor-research/` — the macro PCA,
VAR model, asset-class regressions, VIF diagnostics, the Bayesian
time-varying-beta prototype — is research-track exploration that nothing in
the real portfolio pipeline touches.

---

## `factor-research/` — mostly not relevant to the real portfolio

This folder builds and stress-tests a macro + FF5 factor panel. It's the
"is this factor set trustworthy" research track. For the household's actual
portfolio you only ever need its *output*, not its internals:

| File | Relevant? | Why |
|---|---|---|
| `Data/FF_plus_macro_workable.csv` | **Yes — the only thing that matters** | The FF5 + RF panel `portfolio-benchmarking/` reads directly. |
| `fetch_macro_fred.py`, `pull_yfinance_data.py`, `transform_data_ff.py` | Only if rebuilding the panel | Build that one CSV from raw FRED/Yahoo data. Already run once; output is committed. |
| `quant_model_macro_1–4.py`, `vif_check.py`, `tvp_ffbs_gibbs.py`, plot scripts | **No** | Macro PCA, VAR, asset-class regressions, collinearity checks, a Bayesian time-varying-beta model still under review. All exploratory, none of it feeds the household portfolio. |
| `Portfolio Optimization.py` | **No** | Despite the name, an unrelated GPU/Monte-Carlo scratch script. Doesn't read or write anything used elsewhere. |

**Practical implication**: you almost never need to touch this folder. The
one exception is if you deliberately want to rebuild the factor panel (new
FF data vintage, different macro series) — then `fetch_macro_fred.py` →
`pull_yfinance_data.py` → `transform_data_ff.py` regenerates the CSV
`portfolio-benchmarking/` depends on, and every downstream number
(benchmark, trade proposals, dashboard) changes on the next run.

---

## `portfolio-benchmarking/` — where the real work happens

Three things live in this one folder: the real household pipeline (A), an
older fabricated-data demo (B), and a third-track what-if demo (C). Only A
is real.

For a diagram of exactly how the real vs. demo tracks branch and reconverge
(including which functions compute CVaR/tracking error and where each
track's output lands), see
[`docs/canonical-workflow.mmd`](../docs/canonical-workflow.mmd)/`.html` for
the whole pipeline and
[`docs/cvar-te-workflow.mmd`](../docs/cvar-te-workflow.mmd)/`.html` for the
CVaR/TE computation specifically — this file stays the script-by-script
reference; those diagrams are the visual map.

### A. The real household pipeline (what matters)

Six scripts, run in this order, turn a brokerage export into trade
proposals and a dashboard:

| # | Script | What it does |
|---|---|---|
| 1 | `generate_sample_real_holdings.py` | Fabricates a two-account household holdings input in the same shape a real brokerage export would take (Joint taxable + IRA, split by account label), including a concentrated single-stock position with a large unrealized gain, and computes ST/LT unrealized gain per position. |
| 2 | `fetch_household_holdings_prices.py` | Pulls monthly price history for every held ticker from yfinance. **Needs internet.** |
| 3 | `construct_benchmark_mvo.py` | Builds the policy benchmark: max-Sharpe long-only weights over the 5 FF factors (60–70% Mkt-RF constraint), held buy-and-hold. This is what tracking error is measured against. |
| 4 | `optimize_tax_cvar_mvo.py` | **The FF5/CVaR optimizer.** Minimizes tracking error (plus a turnover penalty) subject to an Active CVaR floor (tail-risk limit) and an annual capital-gains tax budget. Those two constraints don't both fit this year given MSTR's concentration, so it also produces relaxed alternatives (Paths A/B) and MSTR-glide-path alternatives (Paths C/D, 10%/5% sale caps). |
| 5 | `optimize_saa_mvo.py` | Independent all-equity SAA optimizer. Reads the root `all_equity_saa_covariance.xlsx`, minimizes covariance-based tracking error to its policy target (0% single stock), and writes the additional `Path SAA` proposal using the same tax budget and trade conventions. |
| 6 | `portfolio_risk_attribution.py` | Explains *why* each FF5/CVaR path's TE and CVaR are what they are — breaks both down by Fama-French factor, plus cross-sectional factor-tilt z-scores. Pure analysis, no re-optimizing. |
| 7 | `generate_rebalance_dashboard.py` | Builds `rebalance_dashboard.html` — the actual deliverable — from the saved CSV/JSON outputs, including the independent `Path SAA` trade proposal. No optimization happens here. |

Shared CVaR/tracking-error math lives in `analytics/risk.py`
(`scenario_matrix`, `active_cvar`, `tracking_error`, `BenchmarkIdentity`) and
is validated by `analytics/validation.py` (`python -m analytics.validation`
from this folder). `optimize_tax_cvar_mvo.py`'s CVaR scenarios/floor call
into `analytics.risk` directly rather than keeping its own copy of the math;
its tracking-error objective is still a local factor-covariance quadratic
form (see `docs/HOW_IT_WORKS.md`).

**`run_rebalance_pipeline.py`** (new) runs all seven in order, skips steps
whose output is already up to date, and refuses to run a step whose inputs
are missing. Use it instead of remembering the order by hand:

```
python run_rebalance_pipeline.py --list       # see the pipeline, no execution
python run_rebalance_pipeline.py --dry-run    # what would run vs. skip
python run_rebalance_pipeline.py              # run it
python run_rebalance_pipeline.py --offline    # skip the network price fetch
```

### B. Fabricated-data demo (safe to ignore)

| Script | What it does |
|---|---|
| `generate_sample_holdings.py` | Makes up a fake two-account portfolio (fictional tickers, one concentrated recent-IPO position) — exists only to exercise the pipeline before real data existed. |
| `fetch_holdings_prices.py` | Price fetch for that fake portfolio. |
| `portfolio_factor_exposures.py` | Factor-exposure/tracking-error calc for that fake portfolio. |
| `benchmark_tracking_error_model.py` | A separate rolling-beta / Ledoit-Wolf tracking-error prototype, exercised on its own *second* fake portfolio with engineered style drift. Useful as a template if you ever want rolling (not static) betas, but not wired to the real pipeline. |

None of these read or write anything the real pipeline (A) uses. They're
kept as a reference/test harness, not because they're part of managing the
actual portfolio. **Never mistake their output files for real numbers** —
they live in the same `Data/` folder, so check the filename (`holdings_`
vs. `holdings_household_`, `benchmark_static_betas.csv` vs.
`benchmark_mvo_weights.csv`, etc.).

### C. `demos/collar-strategy/` — MSTR option-collar what-if (safe to ignore)

A third, independent demo track — not fabricated data this time, but a
what-if layered on top of pipeline A's *real* optimizer output. Answers "if
we put a zero-cost option collar on the MSTR position, how much does Active
CVaR improve?" without re-running the optimizer or pricing real options.

| Script | What it does |
|---|---|
| `collar_cvar_analysis.py` | Reuses `optimize_tax_cvar_mvo.py`'s own CVaR scenario matrix (via `analytics.risk`) and clips MSTR's scenario return to a costless put/call collar, at four coverage levels. Cross-checks its uncollared baseline against `Data/path_metrics.json` so a stale trade CSV can't silently produce a wrong comparison. |
| `saa_benchmark_te.py` | Re-scores every path's already-optimized weights against the SAA sleeve/policy benchmark (`optimize_saa_mvo.py`'s formula) so Path C/D and SAA/SAA_3pct become TE-comparable, which they aren't on the dashboard (different benchmarks). |
| `generate_collar_html.py` | Builds `post_collar_trade_CVAR_Analytics.html`, a self-contained Chart.js report of CVaR by path x coverage level. |

These import `optimize_tax_cvar_mvo`/`optimize_saa_mvo` from
`portfolio-benchmarking/` directly (each script adds that folder to
`sys.path` at the top) — they read the real pipeline's saved outputs but
never re-optimize or write back into it. Formerly the top-level
`Option Collar Strategy/` folder; moved under `portfolio-benchmarking/demos/`
to keep all three tracks (real, fabricated-demo, what-if-demo) inside one
consistently-structured project instead of a fourth ad hoc top-level folder.

### `Data/` folder — everything both pipelines read and write

One flat folder, both pipelines' outputs (and the one manual input file)
sit here side by side:

- **Fabricated input**: `generate_sample_real_holdings.py` writes
  `holdings_household_clean.csv` / `holdings_household_tickers.txt` directly —
  there is no real brokerage export or workbook in this repo. Re-run that
  script (optionally editing its constants) to regenerate a different fake
  household scenario; everything downstream regenerates from it.
- **Household-pipeline outputs**: `holdings_household_*.csv`, `benchmark_mvo_*`,
  `trade_proposals_path{A,B,C,D,SAA}.csv`, `path_metrics.json`,
  `optimizer_summary.txt`, `risk_attribution.json`.
- **Demo-pipeline outputs**: `holdings_clean.csv`, `holdings_tickers.txt`,
  `holdings_monthly_returns.csv`, `holding_betas.csv`,
  `portfolio_factor_exposures.csv`, `synthetic_portfolio_returns.csv`,
  `rolling_betas.csv`, `ledoit_wolf_covariance.csv`, and their matching
  `*_summary.txt` files.

### Loose HTML files in this folder

- `rebalance_dashboard.html` — the real deliverable, regenerated by step 6.
- `household_portfolio_analytics.html` — an earlier/alternate dashboard build
  sitting alongside it; check its date and content before treating it as
  current, since `run_rebalance_pipeline.py` doesn't regenerate it.

---

## `docs/` and `scripts/`

- `docs/` — the published GitHub Pages results site for the
  **factor-research** side of the project (chart gallery, methodology
  write-up, architecture map). Not portfolio-management infrastructure;
  more of a research showcase.
- `scripts/generate_showcase_charts.py` — regenerates the PNGs in
  `docs/assets/` from committed CSVs. Only run this if you're updating the
  published showcase page, not as part of managing the actual portfolio.

---

## TL;DR — what to actually run, ever

For the real portfolio, over the next 20-25 years, you need exactly:

1. (Optionally) tweak `generate_sample_real_holdings.py`'s constants for a new scenario.
2. `python run_rebalance_pipeline.py`
3. Open `portfolio-benchmarking/rebalance_dashboard.html`.

Do this whenever the brokerage statement changes, at minimum once a year
(the tax budget and MSTR glide-path caps are annual), and after any deliberate
policy change (tax rates, CVaR floor, benchmark mix — all are named constants
near the top of `optimize_tax_cvar_mvo.py` and `construct_benchmark_mvo.py`).
Everything in `factor-research/` and the demo tracks of
`portfolio-benchmarking/` (B and C above) can be ignored unless you're doing
further research, rebuilding the underlying factor panel, or exploring the
MSTR collar what-if.
