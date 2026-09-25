r"""
optimize_tax_cvar_mvo.py

Produces asset-allocation TRADE PROPOSALS for a fabricated household
portfolio (Data\holdings_household_clean.csv, written by
generate_sample_real_holdings.py -- see that script's docstring for what's
fabricated), by solving a tax-aware mean-variance optimization. Two paths
are solved -- Path C and Path D, differing only in how much of the current
MSTR position may be sold this year (see MSTR_CAPPED_PATHS in __main__). Active CVaR is
still COMPUTED and reported for both (and for the do-nothing baseline) so
it stays visible in the dashboard, but it is not a constraint on either
path's solution -- the `solve()` function below still supports a CVaR-floor
constraint (`enforce_cvar=True`) as general machinery, it's simply switched
off for both paths currently solved here.

=====================================================================
THE PROBLEM BEING SOLVED
=====================================================================

Decision variables: one portfolio weight x[ticker, account] for every
(ticker, account) pair currently held -- e.g. x[ITOT, Joint] and
x[ITOT, IRA] are separate variables even though ITOT is the same security,
because a trade in the taxable Joint account has tax consequences and a
trade in the IRA does not. Weights are fractions of TOTAL HOUSEHOLD
market value (Joint + IRA combined), so they all live on one common scale
and sum to 1.

We do NOT introduce any ticker that isn't already held (no new assets) --
this is a reweighting of the existing 24-position, 16-unique-ticker book,
per a fixed scoping decision made before building this (no new assets are ever introduced).

Objective -- minimize tracking error vs. the policy benchmark
---------------------------------------------------------------
The benchmark is the existing 60% Mkt-RF / 40% RMW Fama-French policy mix
built in construct_benchmark_mvo.py. Each ticker's exposure to the 5 FF
factors is estimated by its own HAC-robust OLS regression (same convention
as portfolio_factor_exposures.py). Aggregate portfolio factor exposure is
the ticker-level weighted sum of those betas (summed across BOTH accounts,
since factor risk doesn't care which account holds the security):

    beta_portfolio(x) = sum_(ticker,account) x[ticker,account] * beta[ticker]
    active_exposure    = beta_portfolio(x) - beta_benchmark
    TE(x)^2            = active_exposure' @ Sigma @ active_exposure
                          + w_ticker(x)' @ Omega @ w_ticker(x)

Sigma is an annualized 50/50 blend of the raw full-sample FF5 covariance
and a 12-month-half-life EWMA covariance.

The second term is the idiosyncratic (FF5-residual) contribution to TE.
w_ticker(x) is the FULL portfolio weight per ticker (summed across
accounts) -- NOT active weight -- because the benchmark is a pure factor
portfolio (60% Mkt-RF / 40% RMW) that holds no securities, so its own
idiosyncratic variance is exactly zero; every dollar the household holds
therefore contributes idio risk with no offsetting benchmark term. Omega is
the annualized FF5-residual COVARIANCE across tickers (see
`residual_covariance()`), not just each ticker's own residual variance on
the diagonal: several of the held ETFs (ITOT/IWD/IWF/IJJ/IJK/IJT/IJS/SCHD/
QUAL/MTUM) hold heavily overlapping US-equity constituents, so their FF5
residuals are materially correlated with each other -- the same
factor-unexplained variation showing up in each of them, not independent
draws -- and a diagonal-only term would understate that block's true risk.
Likewise IEMG/IEFA (both missing an EM/intl factor) and SCHH/SHV (missing a
real-estate/duration factor) are "residual" only in the sense that FF5
doesn't span their real systematic driver, not because they're truly
idiosyncratic single-name risk -- their residuals correlate with each
other for exactly that reason, which the full covariance (not the
diagonal) captures. MSTR is the one holding where the diagonal
approximation is close to right on its own -- a single stock's residual
genuinely is closer to independent of the ETFs' -- and it's also the
largest single idio contributor, so this term meaningfully strengthens the
case for trimming it, on top of the factor-TE case already made above.

TE(x)^2 (both terms) is a convex quadratic function of x, so "minimize TE"
is still a plain, well-behaved QP objective -- Omega, like Sigma, is a
covariance matrix and therefore PSD, and the shrinkage in
`residual_covariance()` keeps it that way.

Per-position turnover penalty
------------------------------
Minimum-tracking-error MVO with no trading cost is indifferent between a
small trim spread across the book and a wholesale liquidation into whichever
single holding happens to hedge best -- the classic corner-solution failure
mode of unconstrained MVO. The objective therefore carries an L1 penalty on
how far each position moves from its CURRENT weight:

    objective(x) = TE(x)^2 + lambda * sum_i |x_i - x0_i|

The penalty is per (ticker, account) ROW, not per ticker, so trimming ITOT in
the Joint account and trimming it in the IRA are charged separately -- they
are separate trades with separate costs. An L1 (not L2) penalty is the right
shape here: it leaves genuinely worthwhile trades near full size while
zeroing out the long tail of cosmetic few-hundred-dollar trades, which is the
behavior an advisor actually wants off the trade blotter.

|x_i - x0_i| is not differentiable, so it is linearized the same way the CVaR
and tax terms already are -- one slack per position with a two-sided bound:

    turn_i >= x_i - x0_i,   turn_i >= x0_i - x_i,   turn_i >= 0

and `lambda * sum_i turn_i` enters the objective. Because turn only ever
raises the objective, the optimizer drives each turn_i to its tight value, so
turn_i == |x_i - x0_i| at the optimum. The problem therefore stays a smooth
convex QP that SLSQP solves to the global optimum.

Note the penalty is a REGULARIZER, not a fee model: the tax constraint is
what prices selling, and lambda is tuned by its effect on the trade list
(see TURNOVER_LAMBDA).

Active CVaR -- computed and reported, NOT a constraint for Path C / Path D
----------------------------------------------------------------------------
Earlier versions of this script enforced "5th-percentile Active CVaR of the
total portfolio, annually, must not be worse than -15%" as a hard constraint.
That constraint has been REMOVED from both paths currently solved (Path C
and Path D solve `enforce_cvar=False`); Active CVaR is still computed and
reported for the baseline and every path, purely as context for the trade
decision, not as something either path is required to satisfy.

It is still estimated the same way, for anyone re-enabling the constraint via
`enforce_cvar=True`: empirically (no distributional assumption) from the
tickers' historical monthly returns, using overlapping 12-month rolling
windows over their common history -- the same "common overlap" convention
portfolio_factor_exposures.py uses for its ex-post tracking error, extended
here to build a set of annual return SCENARIOS rather than a single number.
For each rolling window s, the scenario active return is:

    Active_s(x) = sum_(months m in window s) [ r_portfolio,m(x) - r_benchmark,m ]

using the ARITHMETIC sum of the 12 monthly returns in the window (not
geometric compounding) as the annual-return proxy. At monthly-return
magnitudes this is a standard, materially accurate simplification, and
critically it keeps Active_s(x) an exactly LINEAR function of x -- which is
what would make a CVaR constraint a plain linear constraint instead of a
much harder non-convex one, if one were switched back on.

When enabled, CVaR is enforced via the Rockafellar-Uryasev (2000)
formulation, which turns "the average of the worst 5% of scenarios is >=
-15%" into an exact linear program with one extra scalar variable (zeta,
interpreted as the estimated Value-at-Risk cutoff) and one slack variable
per scenario:

    Loss_s(x) = -Active_s(x)                      (loss = negative return)
    u_s >= Loss_s(x) - zeta,   u_s >= 0            (one slack per scenario)
    CVaR(x) = zeta + (1 / (0.05 * S)) * sum_s u_s
    require: CVaR(x) <= 0.15                       (only when enforce_cvar=True)

This is exact (not an approximation) given the linear Active_s(x) above,
and it's a standard formulation -- no CVaR-specific library needed.

Constraint 1 -- Annual capital-gains tax budget
-------------------------------------------------
"Annual tax cost from realized gains, in the taxable account, must not
exceed 2% of the portfolio, assuming a ~25-33% long-term capital gains
rate." Only NET SALES in the Joint (taxable) account trigger a tax bill;
IRA trades and any buys are tax-free. The workbook gives each position's
already-realized-if-sold Short-term ($) and Long-term ($) unrealized gain
directly (no per-lot dates in this data), so a partial sale is modeled as
realizing that same ST/LT split, pro-rated by the dollar fraction sold:

    tax_per_dollar_sold[ticker] = (LT_Gain*TAX_LT + ST_Gain*TAX_ST) / Market_Value

    sell[ticker] = max(x0[ticker,Joint] - x[ticker,Joint], 0)   (weight reduction)
    Tax_Cost(x)  = sum_(Joint tickers) sell[ticker] * tax_per_dollar_sold[ticker]
    require: Tax_Cost(x) <= 0.02

TAX_LT defaults to 29.5% (midpoint of the given 25-33% range). Short-term
gains -- currently near-zero across this book -- are conservatively taxed
at an assumed ~40.8% top ordinary + NIIT rate (TAX_ST below); flagged here
because only a long-term rate range was assumed for this fabricated scenario. The max(.,0) is made exact
and linear the same way as the CVaR slack above: a slack variable
sell[ticker] >= x0 - x, sell[ticker] >= 0, which the optimizer will pin to
its tight value automatically since increasing it only tightens (never
loosens) the tax-budget constraint it feeds.

Cash reserved to actually PAY that tax bill: sum(x) == 1 - true_tax_cost(x)
------------------------------------------------------------------------
The tax BUDGET above caps what counts against a shared 2%/year ceiling
(and, for Paths C/D, excludes MSTR from that count entirely -- see
`exclude_from_tax_budget`). It says nothing about where the cash to pay
the tax actually comes from. The book's fundamental equality constraint
enforces that: sum(x) + max(x0 - x, 0)[joint] @ tax_vec == 1, where tax_vec
is the FULL, unexcluded per-dollar tax rate, computed directly from x (not
from the `sell` slack variable -- see invested_fun()'s docstring comment for
why that distinction matters). Every dollar of realized-gain tax owed --
even on a sale like MSTR's that's carved out of the budget cap -- is money
that leaves the sale proceeds and can't be reinvested, so the post-trade
book sits below 100% invested by exactly that much. That shortfall shows up
in the trade table as an explicit "CASH_RESERVE" row (see
build_trade_table()), not as an unexplained gap.

No new money into the IRA
--------------------------
x[ticker,account] <= x0[ticker,account] for every IRA row -- a plain
per-row upper bound, exactly like NO_ADD_TICKERS, just applied by ACCOUNT
rather than by ticker. An IRA can't receive an arbitrary buy the way a
taxable account can -- annual contribution limits (a few thousand dollars)
are far below what a reweighting-scale trade would need -- and unlike a
taxable account, even reallocating WITHIN the IRA (sell one position to
fund a buy of another) is disallowed: every IRA position may only be
trimmed or held. A ticker held ONLY in the IRA (e.g. XLB, SCHH) is therefore
excluded from Constraint 2's floor exactly like NO_ADD_TICKERS is -- there
is no row left anywhere that could still buy it up to MIN_TICKER_WEIGHT.
Always on, no toggle -- an account-type rule, not a tax or risk one.

Constraint 2 -- Minimum per-ticker allocation floor
-----------------------------------------------------
Every currently-held ticker keeps at least MIN_TICKER_WEIGHT (0.3%) of
household value, summed across both accounts if held in both:

    sum_(accounts) x[ticker,account] >= MIN_TICKER_WEIGHT

This is a structural floor, not a tax or risk rule -- its only job is to stop
the optimizer from fully zeroing out a small sleeve as an incidental
side-effect of chasing tracking error or the tax budget. It is a plain
linear constraint on x, so it costs nothing in terms of the problem's
convexity, and unlike the tax-budget constraint it has no on/off switch:
every solve enforces it -- EXCEPT for tickers in NO_ADD_TICKERS (see
Constraint 3 below), which are excluded from this floor by construction.

Constraint 3 -- No-add tickers (upper-bound only, no floor)
---------------------------------------------------------------
Some tickers may be trimmed or sold but never bought further:

    x[ticker,account] <= x0[ticker,account]     for ticker in NO_ADD_TICKERS

This is a plain per-row upper bound (`scipy.optimize.minimize`'s `bounds`
argument), not an inequality constraint, so it costs nothing extra in the
problem's structure. SHV is the current entry: its combined current weight
(0.298%) sits just under MIN_TICKER_WEIGHT (0.3%), so it is also excluded
from Constraint 2's floor -- requiring "no more than x0" AND "at least 0.3%"
on the same ticker would be infeasible by construction, and here the no-add
mandate is deliberate, not an artifact the floor exists to guard against.

Because all three constraints above are linear (or plain bounds) once the
introduced, and the objective (TE^2) is convex quadratic, the whole problem
is a convex QP. scipy.optimize.minimize(method="SLSQP") -- the same solver
already used in construct_benchmark_mvo.py -- finds the global optimum here
(no cvxpy dependency: this repo's environment is pinned to an EOL Python 3.9
stack, and installing cvxpy pulled in numpy 2.x / scipy 1.13, which broke
the pinned pandas -- not worth destabilizing the rest of the pipeline for).

One QP formulation, not several
---------------------------------
Path C and Path D differ only in their MSTR sale cap -- both are the SAME
underlying QP (`solve()` below), parameterized by
`objective`/`enforce_cvar`/`enforce_tax_budget`/`max_sell_frac`, rather than
separate hand-rolled QPs per path. `solve()` also still supports a CVaR-floor
constraint and a tax-cost-minimizing objective as general machinery -- unused
by anything currently invoked in `__main__`, but kept as one shared,
well-tested formulation rather than deleted and rewritten if a future
configuration needs either again. Collapsing what used to be several
near-duplicate constraint-building blocks into one function is a direct
application of this repo's Occam's-razor rule
(`factor-research/quant_agent_guidelines.md` #0): duplicated math surface
area is exactly where a sign error or index bug hides -- and exactly why the
MIN_TICKER_WEIGHT floor above only had to be added in one place.

=====================================================================
INPUTS
=====================================================================
  - Data\holdings_household_clean.csv            (generate_sample_real_holdings.py)
  - Data\holdings_household_monthly_returns.csv  (fetch_household_holdings_prices.py)
  - Data\benchmark_mvo_weights.csv          (construct_benchmark_mvo.py)
  - Data\benchmark_wealth_curve.csv         (construct_benchmark_mvo.py)
  - ..\factor-research\Data\FF_plus_macro_workable.csv

=====================================================================
OUTPUTS (Data\ folder)
=====================================================================
  - trade_proposals_pathC.csv one row per (ticker, account), INCLUDING rows
  - trade_proposals_pathD.csv with no change (Direction == "HOLD") -- current
                               vs. proposed weight/dollars, trade direction,
                               ticker description (Name/Description column,
                               straight from the brokerage export), realized
                               gain & tax cost where applicable
  - path_metrics.json         TE / CVaR (informational) / tax cost / turnover
                               per path, for the dashboard generator
  - optimizer_summary.txt     full plain-English readout: inputs used,
                               constraint status (binding/slack), before/
                               after TE, CVaR, and tax cost
"""

import json
import os

import numpy as np
import pandas as pd
from analytics.risk import active_cvar, scenario_matrix
import statsmodels.api as sm
from scipy.optimize import minimize

DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
FF_PATH = os.path.join(os.path.dirname(__file__), "..", "factor-research", "Data", "FF_plus_macro_workable.csv")

FF_FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

# Illustrative asset-class label per ticker -- a fixed, hand-maintained map,
# not derived from any data source. Shared by portfolio_risk_attribution.py
# (idiosyncratic-risk-by-class rollup) and generate_rebalance_dashboard.py
# (allocation-by-class views), so this is the one place it's defined. MSTR
# gets its own singleton "class" deliberately: lumping it into a style
# bucket would hide exactly the concentration this whole project exists to
# analyze.
ASSET_CLASS = {
    "MSTR": "Single Stock (Growth)", "SCHD": "US Dividend", "ITOT": "US Total Market",
    "IWD": "US Large Value", "IEMG": "Emerging Markets", "IJJ": "US Mid Value",
    "IJT": "US Small Growth", "IJS": "US Small Value", "QUAL": "US Quality Factor",
    "MTUM": "US Momentum Factor", "IEFA": "Intl Developed", "IWF": "US Large Growth",
    "IJK": "US Mid Growth", "SHV": "Short Treasury", "XLB": "US Materials", "SCHH": "US REIT",
}

MIN_OBS = 24
WINDOW = 12  # months per CVaR scenario ("annual")
CVAR_ALPHA_TAIL = 0.05  # worst 5% of scenarios
CVAR_FLOOR = -0.15  # 5th-pct active CVaR must be >= -15%
TAX_BUDGET = 0.02  # max 2% of household value / year in realized-gain tax
TAX_LT = 0.295  # blended LT cap-gains rate: midpoint of an assumed 25-33% range
TAX_ST = 0.408  # assumed top ordinary + 3.8% NIIT, for the (near-zero) ST portion
MIN_TICKER_WEIGHT = 0.003  # every held ticker keeps >= 0.3% of household value (see Constraint 3)

# Tickers that may be trimmed or sold but never bought further -- their row
# weight is bounded above by its CURRENT weight (x_i <= x0_i), for every path
# and every account. SHV's current weight (0.298%) sits just BELOW
# MIN_TICKER_WEIGHT (0.3%), so it is also excluded from the min-ticker-weight
# floor (Constraint 3) below: enforcing both "no more than x0" and "at least
# 0.3%" on the same ticker would be infeasible by construction, and the
# no-add mandate is the deliberate one here, not an optimizer artifact the
# floor exists to guard against.
NO_ADD_TICKERS = frozenset({"SHV"})

# Per-position L1 turnover penalty, lambda * sum_i |x_i - x0_i|. See the
# "Objective" section of the module docstring. Two constants because the two
# objective families are in different units and a single lambda cannot be
# interpreted in both:
#   TURNOVER_LAMBDA      -- against the TE^2 objective (squared annualized
#                           active return). A pure regularizer; the number has
#                           no dollar reading. 0.01 is the calibrated default.
#   TURNOVER_LAMBDA_TAX  -- against Path B's tax-cost objective, which is
#                           already in fraction-of-household-value units, so
#                           here lambda IS a round-trip transaction cost rate:
#                           0.001 = 10bps per dollar traded.
TURNOVER_LAMBDA = 0.02
TURNOVER_LAMBDA_TAX = 0.001

# Blend a fast, regime-sensitive covariance estimate with the slow historical
# estimate. A 12-month half-life lets the short component react within roughly
# one business cycle year while the raw full-sample covariance preserves the
# long-run factor relationships. The two components receive equal weight.
SHORT_COV_EWMA_HALFLIFE_MONTHS = 12
SHORT_COV_WEIGHT = 0.50
LONG_COV_WEIGHT = 1.0 - SHORT_COV_WEIGHT

# Weight on the DIAGONAL when shrinking the FF5 residual covariance (Omega)
# toward it -- 0.4 means 40% diagonal / 60% raw sample covariance. Not
# optional: with MSTR's ~2020 IPO, the common-history window across all 16
# tickers is thin (see residual_covariance()'s docstring), so the raw
# 16x16 sample covariance (136 free parameters) is noisy, especially
# off-diagonal. This is a placeholder intensity, not a derived one -- if
# sklearn is available, Ledoit-Wolf gives an analytic optimum; otherwise
# sweep e.g. {0.2, 0.4, 0.6} and check the trade list is stable across it,
# same discipline as TURNOVER_LAMBDA.
IDIO_SHRINKAGE = 0.4


# --------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------

def load_ff():
    raw = pd.read_csv(FF_PATH, parse_dates=["Date"]).set_index("Date")
    ff = raw[FF_FACTOR_COLS] / 100.0
    rf = raw["RF"] / 100.0
    return ff, rf


def blended_factor_covariance(ff, annualize=12):
    """Return equal-weighted long-run and short-run annualized covariances."""
    ff = ff.sort_index()
    long_cov = ff.cov()

    # Recursive EWMA covariance with a half-life in monthly observations.
    alpha = 1.0 - np.exp(np.log(0.5) / SHORT_COV_EWMA_HALFLIFE_MONTHS)
    mean = ff.iloc[0].to_numpy(float)
    short_cov = np.zeros((ff.shape[1], ff.shape[1]), dtype=float)
    for observation in ff.iloc[1:].to_numpy(float):
        delta = observation - mean
        mean = mean + alpha * delta
        short_cov = (1.0 - alpha) * (short_cov + alpha * np.outer(delta, delta))
    short_cov = pd.DataFrame(short_cov, index=ff.columns, columns=ff.columns)

    return annualize * (
        LONG_COV_WEIGHT * long_cov + SHORT_COV_WEIGHT * short_cov
    )


def hac_maxlags(n_obs):
    return int(np.floor(4 * (n_obs / 100) ** (2 / 9)))


def regress_tickers(returns_wide, ff, rf, tickers):
    """Per-ticker FF5 beta via HAC OLS, same convention as
    portfolio_factor_exposures.py. Returns (betas, idio_var):
      betas    -- DataFrame index=ticker, cols=FF5 (rows dropped for
                  insufficient history are simply absent from both outputs).
      idio_var -- Series index=ticker, the regression's ANNUALIZED residual
                  variance (mse_resid * 12, i.e. SSR/(n-k) so it's the
                  unbiased estimator net of the 6 fitted parameters, not a
                  naive .var()) -- the ticker's return variance the 5 FF
                  factors do NOT explain. This is what lets a risk
                  decomposition include an idiosyncratic term: total active
                  variance = factor-implied variance + sum_ticker
                  (active weight)^2 * idio_var, the standard structural
                  ("Barra-style") risk-model split. Used by
                  portfolio_risk_attribution.py; the optimizer itself still
                  minimizes only the factor-implied term (see that module's
                  docstring) -- adding idio variance to the QP objective is
                  a bigger, separate design change this doesn't make.
    """
    betas, idio_var = {}, {}
    for t in tickers:
        if t not in returns_wide.columns:
            print(f"  [betas] skipping {t}: no return history available")
            continue
        ret = returns_wide[t].dropna() / 100.0
        data = ff.join(rf.rename("RF")).join(ret.rename("Ret"), how="inner").dropna()
        n_obs = len(data)
        if n_obs < MIN_OBS:
            print(f"  [betas] skipping {t}: only {n_obs} aligned obs (< {MIN_OBS})")
            continue
        excess = data["Ret"] - data["RF"]
        x = sm.add_constant(data[FF_FACTOR_COLS])
        result = sm.OLS(excess, x).fit(cov_type="HAC", cov_kwds={"maxlags": hac_maxlags(n_obs)})
        betas[t] = result.params.drop("const")
        idio_var[t] = result.mse_resid * 12
    return pd.DataFrame(betas).T, pd.Series(idio_var)  # index=ticker, cols=FF5 / index=ticker


def residual_covariance(returns_wide, ff, rf, tickers, shrink=IDIO_SHRINKAGE):
    """Annualized FF5 residual COVARIANCE across `tickers` (Omega), shrunk
    toward its own diagonal. Unlike regress_tickers's per-ticker idio_var
    (each ticker's OWN residual variance, ignoring cross-ticker
    correlation), this is the full n_ticker x n_ticker matrix -- what the
    TE(x)^2 idio term in the module docstring actually needs, since several
    held tickers' residuals are expected to be correlated rather than
    independent (see that docstring).

    Estimated on the COMMON overlap across every ticker (same convention as
    build_cvar_scenarios), since a *covariance* needs contemporaneous
    residuals -- unlike regress_tickers, which uses each ticker's own full
    history independently. That means betas and Omega are technically
    estimated on different samples; with MSTR's ~2020 IPO the common window
    across 16 tickers may only run ~70 months, which is thin for a 16x16
    covariance (136 free parameters) -- exactly why `shrink` is not
    optional. `shrink=1.0` is the pure diagonal (same information as
    idio_var above); `shrink=0.0` is the raw, unshrunk sample covariance.

    Returns a DataFrame, index=columns=tickers, in that order.
    """
    ret = returns_wide[tickers].dropna(how="any") / 100.0
    data = ff.join(rf.rename("RF")).join(ret, how="inner").dropna()
    x = sm.add_constant(data[FF_FACTOR_COLS])
    resid = {}
    for t in tickers:
        excess = data[t] - data["RF"]
        resid[t] = sm.OLS(excess, x).fit().resid
    resid_df = pd.DataFrame(resid)[tickers]
    print(f"  [idio] residual covariance estimated on {len(resid_df)} common months "
          f"({resid_df.index.min().date()} -> {resid_df.index.max().date()}) across {len(tickers)} tickers")

    # Diagnostic, not used in the returned Omega: how correlated are the raw
    # residuals BEFORE shrinkage? This is the check the shrinkage intensity
    # (IDIO_SHRINKAGE) should actually be informed by -- if these come back
    # near zero, the diagonal-only idio_var from regress_tickers would have
    # been enough and this whole function is unnecessary.
    corr = resid_df.corr()
    pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    print("  [idio] residual correlation -- top 5 |pairs| (near 0 => diagonal would've been fine):")
    for (a, b), r in pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(5).items():
        print(f"           {a}/{b}: {r:+.2f}")

    omega_sample = resid_df.cov() * 12
    omega_diag = pd.DataFrame(np.diag(np.diag(omega_sample.values)), index=tickers, columns=tickers)
    return shrink * omega_diag + (1.0 - shrink) * omega_sample


def build_cvar_scenarios(returns_wide, bench_wealth, tickers, window=WINDOW):
    """Overlapping WINDOW-month scenario matrix. Returns:
      A: DataFrame, rows=ticker, cols=scenario id -> sum of that ticker's
         monthly returns (decimal) over the window (so Active_s(x) =
         bench_annual[s] - x_row_weights @ A[:, s] for x expressed per
         (ticker,account) with each row's return series equal to its
         ticker's, via a ticker->row mapping done by the caller).
      bench_annual: Series, scenario id -> summed benchmark monthly return
         over the same window.
    """
    try:
        A, bench_annual = scenario_matrix(
            returns_wide / 100.0, bench_wealth["Benchmark_Return"], tickers, window
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    n_scen = A.shape[1]

    print(f"  [CVaR] {n_scen} overlapping {window}-month scenarios built "
          f"from common history")
    return A.set_axis(range(n_scen), axis=1), bench_annual.set_axis(range(n_scen))


# --------------------------------------------------------------------
# Optimization
# --------------------------------------------------------------------

def solve(holdings, beta_wide, beta_bench, sigma, A_scen, bench_annual, tax_per_dollar,
          objective="te", enforce_cvar=True, enforce_tax_budget=True,
          exclude_from_tax_budget=frozenset(), max_sell_frac=None,
          omega=None):
    """
    ONE QP formulation for every path this script solves -- the joint problem
    (both constraints), Path A (tax budget only), Path B (CVaR floor only,
    minimizing tax cost), and the MSTR-carve-out Paths C/D are all this same
    function with different arguments. See "One QP formulation, not three"
    in the module docstring for why that's one function and not three.

    holdings: cleaned real-holdings DataFrame, one row per (ticker, account),
              restricted to tickers with usable betas.
    beta_wide: DataFrame index=ticker, cols=FF5 (betas)
    beta_bench: Series index=FF5
    sigma: DataFrame FF5 x FF5, annualized covariance
    A_scen: DataFrame index=ticker, cols=scenario -> summed monthly return
    bench_annual: Series scenario -> summed benchmark monthly return
    tax_per_dollar: Series index=ticker (Joint rows only) -> $tax / $ sold
    objective: "te" (default) minimizes tracking error, TE(x)^2. "tax"
        minimizes realized-gain tax cost instead (Path B: the cheapest way
        to hit the CVaR floor, ignoring the tax budget).
    enforce_cvar: include the Active-CVaR-floor constraint. The slack
        variables it needs (zeta, u) are always allocated regardless, so
        toggling this doesn't change the decision-vector layout.
    enforce_tax_budget: include the TAX_BUDGET cap. The sell-side slack it
        (and objective="tax") needs is always allocated and always
        constrained to equal the actual sale fraction -- only the BUDGET
        ceiling on it is optional.
    exclude_from_tax_budget: tickers whose realized-gain tax is tracked and
        reported as usual but does NOT count against TAX_BUDGET -- e.g. a
        concentrated position being handled through a separate taxable
        strategy (exchange fund, 10b5-1, etc.) that isn't modeled here, so
        the shared 2% budget is reserved for everything else.
    max_sell_frac: dict ticker -> max fraction of that ticker's CURRENT
        market value that may be sold this run (e.g. {"MSTR": 0.10} caps
        this year's MSTR sale to 10% of its current position, modeling a
        multi-year systematic diversification glide path rather than a
        one-shot exit). Enforced as a lower bound on the row's weight,
        x_i >= x0_i * (1 - max_sell_frac[ticker]).
    omega: DataFrame index=columns=ticker, the annualized FF5-residual
        covariance from residual_covariance(). When given (objective="te"
        only), the idio term w_ticker(x)' @ omega @ w_ticker(x) is added to
        TE(x)^2 -- see the module docstring's TE(x)^2 formula for why
        w_ticker is the FULL (not active) per-ticker weight. When omitted,
        behavior is unchanged from before this term existed.

    Every currently-held ticker is also floored at MIN_TICKER_WEIGHT of
    household value, summed across accounts -- see Constraint 3 in the
    module docstring. Unlike the two constraints above, this one has no
    on/off switch: it applies to every call.

    Returns a dict with "feasible" (bool, from the solver's own convergence
    flag), "x" (the proposed weight vector regardless of feasibility -- the
    caller decides whether a non-converged result is usable), "message"
    (the solver's status string), plus the intermediate quantities (rows,
    x0, betas, Sigma, CVaR scenario matrix, tax vector) needed to score that
    x afterward (te_of / cvar_of / tax_cost_of in __main__).
    """
    rows = holdings.reset_index(drop=True)
    n = len(rows)
    tickers_row = rows["Ticker"].values
    x0 = rows["Weight_Household"].values
    is_taxable = rows["Is_Taxable"].values

    joint_idx = np.where(is_taxable)[0]
    ira_idx = np.where(~is_taxable)[0]
    n_joint = len(joint_idx)
    n_scen = A_scen.shape[1]

    ff_cols = list(beta_wide.columns)
    beta_matrix = beta_wide.loc[tickers_row].values  # n x 5, one row per (ticker,account) row, ticker's own beta
    beta_bench_vec = beta_bench.reindex(ff_cols).fillna(0.0).values
    sigma_mat = sigma.loc[ff_cols, ff_cols].values

    A_row = A_scen.loc[tickers_row].values  # n x n_scen, each row = its ticker's scenario returns
    bench_annual_vec = bench_annual.values

    # Aggregator from (ticker,account) rows to per-ticker TOTAL weight, in
    # omega's ticker order -- w_ticker(x) = omega_G @ x. Built (and returned
    # in `shared`) regardless of whether `omega` is given, since te_of() in
    # __main__ needs the same G to score any x consistently with solve()'s
    # own objective.
    omega_tickers = list(omega.index) if omega is not None else list(pd.unique(tickers_row))
    omega_pos = {t: i for i, t in enumerate(omega_tickers)}
    omega_G = np.zeros((len(omega_tickers), n))
    for j, t in enumerate(tickers_row):
        if t in omega_pos:
            omega_G[omega_pos[t], j] = 1.0
    omega_mat = omega.loc[omega_tickers, omega_tickers].values if omega is not None else None

    tax_vec = np.array([tax_per_dollar.get(tickers_row[i], 0.0) for i in joint_idx])
    # Budget-counting version: zero out excluded tickers' rate so their sales
    # are unconstrained by (and don't consume) the shared tax budget, while
    # `tax_vec` above still reports their true tax cost for the trade table.
    exclude_from_tax_budget = exclude_from_tax_budget or frozenset()
    tax_vec_budget = np.array([
        0.0 if tickers_row[i] in exclude_from_tax_budget else tax_per_dollar.get(tickers_row[i], 0.0)
        for i in joint_idx
    ])

    # One row-index group per unique ticker, for the min-allocation floor --
    # a ticker held in both accounts is floored on its COMBINED weight.
    # NO_ADD_TICKERS are excluded here: their no-buy upper bound (below) can
    # sit under MIN_TICKER_WEIGHT, and the no-add mandate takes precedence
    # over the floor for those tickers specifically.
    # A ticker held ONLY in the IRA (e.g. XLB, SCHH -- no Joint row exists for
    # them at all) has no row anywhere that's still allowed to buy, now that
    # every IRA position is capped at its current weight (see the IRA-freeze
    # bounds below). Requiring such a ticker to reach MIN_TICKER_WEIGHT would
    # be infeasible by construction, so it's excluded from the floor exactly
    # like NO_ADD_TICKERS already is -- same reasoning, different cause.
    ira_only_tickers = {t for t in pd.unique(tickers_row) if np.all(~is_taxable[tickers_row == t])}
    ticker_groups = [
        np.where(tickers_row == t)[0]
        for t in pd.unique(tickers_row)
        if t not in NO_ADD_TICKERS and t not in ira_only_tickers
    ]

    # Decision vector layout is UNCONDITIONAL -- every call allocates the same
    # five blocks regardless of which constraints end up switched on, so one
    # set of indices/unpack() serves every path:
    #   [x (n), zeta (1), u (n_scen), sell (n_joint), turn (n)]
    # `turn` is the per-position turnover slack pinned to |x_i - x0_i| by the
    # two-sided constraints below.
    n_vars = n + 1 + n_scen + n_joint + n
    ix_x = slice(0, n)
    ix_zeta = n
    ix_u = slice(n + 1, n + 1 + n_scen)
    ix_sell = slice(n + 1 + n_scen, n + 1 + n_scen + n_joint)
    ix_turn = slice(n + 1 + n_scen + n_joint, n_vars)

    def unpack(v):
        return v[ix_x], v[ix_zeta], v[ix_u], v[ix_sell], v[ix_turn]

    turn_lambda = TURNOVER_LAMBDA if objective == "te" else TURNOVER_LAMBDA_TAX

    def objective_fun(v):
        x, _, _, sell, turn = unpack(v)
        turn_term = turn_lambda * float(np.sum(turn))
        if objective == "te":
            db = beta_matrix.T @ x - beta_bench_vec
            te_sq = float(db @ sigma_mat @ db)
            if omega_mat is not None:
                w_ticker = omega_G @ x
                te_sq += float(w_ticker @ omega_mat @ w_ticker)
            return te_sq + turn_term
        return float(sell @ tax_vec) + turn_term  # objective == "tax"

    def objective_grad(v):
        grad = np.zeros(n_vars)
        grad[ix_turn] = turn_lambda
        if objective == "te":
            x = v[ix_x]
            db = beta_matrix.T @ x - beta_bench_vec
            grad[ix_x] = 2.0 * beta_matrix @ (sigma_mat @ db)
            if omega_mat is not None:
                w_ticker = omega_G @ x
                grad[ix_x] += 2.0 * (omega_G.T @ (omega_mat @ w_ticker))
        else:
            grad[ix_sell] = tax_vec  # objective == "tax" is exactly linear in sell
        return grad

    constraints = []

    # sum(x) == 1 - tax reserve: proceeds spent paying this year's realized-gain
    # tax are NOT available for reinvestment, so the post-trade book is under
    # 100% invested by exactly that much -- the shortfall is cash set aside to
    # cover the tax bill, not weight the optimizer is free to place elsewhere.
    # Uses the TRUE tax vector (not tax_vec_budget), since the cash actually
    # owed doesn't depend on which tickers count against the shared budget --
    # an excluded ticker's sale still needs real cash reserved to pay its tax.
    #
    # Computed directly from x (max(x0 - x, 0) @ tax_vec), NOT from the `sell`
    # slack variable above: `sell` is only ever bounded BELOW by the actual
    # sold amount (sell >= x0 - x), so for a ticker excluded from the tax
    # BUDGET (tax_vec_budget == 0, e.g. MSTR in Paths C/D) inflating `sell`
    # costs the optimizer nothing -- it doesn't touch the objective and
    # doesn't tighten the budget inequality. Wiring THIS constraint to `sell`
    # let the solver satisfy it by manufacturing an arbitrarily large
    # fictitious "sale" for that ticker instead of an actual one, producing a
    # cash reserve with no relationship to the real realized tax cost. Tying
    # it to x directly closes that off: the reserve can only grow if x
    # actually moved.
    def invested_fun(v):
        x = v[ix_x]
        sold = np.maximum(x0[joint_idx] - x[joint_idx], 0.0)
        return float(np.sum(x) + sold @ tax_vec - 1.0)

    def invested_jac(v):
        x = v[ix_x]
        jac = np.zeros(n_vars)
        jac[ix_x] = 1.0
        still_selling = (x0[joint_idx] - x[joint_idx]) > 0.0
        jac[joint_idx] -= np.where(still_selling, tax_vec, 0.0)
        return jac

    constraints.append({"type": "eq", "fun": invested_fun, "jac": invested_jac})

    # No new money into the IRA is enforced below as a per-row UPPER BOUND
    # (x[i] <= x0[i] for every IRA row), not a separate constraint here --
    # see the `bounds` construction below, where `ira_idx` caps every IRA
    # position individually. An IRA can't just receive an arbitrary buy --
    # contributions are capped at a few thousand dollars/year by law, far
    # below what a reweighting-scale trade would need -- and unlike a taxable
    # account, even reallocating WITHIN the IRA (sell one position to buy
    # another) is disallowed here: every IRA position may only be trimmed or
    # held, exactly like NO_ADD_TICKERS, just applied by account rather than
    # by ticker.

    # CVaR slack: u_s >= Loss_s(x) - zeta ,  Loss_s(x) = bench_annual[s] - x @ A_row[:,s]
    # Always defined (so zeta/u exist for every path) -- enforce_cvar below
    # only toggles whether the FLOOR on the resulting CVaR is applied.
    def cvar_slack_fun(v):
        x, zeta, u, _, _ = unpack(v)
        loss = bench_annual_vec - A_row.T @ x  # (n_scen,)
        return u - (loss - zeta)  # >= 0

    def cvar_slack_jac(v):
        jac = np.zeros((n_scen, n_vars))
        jac[:, ix_x] = A_row.T  # d/dx [u - loss + zeta] = +A_row.T  (since -loss = -( -A_row.T x) = A_row.T x)
        jac[:, ix_zeta] = 1.0
        jac[np.arange(n_scen), np.arange(n + 1, n + 1 + n_scen)] = 1.0
        return jac

    constraints.append({"type": "ineq", "fun": cvar_slack_fun, "jac": cvar_slack_jac})

    cvar_coef = 1.0 / (CVAR_ALPHA_TAIL * n_scen)
    if enforce_cvar:
        # CVaR budget: zeta + (1/(alpha*S)) * sum(u) <= CVAR_FLOOR_AS_LOSS  ->  written as >=0 form
        def cvar_budget_fun(v):
            _, zeta, u, _, _ = unpack(v)
            cvar = zeta + cvar_coef * np.sum(u)
            return (-CVAR_FLOOR) - cvar  # >= 0  <=>  cvar <= -CVAR_FLOOR (0.15)

        def cvar_budget_jac(v):
            jac = np.zeros(n_vars)
            jac[ix_zeta] = -1.0
            jac[ix_u] = -cvar_coef
            return jac

        constraints.append({"type": "ineq", "fun": cvar_budget_fun, "jac": cvar_budget_jac})

    # Tax slack: sell_j >= x0[joint_idx] - x[joint_idx]. Always defined (sell
    # must be known for reporting and for objective="tax") -- enforce_tax_budget
    # below only toggles whether a BUDGET cap on it is applied.
    def tax_slack_fun(v):
        x, _, _, sell, _ = unpack(v)
        return sell - (x0[joint_idx] - x[joint_idx])  # >= 0

    def tax_slack_jac(v):
        jac = np.zeros((n_joint, n_vars))
        jac[np.arange(n_joint), joint_idx] = 1.0  # d/dx_i [sell - x0 + x_i] = +1
        jac[np.arange(n_joint), np.arange(ix_sell.start, ix_sell.stop)] = 1.0
        return jac

    constraints.append({"type": "ineq", "fun": tax_slack_fun, "jac": tax_slack_jac})

    if enforce_tax_budget:
        # Tax budget: TAX_BUDGET - sum(sell * tax_vec_budget) >= 0
        # (excluded tickers' rate is zeroed in tax_vec_budget, so their sales are
        # free from this constraint's point of view; tax_vec, unchanged, is what
        # gets reported per-position in the trade table.)
        def tax_budget_fun(v):
            _, _, _, sell, _ = unpack(v)
            return TAX_BUDGET - float(sell @ tax_vec_budget)

        def tax_budget_jac(v):
            jac = np.zeros(n_vars)
            jac[ix_sell] = -tax_vec_budget
            return jac

        constraints.append({"type": "ineq", "fun": tax_budget_fun, "jac": tax_budget_jac})

    # Turnover slack, two-sided, one pair per position:
    #     turn_i >= x_i - x0_i   and   turn_i >= x0_i - x_i
    # Together these say turn_i >= |x_i - x0_i|. Since turn only ever ENTERS
    # the objective with a positive coefficient, the optimizer drives each
    # turn_i down to its tight value, so turn_i == |x_i - x0_i| at the optimum
    # -- the same slack-variable device already used for the CVaR and tax
    # terms, and what keeps an L1 penalty inside a smooth convex QP. Always on.
    def turnover_up_fun(v):
        x, _, _, _, turn = unpack(v)
        return turn - (x - x0)  # >= 0

    def turnover_up_jac(v):
        jac = np.zeros((n, n_vars))
        jac[np.arange(n), np.arange(n)] = -1.0
        jac[np.arange(n), np.arange(ix_turn.start, ix_turn.stop)] = 1.0
        return jac

    def turnover_dn_fun(v):
        x, _, _, _, turn = unpack(v)
        return turn + (x - x0)  # >= 0

    def turnover_dn_jac(v):
        jac = np.zeros((n, n_vars))
        jac[np.arange(n), np.arange(n)] = 1.0
        jac[np.arange(n), np.arange(ix_turn.start, ix_turn.stop)] = 1.0
        return jac

    constraints.append({"type": "ineq", "fun": turnover_up_fun, "jac": turnover_up_jac})
    constraints.append({"type": "ineq", "fun": turnover_dn_fun, "jac": turnover_dn_jac})

    # Minimum per-ticker allocation floor (Constraint 3): sum of a ticker's
    # rows (across accounts) >= MIN_TICKER_WEIGHT. Always on, no toggle.
    def min_ticker_fun(v):
        x = v[ix_x]
        return np.array([x[idx].sum() - MIN_TICKER_WEIGHT for idx in ticker_groups])

    def min_ticker_jac(v):
        jac = np.zeros((len(ticker_groups), n_vars))
        for k, idx in enumerate(ticker_groups):
            jac[k, idx] = 1.0
        return jac

    constraints.append({"type": "ineq", "fun": min_ticker_fun, "jac": min_ticker_jac})

    max_sell_frac = max_sell_frac or {}
    x_bounds = []
    for i in range(n):
        cap = max_sell_frac.get(tickers_row[i])
        lower = x0[i] * (1.0 - cap) if cap is not None else 0.0
        # No new money into the IRA: every IRA row is capped at its current
        # weight, same as NO_ADD_TICKERS -- an IRA row may be trimmed or
        # held, never bought, not even by reallocating from another IRA
        # position (see the module docstring's "No new money into the IRA").
        upper = x0[i] if (tickers_row[i] in NO_ADD_TICKERS or not is_taxable[i]) else 1.0
        x_bounds.append((min(lower, upper), upper))
    bounds = (x_bounds + [(None, None)] + [(0.0, None)] * n_scen
              + [(0.0, 1.0)] * n_joint + [(0.0, 1.0)] * n)

    v0 = np.zeros(n_vars)
    v0[ix_x] = x0
    loss0 = bench_annual_vec - A_row.T @ x0
    v0[ix_zeta] = np.percentile(loss0, 100 * (1 - CVAR_ALPHA_TAIL))
    v0[ix_u] = np.maximum(loss0 - v0[ix_zeta], 0.0)
    v0[ix_sell] = 0.0
    v0[ix_turn] = 0.0  # x starts at x0, so zero turnover is feasible here

    result = minimize(
        objective_fun, v0, jac=objective_grad, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12},
    )

    shared = dict(rows=rows, x0=x0, joint_idx=joint_idx, beta_matrix=beta_matrix,
                  beta_bench_vec=beta_bench_vec, sigma_mat=sigma_mat, A_row=A_row,
                  bench_annual_vec=bench_annual_vec, tax_vec=tax_vec, n=n, n_scen=n_scen,
                  omega_G=omega_G, omega_mat=omega_mat, omega_tickers=omega_tickers)
    x_opt, _zeta_opt, _u_opt, _sell_opt, _turn_opt = unpack(result.x)
    return {"feasible": bool(result.success), "x": x_opt, "message": result.message, **shared}


def cvar_of(x, A_row, bench_annual_vec, alpha_tail=CVAR_ALPHA_TAIL):
    result = active_cvar(
        x,
        pd.DataFrame(np.asarray(A_row)),
        bench_annual_vec,
        alpha=alpha_tail,
    )
    return result.value


def build_trade_table(rows, x0, x_new, tax_vec_full, total_household_value):
    trades = rows[["Account", "Ticker", "Name", "Is_Taxable", "Market_Value"]].copy()
    trades = trades.rename(columns={"Name": "Description"})
    trades["Current_Weight"] = x0
    trades["Proposed_Weight"] = x_new
    trades["Weight_Change"] = x_new - x0
    trades["Current_Dollars"] = trades["Current_Weight"] * total_household_value
    trades["Proposed_Dollars"] = trades["Proposed_Weight"] * total_household_value
    trades["Trade_Dollars"] = trades["Proposed_Dollars"] - trades["Current_Dollars"]
    trades["Direction"] = np.where(trades["Trade_Dollars"] > 1e-6, "BUY",
                            np.where(trades["Trade_Dollars"] < -1e-6, "SELL", "HOLD"))
    trades["Realized_Gain_Dollars"] = 0.0
    trades["Tax_Cost_Dollars"] = 0.0
    sell_mask = (trades["Direction"] == "SELL") & trades["Is_Taxable"]
    if sell_mask.any():
        sold_frac = (-trades.loc[sell_mask, "Weight_Change"]) / trades.loc[sell_mask, "Current_Weight"]
        gain_dollars = rows.loc[sell_mask.values, "Total_Gain"].values
        trades.loc[sell_mask, "Realized_Gain_Dollars"] = sold_frac.values * gain_dollars
        trades.loc[sell_mask, "Tax_Cost_Dollars"] = (
            -trades.loc[sell_mask, "Trade_Dollars"] * tax_vec_full[sell_mask.values]
        )

    # Cash reserved to pay this year's realized-gain tax: sum(x_new) is now
    # 1 - true_tax_cost(x) (see the "sum(x) == 1 - tax reserve" constraint),
    # so the trade table's weights alone would silently fall short of 100%.
    # Add it back as an explicit row so Proposed_Weight/_Dollars reconcile to
    # the full household value -- this is un-invested cash set aside for the
    # IRS, not money the optimizer lost track of.
    cash_reserved_dollars = total_household_value - trades["Proposed_Dollars"].sum()
    if abs(cash_reserved_dollars) > 1e-6:
        cash_row = pd.DataFrame([{
            "Account": "Cash", "Ticker": "CASH_RESERVE", "Description": "Reserved for realized-gain tax",
            "Is_Taxable": False, "Market_Value": 0.0,
            "Current_Weight": 0.0, "Proposed_Weight": cash_reserved_dollars / total_household_value,
            "Weight_Change": cash_reserved_dollars / total_household_value,
            "Current_Dollars": 0.0, "Proposed_Dollars": cash_reserved_dollars,
            "Trade_Dollars": cash_reserved_dollars, "Direction": "RESERVE",
            "Realized_Gain_Dollars": 0.0, "Tax_Cost_Dollars": 0.0,
        }])
        trades = pd.concat([trades, cash_row], ignore_index=True)
    return trades.sort_values("Trade_Dollars")


def print_and_return_trades(trades, label):
    print(f"\n=== {label} (sorted by dollar trade size) ===")
    print(trades[["Account", "Ticker", "Direction", "Current_Weight", "Proposed_Weight",
                   "Trade_Dollars", "Tax_Cost_Dollars"]]
          .to_string(index=False, formatters={
              "Current_Weight": "{:.3%}".format, "Proposed_Weight": "{:.3%}".format,
              "Trade_Dollars": "${:,.0f}".format, "Tax_Cost_Dollars": "${:,.0f}".format,
          }))
    return trades


if __name__ == "__main__":
    # Clear every trade-proposal CSV from a prior run before writing new ones.
    # Only Path C and Path D are solved now (see below) -- trade_proposals.csv
    # / _pathA.csv / _pathB.csv are legacy names from when this script also
    # solved a CVaR-constrained joint problem and its relaxations; clearing
    # them too means a stale file from an older run of this script is never
    # silently mistaken for current output.
    for stale in ["trade_proposals.csv", "trade_proposals_pathA.csv", "trade_proposals_pathB.csv",
                  "trade_proposals_pathC.csv", "trade_proposals_pathD.csv"]:
        stale_path = os.path.join(DATA_DIR, stale)
        if os.path.exists(stale_path):
            os.remove(stale_path)

    holdings_all = pd.read_csv(os.path.join(DATA_DIR, "holdings_household_clean.csv"))
    returns_wide = pd.read_csv(os.path.join(DATA_DIR, "holdings_household_monthly_returns.csv"),
                                parse_dates=["Date"]).set_index("Date")
    ff, rf = load_ff()
    bench_weights = pd.read_csv(os.path.join(DATA_DIR, "benchmark_mvo_weights.csv"), index_col=0)["Weight"]
    bench_wealth = pd.read_csv(os.path.join(DATA_DIR, "benchmark_wealth_curve.csv"),
                                parse_dates=["Date"]).set_index("Date")

    tickers = sorted(holdings_all["Ticker"].unique())
    print(f"Regressing {len(tickers)} tickers on FF5 (HAC SEs)...")
    beta_wide, _idio_var = regress_tickers(returns_wide, ff, rf, tickers)  # idio_var unused here -- see its docstring

    usable_tickers = list(beta_wide.index)
    dropped = sorted(set(tickers) - set(usable_tickers))
    if dropped:
        print(f"\nDropped for insufficient/missing price history: {dropped}")
    holdings = holdings_all[holdings_all["Ticker"].isin(usable_tickers)].copy()
    holdings["Weight_Household"] = holdings["Market_Value"] / holdings_all["Market_Value"].sum()

    beta_bench = pd.Series(0.0, index=FF_FACTOR_COLS)
    beta_bench.update(bench_weights)
    sigma = blended_factor_covariance(ff, annualize=12)

    print("\nBuilding CVaR scenarios from common price history...")
    A_scen, bench_annual = build_cvar_scenarios(returns_wide, bench_wealth, usable_tickers)

    print("\nEstimating FF5 residual covariance for the idio TE term...")
    omega = residual_covariance(returns_wide, ff, rf, usable_tickers, shrink=IDIO_SHRINKAGE)

    # tax $ per $ sold, keyed by ticker, Joint account only
    joint = holdings[holdings["Is_Taxable"]]
    tax_per_dollar = (
        (joint["LT_Gain"] * TAX_LT + joint["ST_Gain"] * TAX_ST) / joint["Market_Value"]
    )
    tax_per_dollar.index = joint["Ticker"].values

    # A throwaway solve (CVaR and tax budget both off) purely to get the
    # decision-vector layout / rows / x0 / beta and sigma matrices that Path C
    # and Path D below both need -- no constraint choice here matters since
    # only `outcome["rows"]`, `outcome["x0"]`, etc. (identical across every
    # call to solve() for this holdings set) are read from it.
    outcome = solve(holdings, beta_wide, beta_bench, sigma, A_scen, bench_annual, tax_per_dollar,
                     enforce_cvar=False, enforce_tax_budget=False, omega=omega)

    rows = outcome["rows"]
    x0 = outcome["x0"]
    beta_matrix, beta_bench_vec, sigma_mat = outcome["beta_matrix"], outcome["beta_bench_vec"], outcome["sigma_mat"]
    A_row, bench_annual_vec = outcome["A_row"], outcome["bench_annual_vec"]
    omega_G, omega_mat = outcome["omega_G"], outcome["omega_mat"]
    total_household_value = holdings_all["Market_Value"].sum()
    joint_mask = rows["Is_Taxable"].values
    tax_vec_full = np.array([tax_per_dollar.get(t, 0.0) for t in rows["Ticker"].values])

    def te_of(x):
        """TE(x), factor term plus idio term -- must match solve()'s own
        objective (see the module docstring's TE(x)^2 formula) so
        before/after TE comparisons are apples to apples."""
        db = beta_matrix.T @ x - beta_bench_vec
        te_sq = float(db @ sigma_mat @ db)
        w_ticker = omega_G @ x
        te_sq += float(w_ticker @ omega_mat @ w_ticker)
        return float(np.sqrt(te_sq))

    def te_factor_of(x):
        """Factor-only TE (the pre-idio-term definition) -- reported
        alongside te_of() so the size of the idio addition is visible."""
        db = beta_matrix.T @ x - beta_bench_vec
        return float(np.sqrt(db @ sigma_mat @ db))

    def tax_cost_of(x):
        sell = np.maximum(x0 - x, 0.0) * joint_mask
        return float((sell * tax_vec_full).sum()) * total_household_value

    def turnover_of(x):
        """One-way turnover: 0.5 * sum|x - x0|. Sells now slightly exceed buys
        (sum x0 - sum x == the tax reserve carved out of proceeds -- see the
        "sum(x) == 1 - tax reserve" constraint), so this is not quite half of
        a perfectly netting book, but it's still the standard one-way-dollar-
        moved reading of the L1 turnover penalty."""
        return 0.5 * float(np.abs(x - x0).sum())

    te_before = te_of(x0)
    te_factor_before = te_factor_of(x0)
    cvar_before = cvar_of(x0, A_row, bench_annual_vec)
    path_metrics = {"before": {"feasible": True, "te": te_before, "te_factor_only": te_factor_before,
                                "cvar": cvar_before, "tax_cost": 0.0, "turnover": 0.0}}
    summary_lines = [
        "Tax-Aware Minimum-Tracking-Error Optimization (Paths C / D)",
        "=================================================================",
        "",
        f"Household market value: ${total_household_value:,.2f}",
        f"Tickers used: {usable_tickers}",
        f"Tickers dropped (insufficient history): {dropped or 'none'}",
        "",
        "Objective: minimize ex-ante tracking error vs. the policy benchmark "
        "(construct_benchmark_mvo.py) -- factor term uses the annualized 50/50 blended FF5 covariance "
        "(raw full-sample covariance + 12-month-half-life EWMA) implemented by blended_factor_covariance(), "
        "PLUS an idio term (FF5-residual covariance across the FULL ticker weight, shrunk "
        f"{IDIO_SHRINKAGE:.0%} toward its diagonal -- see residual_covariance()), plus a per-position "
        f"L1 turnover penalty lambda * sum_i |x_i - x0_i| (lambda = {TURNOVER_LAMBDA}). "
        "No CVaR constraint -- Active CVaR is still computed and reported below for context, but it does "
        "not gate either path's solution.",
        f"Every held ticker floored at {MIN_TICKER_WEIGHT:.2%} of household value (Constraint 3).",
        f"Before any trades: TE = {te_before:.4%} (factor-only: {te_factor_before:.4%}), "
        f"Active CVaR(5%) = {cvar_before:.4%} (informational only).",
        "",
    ]

    # Path C and Path D: minimize TE + turnover subject ONLY to the tax
    # budget and the min-ticker floor -- no CVaR constraint (removed from
    # both paths' problem entirely; CVaR is still computed and reported for
    # context, it just no longer gates the solution). MSTR's realized-gain
    # tax is carved out of the 2% budget entirely -- e.g. because a separate
    # taxable strategy for that specific position (exchange fund, 10b5-1,
    # etc.) is being considered elsewhere and isn't modeled here. MSTR's true
    # tax cost is still reported per-position; it just doesn't count against
    # the shared budget. On top of that, each path caps how much of the
    # CURRENT MSTR position may be sold this run -- modeling a multi-year
    # glide path instead of a one-shot exit -- at 70% (Path C) and 30% (Path D).
    MSTR_CAPPED_PATHS = [("C", 0.70), ("D", 0.30)]
    for label, sell_cap in MSTR_CAPPED_PATHS:
        print(f"\nSolving Path {label} -- min TE s.t. tax budget only (no CVaR constraint), "
              f"MSTR excluded from the tax budget, MSTR sale capped at {sell_cap:.0%} of its "
              f"current position...")
        outcome_x = solve(holdings, beta_wide, beta_bench, sigma, A_scen, bench_annual, tax_per_dollar,
                           enforce_cvar=False, enforce_tax_budget=True,
                           exclude_from_tax_budget={"MSTR"}, max_sell_frac={"MSTR": sell_cap}, omega=omega)
        if outcome_x["feasible"]:
            x_p = outcome_x["x"]
            te_p, te_factor_p = te_of(x_p), te_factor_of(x_p)
            cvar_p = cvar_of(x_p, A_row, bench_annual_vec)
            tax_p_total = tax_cost_of(x_p)
            trades_p = build_trade_table(rows, x0, x_p, tax_vec_full, total_household_value)
            snow_tax_p = float(trades_p.loc[trades_p["Ticker"] == "MSTR", "Tax_Cost_Dollars"].sum())
            tax_p_ex_snow = tax_p_total - snow_tax_p
            trades_p.to_csv(os.path.join(DATA_DIR, f"trade_proposals_path{label}.csv"), index=False)
            path_metrics[label] = {"feasible": True, "te": te_p, "te_factor_only": te_factor_p, "cvar": cvar_p,
                                    "tax_cost": tax_p_ex_snow, "tax_cost_snow": snow_tax_p,
                                    "tax_cost_total": tax_p_total, "sell_cap": sell_cap, "turnover": turnover_of(x_p)}
            print_and_return_trades(trades_p, f"Path {label} -- tax budget only (no CVaR constraint), "
                                     f"MSTR excluded from budget, sale capped at {sell_cap:.0%}")
            print(f"  TE: {te_before:.4%} -> {te_p:.4%} (factor-only: {te_factor_before:.4%} -> {te_factor_p:.4%})"
                  f"   CVaR (informational): {cvar_before:.4%} -> {cvar_p:.4%}")
            print(f"  Tax cost ex-MSTR (counts against budget): ${tax_p_ex_snow:,.0f} "
                  f"({tax_p_ex_snow/total_household_value:.4%} of {TAX_BUDGET:.2%} budget)")
            print(f"  MSTR's own tax cost (excluded from budget, handled separately): ${snow_tax_p:,.0f}")
            print(f"  Total realized tax cost (both): ${tax_p_total:,.0f}")
            summary_lines += [
                "",
                f"Path {label} -- min TE s.t. tax budget only (no CVaR constraint), MSTR's tax excluded "
                f"from the 2% budget (separate taxable strategy assumed), MSTR sale capped at "
                f"{sell_cap:.0%} of its current position:",
                f"  TE: {te_before:.4%} -> {te_p:.4%} (factor-only: {te_factor_before:.4%} -> {te_factor_p:.4%})"
                f"   CVaR (informational): {cvar_before:.4%} -> {cvar_p:.4%}",
                f"  Tax cost ex-MSTR (counts against budget): ${tax_p_ex_snow:,.0f} ({tax_p_ex_snow/total_household_value:.4%})",
                f"  MSTR's own tax cost (excluded, separate strategy): ${snow_tax_p:,.0f}",
                f"  Total realized tax cost (both): ${tax_p_total:,.0f} ({tax_p_total/total_household_value:.4%})",
                trades_p[["Account", "Ticker", "Direction", "Current_Weight", "Proposed_Weight",
                          "Trade_Dollars", "Realized_Gain_Dollars", "Tax_Cost_Dollars"]].round(4).to_string(index=False),
            ]
        else:
            print(f"  Path {label} infeasible: even spending the full 2% ex-MSTR budget elsewhere and "
                  f"selling up to {sell_cap:.0%} of MSTR, no feasible reweighting exists ({outcome_x['message']}).")
            path_metrics[label] = {"feasible": False, "sell_cap": sell_cap}
            summary_lines += [
                "",
                f"Path {label} -- tax budget only, MSTR excluded from budget, sale capped at "
                f"{sell_cap:.0%}: INFEASIBLE ({outcome_x['message']}).",
            ]

    with open(os.path.join(DATA_DIR, "optimizer_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))
    with open(os.path.join(DATA_DIR, "path_metrics.json"), "w") as f:
        json.dump({
            "total_household_value": float(total_household_value),
            "cvar_floor": CVAR_FLOOR, "tax_budget": TAX_BUDGET,
            "min_ticker_weight": MIN_TICKER_WEIGHT,
            "tax_lt_rate": TAX_LT, "tax_st_rate": TAX_ST,
            "turnover_lambda": TURNOVER_LAMBDA, "turnover_lambda_tax": TURNOVER_LAMBDA_TAX,
            "idio_shrinkage": IDIO_SHRINKAGE,
            "paths": path_metrics,
        }, f, indent=2)

    print(f"\nSaved outputs to {DATA_DIR}:")
    print("  trade_proposals_pathC.csv  (no CVaR constraint, MSTR excluded from budget, sale capped at 70%)")
    print("  trade_proposals_pathD.csv  (no CVaR constraint, MSTR excluded from budget, sale capped at 30%)")
    print("  path_metrics.json  (TE/CVaR/tax cost summary for every path, for the dashboard generator)")
    print("  optimizer_summary.txt")
