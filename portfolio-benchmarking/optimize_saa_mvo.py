"""Minimize covariance-based tracking error to the workbook SAA policy target.

This is a second, independent optimizer for the same household holdings.  It
uses the sleeve covariance matrix and policy targets in
``all_equity_saa_covariance.xlsx`` rather than the FF5 policy benchmark used
by ``optimize_tax_cvar_mvo.py``.  Positions remain account-level so taxable
sales use the same account-level trade-table convention as the primary
optimizer, and this path runs its OWN 2% tax budget (numerically the same as
the primary optimizer's, but tracked independently -- see ``SAA_TAX_BUDGET``)
plus its own MSTR-specific carve-out: MSTR's realized-gain tax is excluded
from that budget entirely (assumed handled by a separate taxable strategy
elsewhere), and this year's MSTR sale is capped at 50% of its current
position -- see ``SAA_MSTR_SELL_CAP`` below, the same one-year-glide-path
device as Paths C/D in the primary optimizer (which cap MSTR at 70% / 30%).
SHV may not be added to, here or in any path -- see ``NO_ADD_TICKERS`` in
``optimize_tax_cvar_mvo.py``, imported and applied identically here.

Cash reserved to actually pay the tax bill: sum(x) == 1 - true_tax_cost(x)
---------------------------------------------------------------------------
Each path's own tax BUDGET (``SAA_PATHS[label]["tax_budget"]``) caps what
counts against that path's ceiling -- for "SAA" this excludes MSTR's own
tax entirely (see ``snow_carveout``). It says nothing about where the cash
to actually pay the tax comes from. Every path's book is instead subject to
sum(x) + true_tax_cost_fraction(x) == 1, using the FULL (never carved-out)
per-dollar tax rate -- so realized-gain tax on ANY sale, including MSTR's
when it's excluded from the budget, is money that leaves the sale proceeds
and isn't available for reinvestment. The post-trade book sits below 100%
invested by exactly that shortfall, which shows up in the trade table as an
explicit "CASH_RESERVE" row (``build_trade_table()``, imported from
``optimize_tax_cvar_mvo.py``), not an unexplained gap.

No new money into the IRA
--------------------------
Every IRA row is capped at its current weight (``bounds``, in ``main()``) --
a plain per-row upper bound, same device as ``NO_ADD_TICKERS``, just applied
by account rather than by ticker. Annual contribution limits are far below
what a reweighting-scale trade would need, and unlike a taxable account,
even reallocating WITHIN the IRA (sell one position to fund a buy of
another) is disallowed here: every IRA position may only be trimmed or
held. A ticker held ONLY in the IRA (e.g. XLB, SCHH -- no Joint row exists
for them at all) is therefore excluded from the MIN_TICKER_WEIGHT floor
exactly like ``NO_ADD_TICKERS`` is (``ira_only_tickers`` in ``main()``) --
there is no row left anywhere that could still buy it up to the floor. Same
rule as ``optimize_tax_cvar_mvo.py``'s Paths C/D.

Idiosyncratic risk: a different gap than the FF5 optimizer's, and mostly NOT
applicable here
---------------------------------------------------------------------------
The workbook's Covariance sheet is a full SLEEVE-level covariance matrix --
each sleeve's own diagonal entry is already that sleeve's TOTAL variance
(e.g. "Single Stock" is ~0.0784 annualized, ~28% vol -- MSTR's real
volatility, not some small residual). Unlike ``optimize_tax_cvar_mvo.py``'s
factor model, there is no lower-dimensional projection here for anything to
be "left over" from, so there is no classic idio term to add for a sleeve
that maps 1:1 to a single ticker (MSTR is exactly this case, and is why
this optimizer doesn't need the FF5 optimizer's idio-covariance treatment).

The real gap here is different: ``mapping @ x`` (see ``main()`` below)
COLLAPSES every ticker assigned to the same sleeve into one aggregate
weight before the covariance is ever applied, so the model cannot see
*which* ticker within a sleeve is held -- it assumes perfect
interchangeability. Two sleeves hold more than one ticker: "US Large
Growth" (ITOT, IWF) and "US Large Value" (IWD, XLB, SCHH). Swapping between
two members of the same sleeve registers as EXACTLY ZERO change in
tracking error under the sleeve-level model, which is only a reasonable
approximation when the members actually share the sleeve's risk driver.

``compute_proxy_tracking_variance()`` estimates this directly from realized
data already in this repo (``holdings_household_monthly_returns.csv``, the same
file the FF5 optimizer uses): each multi-ticker sleeve's "proxy return" is
built as the market-value-weighted average of its OTHER members' realized
monthly returns, and each ticker's idio variance is the residual variance
from a single-factor OLS regression of its own return on that proxy -- same
construction as ``regress_tickers()``'s FF5 idio_var, just with a
one-ticker "sleeve" factor standing in for the 5 FF factors, since there is
no broader factor model at the sleeve level to regress against. It is
added to the objective as a per-ticker diagonal penalty, `sum_ticker
(ticker weight)^2 * tau^2`, the same mechanical form the FF5 optimizer's
idio term takes, using FULL (not active) weight for the same reason: the
policy target holds no individual tickers, only sleeve weights, so it
contributes zero idio risk of its own.

SHV: a near-zero-beta ticker needed its OWN sleeve, not a bigger idio term
---------------------------------------------------------------------------
SHV was originally proxied into "US Large Value" alongside IWD/XLB/SCHH.
Regressing it against that sleeve's other members finds beta ~ -0.01 --
essentially zero: SHV (a short-Treasury/cash-like fund) simply doesn't
move with large-value equities at all, so its "residual" against that
proxy IS its own return, not some large tracking error. That's a beta
misspecification in the SLEEVE MAPPING itself (``mapping @ x`` was
asserting a ~beta-1 equity exposure this ticker doesn't have), which no
per-ticker idio penalty can fix -- the fix has to be in the mapping, not
the risk term. ``CASH_SLEEVE`` (see ``estimate_cash_sleeve_row()``) gives
SHV its own single-member sleeve instead, with its variance and
covariance against every real sleeve estimated empirically from its own
return history (the workbook has no cash/short-Treasury sleeve to source a
number from). Target weight for ``CASH_SLEEVE`` is 0% -- this is an
ALL-EQUITY SAA policy, so cash isn't part of the target mix -- but
NO_ADD_TICKERS already prevents SHV from ever being bought back up, so a
0% target only ever pushes toward trimming, at whatever pace the tax
budget and turnover penalty allow.
"""

import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

from optimize_tax_cvar_mvo import (
    MIN_OBS,
    MIN_TICKER_WEIGHT,
    NO_ADD_TICKERS,
    TAX_LT,
    TAX_ST,
    build_cvar_scenarios,
    build_trade_table,
    cvar_of,
)


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "Data")
WORKBOOK = os.path.join(HERE, "..", "all_equity_saa_covariance.xlsx")
METRICS = os.path.join(DATA_DIR, "path_metrics.json")

# This optimizer runs its own tax budget(s), tracked independently of the
# primary FF5 optimizer's (both happen to use a 2%-of-household-value/year
# budget for the base path, and, like Paths C/D there, carve MSTR's own
# realized-gain tax out of that budget entirely (a separate taxable strategy
# is assumed to handle it; MSTR's true tax cost is still computed and
# reported, it just doesn't consume the shared budget) while capping how
# much of the current MSTR position this run may sell, modeling the same
# multi-year glide path rather than a one-shot exit.
#
# Two paths are solved below. "SAA" is the original: a 2% tax budget with
# MSTR's own realized-gain tax carved out entirely (assumed handled by a
# separate taxable strategy, so it doesn't consume the shared budget).
# "SAA_3pct" is the new analog path requested: same objective, same MSTR
# sell-cap glide path, but the tax budget is raised to 3% AND MSTR's tax
# cost counts against that budget like any other position (no carve-out) --
# so a bigger MSTR sale must be paid for out of the same 3% budget as
# everything else, not exempted from it.
SAA_PATHS = {
    "SAA": {"tax_budget": 0.02, "snow_carveout": True,
            "trade_csv": "trade_proposals_pathSAA.csv",
            "summary_txt": "saa_optimizer_summary.txt"},
    "SAA_3pct": {"tax_budget": 0.03, "snow_carveout": False,
                 "trade_csv": "trade_proposals_pathSAA_3pct.csv",
                 "summary_txt": "saa_optimizer_summary_3pct.txt"},
}
SAA_TURNOVER_LAMBDA = 0.002
SAA_MSTR_TICKER = "MSTR"
SAA_MSTR_SELL_CAP = 0.50

# A synthetic sleeve, not in the workbook -- see the module docstring's
# "SHV: a near-zero-beta ticker needed its OWN sleeve" section. Estimated
# empirically by estimate_cash_sleeve_row() rather than sourced from the
# workbook, since the workbook has no cash/short-Treasury sleeve.
CASH_SLEEVE = "Near-Cash (SHV)"

# The workbook has no dedicated sleeves for these existing holdings.  These
# are transparent investable proxies, keeping every held position in the
# household optimization instead of silently dropping it. SHV maps to
# CASH_SLEEVE, not a workbook sleeve -- see the module docstring.
SLEEVE_BY_TICKER = {
    "MSTR": "Single Stock",
    "ITOT": "US Large Growth",
    "IWF": "US Large Growth",
    "IWD": "US Large Value",
    "SCHD": "US Dividend",
    "QUAL": "US Quality Factor",
    "MTUM": "US Momentum Factor",
    "IJJ": "US Mid Value",
    "IJK": "US Mid Growth",
    "IJS": "US Small Value",
    "IJT": "US Small Growth",
    "IEFA": "Intl Developed",
    "IEMG": "Emerging Markets",
    "XLB": "US Large Value",
    "SCHH": "US Large Value",
    "SHV": CASH_SLEEVE,
}

RETURNS_FILE = os.path.join(DATA_DIR, "holdings_household_monthly_returns.csv")


def estimate_cash_sleeve_row(returns_wide, sleeve_by_ticker, ticker_weights, sleeves, cash_ticker,
                              min_obs=MIN_OBS):
    """Empirically estimate CASH_SLEEVE's own annualized variance and its
    covariance with every EXISTING workbook sleeve, from realized returns --
    the workbook has no such sleeve to source a number from, since it's
    synthetic (see the module docstring). Each existing sleeve's proxy
    return, for this covariance estimate only, is the market-value-weighted
    average return of its own CURRENTLY HELD member tickers (same
    convention as compute_proxy_tracking_variance, without the leave-one-out
    exclusion since `cash_ticker` is never one of these sleeves' own
    members).

    Returns (variance: float, covariances: Series index=sleeves), both
    ANNUALIZED -- 0.0 covariance for a sleeve with no held member, or too
    little common history with `cash_ticker` to estimate.
    """
    if cash_ticker not in returns_wide.columns:
        raise RuntimeError(f"No return history for {cash_ticker} -- cannot estimate {CASH_SLEEVE}'s risk.")
    cash_ret = returns_wide[cash_ticker].dropna() / 100.0
    if len(cash_ret) < min_obs:
        raise RuntimeError(f"Only {len(cash_ret)} months of {cash_ticker} history (< {min_obs}) -- "
                            f"cannot estimate {CASH_SLEEVE}'s own variance.")
    variance = float(cash_ret.var(ddof=1) * 12)

    sleeve_members = {}
    for ticker, sleeve in sleeve_by_ticker.items():
        if ticker != cash_ticker:
            sleeve_members.setdefault(sleeve, []).append(ticker)

    cov = {}
    for sleeve in sleeves:
        members = [t for t in sleeve_members.get(sleeve, []) if t in returns_wide.columns]
        if not members:
            cov[sleeve] = 0.0
            continue
        ret = returns_wide[members + [cash_ticker]].dropna(how="any") / 100.0
        if len(ret) < min_obs:
            print(f"  [saa cash] {sleeve}: only {len(ret)} common months with {cash_ticker} "
                  f"(< {min_obs}), treating its covariance with {CASH_SLEEVE} as 0")
            cov[sleeve] = 0.0
            continue
        w = ticker_weights.reindex(members).fillna(0.0)
        w = w / w.sum() if w.sum() > 0 else pd.Series(1.0 / len(members), index=members)
        sleeve_proxy = (ret[members] * w).sum(axis=1)
        cov[sleeve] = float(np.cov(ret[cash_ticker], sleeve_proxy, ddof=1)[0, 1] * 12)
    return variance, pd.Series(cov)


def compute_proxy_tracking_variance(sleeve_by_ticker, ticker_weights, returns_path=RETURNS_FILE):
    """Per-ticker annualized idio variance vs. its own sleeve -- see the
    module docstring's "Idiosyncratic risk" section. Zero for any ticker
    that is its sleeve's only member (no proxy error possible --
    MSTR/"Single Stock" is this case) or with insufficient realized-return
    history to estimate against its sleeve-mates.

    Estimated the same way regress_tickers() (optimize_tax_cvar_mvo.py)
    estimates FF5 idio_var -- a single-factor OLS regression,

        ticker_t = alpha + beta * sleeve_proxy_t + residual

    against a LEAVE-ONE-OUT sleeve proxy (the market-value-weighted average
    return of every OTHER member of t's sleeve -- excluding t itself, so a
    dominant-weight member like IWD isn't regressed against a proxy it
    mostly constitutes). Using a regression here -- rather than the simpler
    var(ticker - proxy), an implicit beta=1/no-intercept version of the
    same thing -- matters because sleeve-mates can have a genuinely
    different LOADING on the sleeve's common driver (e.g. IWF's higher beta
    to broad market moves than ITOT's) without that being idiosyncratic risk
    at all; the regression's beta absorbs that tilt, and mse_resid (annualized)
    is what's actually left over and unexplained -- the same
    "idiosyncratic-by-construction" reading regress_tickers()'s own
    docstring gives mse_resid for the FF5 case.

    ticker_weights: Series index=ticker -> current household weight, used
    to build each sleeve's proxy return as a market-value-weighted (not
    equal-weighted) average of a ticker's OTHER sleeve-mates.

    Returns Series index=ticker -> annualized idio variance (>= 0.0).
    """
    sleeve_members = {}
    for ticker, sleeve in sleeve_by_ticker.items():
        sleeve_members.setdefault(sleeve, []).append(ticker)

    returns_wide = pd.read_csv(returns_path, parse_dates=["Date"]).set_index("Date")

    tau2 = {t: 0.0 for t in sleeve_by_ticker}
    for sleeve, members in sleeve_members.items():
        if len(members) < 2:
            continue  # single-ticker sleeve: no proxy error is possible
        have_history = [t for t in members if t in returns_wide.columns]
        missing = sorted(set(members) - set(have_history))
        if missing:
            print(f"  [saa idio] {sleeve}: no return history for {missing}, "
                  f"treating their idio variance as 0")
        if len(have_history) < 2:
            continue
        ret = returns_wide[have_history].dropna(how="any") / 100.0
        w = ticker_weights.reindex(have_history).fillna(0.0)
        w = w / w.sum() if w.sum() > 0 else pd.Series(1.0 / len(have_history), index=have_history)
        for t in have_history:
            other = [m for m in have_history if m != t]
            w_other = w[other] / w[other].sum()
            sleeve_proxy = (ret[other] * w_other).sum(axis=1)
            n_obs = len(ret)
            if n_obs < MIN_OBS:
                print(f"  [saa idio] {sleeve}/{t}: only {n_obs} common obs with its sleeve-mates "
                      f"(< {MIN_OBS}), treating its idio variance as 0")
                continue
            x = sm.add_constant(sleeve_proxy.rename("proxy"))
            model = sm.OLS(ret[t], x).fit()
            tau2[t] = float(model.mse_resid * 12)  # SSR/(n-2), net of alpha+beta -- see docstring
    return pd.Series(tau2)


# User-selected policy mix:
#   - 35% custom asset-class target
#   - 35% all-equity image mix (cash forced to 0%)
#   - 30% workbook target from all_equity_saa_covariance.xlsx
#
# The two custom mixes are mapped into the workbook sleeve names used by the
# optimizer. This keeps the code compatible with the worksheet schema while
# reflecting the policy blend you requested. If you later want a more exact
# sleeve-level overlay, replace the simplified dictionaries below with the exact
# sleeve percentages you want for each workbook sleeve.
CUSTOM_ASSET_CLASS_TARGET = pd.Series({
    "US Large Growth": 0.60,
    "Intl Developed": 0.25,
    "US Small Growth": 0.05,
    "US Small Value": 0.05,
    "Emerging Markets": 0.05,
}, dtype=float)

IMAGE_ALL_EQUITY_TARGET = pd.Series({
    "US Large Growth": 0.51,
    "US Large Value": 0.00,
    "Intl Developed": 0.19,
    "US Mid Growth": 0.065,
    "US Mid Value": 0.065,
    "US Small Growth": 0.025,
    "US Small Value": 0.025,
    "Emerging Markets": 0.06,
}, dtype=float)
# Raw sleeve weights above sum to 94%, not 100% -- normalize here so every
# sleeve keeps its relative share and the mix still nets to a full 100%
# input to build_blended_policy's 35% weight, instead of silently landing
# at ~32.9% once the overall blend is renormalized.
IMAGE_ALL_EQUITY_TARGET = IMAGE_ALL_EQUITY_TARGET / IMAGE_ALL_EQUITY_TARGET.sum()

for _name, _mix in [
    ("CUSTOM_ASSET_CLASS_TARGET", CUSTOM_ASSET_CLASS_TARGET),
    ("IMAGE_ALL_EQUITY_TARGET", IMAGE_ALL_EQUITY_TARGET),
]:
    if abs(_mix.sum() - 1.0) > 1e-9:
        raise ValueError(f"{_name} must sum to 100%, got {_mix.sum():.4%}")


def build_blended_policy(policy):
    """Blend 35% custom asset mix + 35% all-equity image mix + 30% workbook policy."""
    blended = pd.Series(0.0, index=policy.index, dtype=float)
    for weight, mix in [
        (0.35, CUSTOM_ASSET_CLASS_TARGET),
        (0.35, IMAGE_ALL_EQUITY_TARGET),
        (0.30, policy),
    ]:
        for sleeve, value in mix.items():
            if sleeve in blended.index:
                blended[sleeve] += weight * value

    if blended.sum() <= 0.0:
        return policy.copy()

    blended = blended / blended.sum()
    return blended


def load_saa_inputs():
    targets = pd.read_excel(WORKBOOK, sheet_name="SAA Targets", header=2)
    targets = targets.set_index("Sleeve")
    policy_col = "Policy Target (no single stock)"
    covariance = pd.read_excel(WORKBOOK, sheet_name="Covariance", header=2, index_col=0)
    covariance.index = covariance.index.astype(str)
    covariance.columns = covariance.columns.astype(str)
    targets = targets.loc[targets.index.astype(str).isin(covariance.index)]
    targets = targets.loc[pd.to_numeric(targets[policy_col], errors="coerce").notna()]
    covariance = covariance.loc[targets.index, targets.index].astype(float)
    policy = targets[policy_col].astype(float)
    policy = build_blended_policy(policy)
    return list(targets.index), policy, covariance


def load_saa_te_inputs(current_ticker_weight, returns_wide=None):
    """Builds the calibration inputs main()'s own te_total() closure uses
    (sleeve covariance + CASH_SLEEVE, policy target + CASH_SLEEVE, per-ticker
    proxy-tracking/idio variance), as a reusable, path-independent bundle.

    `current_ticker_weight` (Series index=ticker, summed across accounts) is
    the ONLY path-specific-looking input, and even it isn't really -- it's
    the calibration weighting for the CASH_SLEEVE covariance estimate and the
    idio proxy-tracking variance (see estimate_cash_sleeve_row's and
    compute_proxy_tracking_variance's docstrings), not the position being
    scored. main() always passes the PRE-TRADE book's weights here; callers
    that want to score OTHER paths' post-trade weights against this same
    benchmark (see saa_te_of() below) should do the same, so every path is
    scored against one fixed, consistently-calibrated benchmark rather than
    each path re-calibrating its own.

    Returns {"sleeves", "target_vec", "covariance_mat", "tau2"} -- target_vec
    and covariance_mat are already reindexed to `sleeves`' order as plain
    numpy arrays, ready for saa_te_of().
    """
    if returns_wide is None:
        returns_wide = pd.read_csv(RETURNS_FILE, parse_dates=["Date"]).set_index("Date")

    sleeves, target, covariance = load_saa_inputs()
    cash_var, cash_cov = estimate_cash_sleeve_row(
        returns_wide, SLEEVE_BY_TICKER, current_ticker_weight, sleeves, "SHV")
    covariance[CASH_SLEEVE] = cash_cov.reindex(covariance.index).fillna(0.0)
    covariance.loc[CASH_SLEEVE] = cash_cov.reindex(covariance.columns).fillna(0.0)
    covariance.loc[CASH_SLEEVE, CASH_SLEEVE] = cash_var
    target[CASH_SLEEVE] = 0.0
    sleeves = sleeves + [CASH_SLEEVE]

    tau2 = compute_proxy_tracking_variance(SLEEVE_BY_TICKER, current_ticker_weight)

    return {
        "sleeves": sleeves,
        "target_vec": target.reindex(sleeves).to_numpy(float),
        "covariance_mat": covariance.reindex(index=sleeves, columns=sleeves).to_numpy(float),
        "tau2": tau2,
    }


def saa_te_of(x, tickers_row, sleeves, target_vec, covariance_mat, tau2):
    """Same TE(x) formula as main()'s te_total() closure -- active sleeve
    variance (vs. the SAA policy target) plus per-ticker proxy-tracking idio
    variance -- reimplemented generically via groupby-sum on (ticker, weight)
    pairs instead of a fixed position-indexed mapping matrix, so it can score
    ANY path's weight vector regardless of that path's own row ordering (not
    just the one `rows` ordering main() happens to build its matrices from).

    Returns (te_total, te_sleeve_only) -- the latter omits the idio term,
    mirroring optimize_tax_cvar_mvo.py's te/te_factor_only split.
    """
    sleeve_index = {s: i for i, s in enumerate(sleeves)}
    w = pd.Series(x, index=tickers_row)

    missing = sorted(set(tickers_row) - set(SLEEVE_BY_TICKER))
    if missing:
        raise RuntimeError(f"No SAA sleeve mapping for held ticker(s): {missing}")

    ticker_w = w.groupby(level=0).sum()

    sleeve_w = np.zeros(len(sleeves))
    for ticker, weight in ticker_w.items():
        sleeve_w[sleeve_index[SLEEVE_BY_TICKER[ticker]]] += weight

    active = sleeve_w - target_vec
    sleeve_variance = float(active @ covariance_mat @ active)
    idio_variance = float((ticker_w ** 2 * tau2.reindex(ticker_w.index).fillna(0.0)).sum())

    return float(np.sqrt(sleeve_variance + idio_variance)), float(np.sqrt(sleeve_variance))


def main():
    holdings = pd.read_csv(os.path.join(DATA_DIR, "holdings_household_clean.csv"))
    total = float(holdings["Market_Value"].sum())
    rows = holdings.reset_index(drop=True)
    x0 = (rows["Market_Value"] / total).to_numpy(float)
    current_ticker_weight = pd.Series(x0, index=rows["Ticker"].values).groupby(level=0).sum()
    returns_wide = pd.read_csv(RETURNS_FILE, parse_dates=["Date"]).set_index("Date")
    bench_wealth = pd.read_csv(
        os.path.join(DATA_DIR, "benchmark_wealth_curve.csv"), parse_dates=["Date"]
    ).set_index("Date")
    scenario_tickers = sorted(rows["Ticker"].unique())
    A_scen, bench_annual = build_cvar_scenarios(returns_wide, bench_wealth, scenario_tickers)
    A_row = A_scen.loc[rows["Ticker"]].to_numpy()
    bench_annual_vec = bench_annual.to_numpy()

    sleeves, target, covariance = load_saa_inputs()

    # Extend the workbook's sleeves with CASH_SLEEVE for SHV -- see the
    # module docstring's "SHV: a near-zero-beta ticker needed its OWN
    # sleeve" section. Both `target` and `covariance` are appended at the
    # END here, which lines up with `sleeves = sleeves + [CASH_SLEEVE]`
    # below -- required, since the code downstream reads `target` and
    # `covariance` positionally against `sleeves`' order, not by re-aligning
    # on labels.
    cash_var, cash_cov = estimate_cash_sleeve_row(returns_wide, SLEEVE_BY_TICKER, current_ticker_weight,
                                                   sleeves, "SHV")
    covariance[CASH_SLEEVE] = cash_cov.reindex(covariance.index).fillna(0.0)
    covariance.loc[CASH_SLEEVE] = cash_cov.reindex(covariance.columns).fillna(0.0)
    covariance.loc[CASH_SLEEVE, CASH_SLEEVE] = cash_var
    target[CASH_SLEEVE] = 0.0
    sleeves = sleeves + [CASH_SLEEVE]

    missing = sorted(set(rows["Ticker"]) - set(SLEEVE_BY_TICKER))
    if missing:
        raise RuntimeError(f"No SAA sleeve mapping exists for held ticker(s): {missing}")
    unknown_sleeves = sorted(set(SLEEVE_BY_TICKER.values()) - set(sleeves))
    if unknown_sleeves:
        raise RuntimeError(f"SAA mapping references missing workbook sleeve(s): {unknown_sleeves}")

    sleeve_index = {sleeve: i for i, sleeve in enumerate(sleeves)}
    row_sleeves = rows["Ticker"].map(SLEEVE_BY_TICKER).to_numpy()
    mapping = np.zeros((len(sleeves), len(rows)))
    for j, sleeve in enumerate(row_sleeves):
        mapping[sleeve_index[sleeve], j] = 1.0

    # Proxy-tracking ("implementation risk") idio term -- see the module
    # docstring's "Idiosyncratic risk" section. Current per-ticker weight
    # (summed across accounts) is what compute_proxy_tracking_variance
    # weights each sleeve's proxy return by. SHV now maps 1:1 to
    # CASH_SLEEVE, so it naturally gets 0 idio here too -- no special-casing
    # needed, compute_proxy_tracking_variance already zeros any ticker
    # that's its sleeve's only member.
    tau2 = compute_proxy_tracking_variance(SLEEVE_BY_TICKER, current_ticker_weight)
    all_tickers = sorted(SLEEVE_BY_TICKER)
    ticker_pos = {t: i for i, t in enumerate(all_tickers)}
    ticker_G = np.zeros((len(all_tickers), len(rows)))
    for j, t in enumerate(rows["Ticker"].values):
        ticker_G[ticker_pos[t], j] = 1.0
    tau2_vec = tau2.reindex(all_tickers).fillna(0.0).to_numpy()

    tax_per_dollar = np.zeros(len(rows))
    taxable = rows["Is_Taxable"].to_numpy(bool)
    tax_per_dollar[taxable] = (
        rows.loc[taxable, "LT_Gain"].to_numpy(float) * TAX_LT
        + rows.loc[taxable, "ST_Gain"].to_numpy(float) * TAX_ST
    ) / rows.loc[taxable, "Market_Value"].to_numpy(float)

    is_mstr = (rows["Ticker"].to_numpy() == SAA_MSTR_TICKER)
    is_ira = ~taxable

    # NO_ADD_TICKERS (SHV) excluded from the floor below -- its current
    # weight sits just under MIN_TICKER_WEIGHT, and its no-buy upper bound
    # (below) would conflict with a floor requiring more than that. A ticker
    # held ONLY in the IRA (e.g. XLB, SCHH -- no Joint row exists for them at
    # all) is excluded for the same reason: every IRA row is capped at its
    # current weight (see the IRA-freeze bounds below), so there's no row
    # left anywhere that could still buy such a ticker up to the floor.
    tickers_arr = rows["Ticker"].to_numpy()
    ira_only_tickers = {t for t in rows["Ticker"].unique() if np.all(is_ira[tickers_arr == t])}
    ticker_groups = {
        ticker: np.flatnonzero(tickers_arr == ticker)
        for ticker in rows["Ticker"].unique()
        if ticker not in NO_ADD_TICKERS and ticker not in ira_only_tickers
    }
    is_no_add = rows["Ticker"].isin(NO_ADD_TICKERS).to_numpy()

    def active(x):
        return mapping @ x - target.to_numpy()

    def idio_variance(x):
        """Proxy-tracking variance, full (not active) ticker weight -- the
        policy target holds no individual tickers, only sleeve weights, so
        it contributes none of this term. See the module docstring."""
        w_ticker = ticker_G @ x
        return float((w_ticker ** 2 * tau2_vec).sum())

    def te_total(x):
        active_sleeves = active(x)
        sleeve_variance = float(active_sleeves @ covariance.to_numpy() @ active_sleeves)
        return float(np.sqrt(sleeve_variance + idio_variance(x)))

    # Turnover slack: turn_i >= |x_i - x0_i|. This keeps the L1 penalty
    # linear and gives SLSQP the same smooth QP formulation as the FF5 path.
    n = len(x0)
    ix_x = slice(0, n)
    ix_turn = slice(n, 2 * n)

    def objective(v):
        x = v[ix_x]
        turn = v[ix_turn]
        active_sleeves = active(x)
        sleeve_variance = float(active_sleeves @ covariance.to_numpy() @ active_sleeves)
        return sleeve_variance + idio_variance(x) + SAA_TURNOVER_LAMBDA * float(turn.sum())

    def ticker_floor(x, indices):
        return float(x[indices].sum() - MIN_TICKER_WEIGHT)

    with open(METRICS, encoding="utf-8") as f:
        metrics_doc = json.load(f)
    metrics_doc.setdefault("saa", {})

    for label, cfg in SAA_PATHS.items():
        tax_budget = cfg["tax_budget"]
        # MSTR carve-out: when True, MSTR's rate is zeroed out of the
        # budget-facing tax vector so its sale is free from the tax-budget
        # constraint's point of view (the original "SAA" path). When False
        # (the new 3% path), MSTR's realized tax counts against the same
        # budget as every other position -- `tax_per_dollar` itself
        # (unchanged either way) still reports MSTR's true tax cost in the
        # trade table.
        if cfg["snow_carveout"]:
            tax_per_dollar_budget = np.where(is_mstr, 0.0, tax_per_dollar)
        else:
            tax_per_dollar_budget = tax_per_dollar

        def tax_cost_fraction(x, tax_per_dollar_budget=tax_per_dollar_budget):
            return float((np.maximum(x0 - x, 0.0) * taxable * tax_per_dollar_budget).sum())

        def true_tax_cost_fraction(x):
            """Real cash owed in realized-gain tax, using the UNBUDGETED
            tax_per_dollar (never the carve-out vector) -- an excluded
            ticker's sale still needs real cash reserved to pay its tax even
            when that tax doesn't count against the budget constraint."""
            return float((np.maximum(x0 - x, 0.0) * taxable * tax_per_dollar).sum())

        constraints = [
            # sum(x) == 1 - tax reserve: proceeds spent paying this year's
            # realized-gain tax aren't available for reinvestment, so the
            # post-trade book is under 100% invested by exactly that much --
            # the shortfall is cash set aside to cover the tax bill.
            {"type": "eq", "fun": lambda v: float(v[ix_x].sum() + true_tax_cost_fraction(v[ix_x]) - 1.0)},
            {"type": "ineq", "fun": lambda v, tax_budget=tax_budget: tax_budget - tax_cost_fraction(v[ix_x])},
        ]
        # No new money into the IRA is enforced below as a per-row upper
        # bound (x[i] <= x0[i] for every IRA row -- see `bounds`), not a
        # constraint here. Annual contribution limits are far below what a
        # reweighting-scale trade would need, and unlike a taxable account,
        # even reallocating WITHIN the IRA (sell one position to fund a buy
        # of another) is disallowed -- every IRA position may only be
        # trimmed or held. Same rule as optimize_tax_cvar_mvo.py's Paths C/D.
        constraints.extend(
            {"type": "ineq", "fun": lambda v, indices=indices: ticker_floor(v[ix_x], indices)}
            for indices in ticker_groups.values()
        )
        constraints.extend([
            {"type": "ineq", "fun": lambda v: v[ix_turn] - (v[ix_x] - x0)},
            {"type": "ineq", "fun": lambda v: v[ix_turn] + (v[ix_x] - x0)},
        ])
        # MSTR's row(s) can't be sold below (1 - cap) of their current weight and
        # can't be bought above their current weight -- the glide path is for
        # reducing concentration, not reallocating into it. Same cap for both
        # paths; the 3% path additionally has to pay for that sale out of its
        # own (bigger) tax budget since MSTR is no longer carved out.
        # NO_ADD_TICKERS rows (SHV) are capped above at their current weight --
        # may be trimmed or sold, never bought further. Every IRA row is
        # capped the same way -- no new money into the IRA, not even by
        # reallocating from another IRA position (see the module docstring's
        # "No new money into the IRA" section).
        bounds = []
        for i in range(len(x0)):
            if is_mstr[i]:
                bounds.append((x0[i] * (1.0 - SAA_MSTR_SELL_CAP), x0[i]))
            elif is_no_add[i] or is_ira[i]:
                bounds.append((0.0, x0[i]))
            else:
                bounds.append((0.0, 1.0))
        v0 = np.zeros(2 * n)
        v0[ix_x] = x0
        result = minimize(
            objective,
            v0,
            method="SLSQP",
            bounds=bounds + [(0.0, 1.0)] * n,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 2000, "disp": False},
        )
        if not result.success:
            raise RuntimeError(f"SAA optimizer ({label}) failed: {result.message}")

        x_new = result.x[ix_x]
        trades = build_trade_table(rows, x0, x_new, tax_per_dollar, total)
        trade_csv = os.path.join(DATA_DIR, cfg["trade_csv"])
        trades.to_csv(trade_csv, index=False)

        tax_cost_total = float((trades["Tax_Cost_Dollars"]).sum())
        tax_cost_mstr = float(trades.loc[trades["Ticker"] == SAA_MSTR_TICKER, "Tax_Cost_Dollars"].sum())
        # "what actually counted against tax_budget": ex-MSTR when carved
        # out, the full total when MSTR counts against the budget too.
        tax_cost_budgeted = tax_cost_total if not cfg["snow_carveout"] else tax_cost_total - tax_cost_mstr
        active_before = active(x0)
        active_after = active(x_new)
        te_sleeve_before = float(np.sqrt(active_before @ covariance.to_numpy() @ active_before))
        te_sleeve_after = float(np.sqrt(active_after @ covariance.to_numpy() @ active_after))
        te_before = te_total(x0)
        te_after = te_total(x_new)
        cvar_before = cvar_of(x0, A_row, bench_annual_vec)
        cvar_after = cvar_of(x_new, A_row, bench_annual_vec)
        metrics_doc["paths"][label] = {
            "feasible": True,
            "te": te_after,
            "te_sleeve_only": te_sleeve_after,
            "cvar": cvar_after,
            "cvar_before": cvar_before,
            "tax_cost": tax_cost_budgeted,
            "tax_cost_mstr": tax_cost_mstr,
            "tax_cost_total": tax_cost_total,
            "tax_budget": tax_budget,
            "snow_carveout": cfg["snow_carveout"],
            "turnover_lambda": SAA_TURNOVER_LAMBDA,
            "sell_cap": SAA_MSTR_SELL_CAP,
            "turnover": 0.5 * float(np.abs(x_new - x0).sum()),
            "te_before": te_before,
            "objective": "all-equity SAA covariance vs. policy target (0% single stock)",
        }
        metrics_doc["saa"][label] = {
            "workbook": os.path.basename(WORKBOOK),
            "target": {sleeve: float(target[sleeve]) for sleeve in sleeves},
            "current": {sleeve: float((mapping @ x0)[i]) for i, sleeve in enumerate(sleeves)},
            "proposed": {sleeve: float((mapping @ x_new)[i]) for i, sleeve in enumerate(sleeves)},
            "ticker_sleeves": SLEEVE_BY_TICKER,
        }

        summary_txt = os.path.join(DATA_DIR, cfg["summary_txt"])
        with open(summary_txt, "w", encoding="utf-8") as f:
            f.write(f"All-Equity SAA Covariance Minimum-Tracking-Error Optimization ({label})\n")
            f.write(f"Workbook: {WORKBOOK}\n")
            f.write(f"Policy-target TE: {te_before:.4%} -> {te_after:.4%} "
                    f"(sleeve-only, no proxy-tracking term: {te_sleeve_before:.4%} -> {te_sleeve_after:.4%})\n")
            if cfg["snow_carveout"]:
                f.write(f"Tax budget: {tax_budget:.2%} of household value/year "
                        f"(MSTR's own tax excluded from this budget)\n")
                f.write(f"Tax cost ex-MSTR (counts against budget): ${tax_cost_budgeted:,.2f} "
                        f"({tax_cost_budgeted / total:.4%} of household value)\n")
                f.write(f"MSTR's own tax cost (excluded from budget): ${tax_cost_mstr:,.2f}\n")
            else:
                f.write(f"Tax budget: {tax_budget:.2%} of household value/year "
                        f"(MSTR's own tax counts against this budget, no carve-out)\n")
                f.write(f"Tax cost counting against budget (incl. MSTR): ${tax_cost_budgeted:,.2f} "
                        f"({tax_cost_budgeted / total:.4%} of household value)\n")
                f.write(f"  of which MSTR's own tax cost: ${tax_cost_mstr:,.2f}\n")
            f.write(f"SAA turnover penalty lambda: {SAA_TURNOVER_LAMBDA}\n")
            f.write(f"MSTR sale capped at {SAA_MSTR_SELL_CAP:.0%} of its current position this run\n")
            f.write(f"Total realized tax cost (both): ${tax_cost_total:,.2f} "
                    f"({tax_cost_total / total:.4%} of household value)\n")
            f.write(f"One-way turnover: {0.5 * np.abs(x_new - x0).sum():.4%}\n")
            f.write("\nTicker-to-sleeve proxies:\n")
            for ticker, sleeve in sorted(SLEEVE_BY_TICKER.items()):
                f.write(f"  {ticker}: {sleeve}\n")
            f.write("\nProxy-tracking annualized variance by ticker (0 = sleeve's only member, "
                     "or no history):\n")
            for ticker in sorted(all_tickers):
                f.write(f"  {ticker}: {tau2.get(ticker, 0.0):.4f}  (vol {np.sqrt(tau2.get(ticker, 0.0)):.2%})\n")
        print(f"Wrote {trade_csv}")
        print(f"[{label}] SAA TE: {te_before:.4%} -> {te_after:.4%} "
              f"(sleeve-only: {te_sleeve_before:.4%} -> {te_sleeve_after:.4%})")

    with open(METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics_doc, f, indent=2)


if __name__ == "__main__":
    main()

