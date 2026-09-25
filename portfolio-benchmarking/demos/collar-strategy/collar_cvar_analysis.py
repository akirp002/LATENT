"""
collar_cvar_analysis.py
=====================================================================
Answers one question: "If I put a zero-cost option collar on the MSTR
position, how much does that change Active CVaR for each of the
already-optimized trade paths (Current / Path C / Path D / SAA / SAA 3%)?"

This does NOT re-run the optimizer and does NOT price real options. It
reuses portfolio-benchmarking/optimize_tax_cvar_mvo.py's own CVaR scenario
machinery (build_cvar_scenarios, cvar_of) -- the exact same 58 overlapping
12-month scenarios, same common-history window (Oct 2020 -> Jun 2026, capped
by MSTR's IPO), same worst-5%-of-58-scenarios convention already documented
and verified for the dashboard's "Active CVaR" numbers -- and asks a single
what-if: replace MSTR's scenario return in each 12-month window with a
collared version, holding every other ticker (and every path's post-trade
weights) exactly as the optimizer already produced them.

Collar mechanic (confirmed with the user)
------------------------------------------
- ONE collar per 12-month scenario window, struck at the start of that
  window (not rolled monthly): MSTR's own cumulative 12-month arithmetic
  return for scenario s (already stored as A_scen.loc["MSTR", s] by
  build_cvar_scenarios) is clipped to [PUT_FLOOR, CALL_CAP]:

      collared_snow_s = clip(raw_snow_annual_return_s, PUT_FLOOR, CALL_CAP)

  PUT_FLOOR  = -0.10  (10% OTM put -- annual loss cannot exceed -10%)
  CALL_CAP   = +0.20  (20% OTM call -- annual gain capped at +20%)

- Zero-cost / costless clip: no premium cash flow is modeled. The floor and
  cap simply clip the realized return each scenario -- no Black-Scholes, no
  implied-vol assumption. This is deliberate: modeling real premium would
  require an IV assumption for a name as volatile as MSTR, which would swing
  the answer more than the collar itself. This script answers "how much does
  bounding the outcome change the tail," not "what would this actually cost
  to trade."

- Only MSTR's row in the scenario matrix is touched. Every other ticker's
  scenario returns, every path's post-trade weights, and the benchmark's own
  scenario returns are exactly what optimize_tax_cvar_mvo.py already used --
  so any CVaR change reported here is coming ONLY from the collar, not from
  a different weighting or a different data window.

What "post-trade weight" means for each path
-----------------------------------------------
Each Data\\trade_proposals_path{X}.csv already has Current_Weight and
Proposed_Weight for every (Account, Ticker) row (see build_trade_table in
optimize_tax_cvar_mvo.py) -- this script reads Proposed_Weight directly
rather than re-deriving it, so it can never drift from what the optimizer
actually produced. CASH_RESERVE rows (the post-tax cash shortfall row) are
dropped before building the weight vector -- cash has no return series and
contributes 0 to both the collared and uncollared portfolio return, so
excluding it changes nothing about the CVaR comparison.

The "before" (do-nothing) baseline uses Current_Weight from any one trade
CSV, which is identical across all of them (it's the same pre-trade book).

Notional sizing
----------------
Each path's collar notional is its POST-TRADE MSTR dollar exposure:
  snow_notional = snow_weight (from Proposed_Weight) * total_household_value
(total_household_value comes straight from Data\\path_metrics.json -- the
same denominator the dashboard uses for every dollar figure).

Per-share strikes use MSTR's current implied price, back-solved from the
holdings file as Market_Value / Quantity (no live quote feed here):
  put_strike  = price * (1 + PUT_FLOOR)   e.g. price * 0.90
  call_strike = price * (1 + CALL_CAP)    e.g. price * 1.20
Shares/contracts are sized off the POST-TRADE notional (shares = notional /
price; contracts = shares / 100, standard equity-option lot size) -- i.e.
"how big a collar would you need on what's left of the MSTR position after
each path's trade," not on today's full pre-trade position.

Cross-check against the dashboard
------------------------------------
Before reporting anything, this script re-derives each path's UNCOLLARED
Active CVaR from scratch (same scenario matrix, same Proposed_Weight vectors)
and diffs it against Data\\path_metrics.json's stored "cvar" field -- the
exact number the rebalance dashboard (generate_rebalance_dashboard.py) reads
and displays on the path cards, banner, and bar chart. Any mismatch above a
tight floating-point tolerance is printed as a WARNING rather than silently
proceeding, so a stale trade CSV or a stale path_metrics.json can never
produce a collar comparison that quietly disagrees with what's on the
dashboard.

Output
------
Prints a comparison table (Active CVaR, uncollared vs. collared, and the
percentage-point improvement) for every path, plus each path's post-trade
MSTR weight, notional, strikes and contract count. Also writes that table to
collar_cvar_comparison.csv in this folder.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

# This script lives at portfolio-benchmarking/demos/collar-strategy/, three
# levels below portfolio-benchmarking/ itself -- walk up three dirnames to
# reach it and put it on sys.path so `optimize_tax_cvar_mvo` resolves as a
# top-level module exactly as it does when that script is run directly.
BENCHMARKING_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BENCHMARKING_DIR)

from optimize_tax_cvar_mvo import (  # noqa: E402
    DATA_DIR,
    build_cvar_scenarios,
    cvar_of,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(OUT_DIR, "collar_cvar_comparison.csv")

# Collar strikes, confirmed with the user: one collar per 12-month scenario
# window, struck at the window's start, zero-cost costless clip (no premium
# cash flow modeled).
PUT_FLOOR = -0.10  # 10% OTM put -- annual MSTR loss floored at -10%
CALL_CAP = 0.20    # 20% OTM call -- annual MSTR gain capped at +20%

PATH_TRADE_FILES = {
    "C": "trade_proposals_pathC.csv",
    "D": "trade_proposals_pathD.csv",
    "SAA": "trade_proposals_pathSAA.csv",
    "SAA_3pct": "trade_proposals_pathSAA_3pct.csv",
}


def load_path_weights():
    """Returns {label: DataFrame[Ticker, Weight]} for 'before' + every path
    found in PATH_TRADE_FILES, reading Proposed_Weight (and Current_Weight
    for 'before') straight from the optimizer's own saved trade CSVs.
    CASH_RESERVE rows are dropped -- no return series, contributes 0 either
    way to the CVaR comparison.
    """
    weights = {}
    before_done = False
    for label, fname in PATH_TRADE_FILES.items():
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  [skip] {fname} not found -- path '{label}' omitted")
            continue
        trades = pd.read_csv(fpath)
        trades = trades[trades["Ticker"] != "CASH_RESERVE"]

        if not before_done:
            weights["before"] = trades[["Ticker", "Current_Weight"]].rename(
                columns={"Current_Weight": "Weight"})
            before_done = True

        weights[label] = trades[["Ticker", "Proposed_Weight"]].rename(
            columns={"Proposed_Weight": "Weight"})
    return weights


def blended_cvar_of(x, tickers_row, A_scen, bench_annual_vec, put_floor, call_cap, coverage=1.0):
    """Same empirical CVaR as optimize_tax_cvar_mvo.cvar_of(), except MSTR's
    scenario return in each window is blended between its raw value and its
    collared (clipped) value by `coverage`:

        blended_s = coverage * clip(raw_s, put_floor, call_cap) + (1 - coverage) * raw_s

    coverage=1.0 -> fully collared (every MSTR share covered).
    coverage=0.0 -> uncollared (identical to cvar_of on the raw scenario matrix).
    coverage=0.3/0.5 -> a partial collar, e.g. only 30%/50% of the MSTR
    position is covered by the option structure and the rest stays naked --
    modeled as a straight linear blend of the two return streams, which is
    exactly what holding `coverage` fraction of shares collared and
    `1-coverage` fraction uncollared amounts to in portfolio-return terms.

    Every other ticker's scenario returns pass through untouched.
    """
    A_row = A_scen.loc[tickers_row].values.copy()  # (n_rows, n_scen)
    snow_mask = (tickers_row == "MSTR")
    if snow_mask.any():
        raw = A_row[snow_mask, :]
        collared = np.clip(raw, put_floor, call_cap)
        A_row[snow_mask, :] = coverage * collared + (1 - coverage) * raw
    return cvar_of(x, A_row, bench_annual_vec)


def blended_scenario_te_of(x, tickers_row, A_scen, bench_annual_vec, put_floor, call_cap, coverage=1.0):
    """Empirical 'scenario TE': the sample std dev of the SAME 58 scenario
    active returns (bench - portfolio, one number per 12-month window) that
    cvar_of()/blended_cvar_of() already build, with MSTR's row blended by
    `coverage` exactly like blended_cvar_of().

    This is NOT the dashboard's "Tracking Error" (optimize_tax_cvar_mvo.py's
    te_of(), a parametric factor-covariance + idiosyncratic-residual-variance
    quadratic form estimated from monthly betas/residuals -- it never reads
    A_scen at all, which is exactly why collaring MSTR in A_scen cannot move
    it). This IS the literal "standard deviation of active returns" a collar
    should mechanically shrink, computed the same empirical way CVaR already
    is -- so, unlike the dashboard's TE, this number responds to `coverage`.
    """
    A_row = A_scen.loc[tickers_row].values.copy()
    snow_mask = (tickers_row == "MSTR")
    if snow_mask.any():
        raw = A_row[snow_mask, :]
        collared = np.clip(raw, put_floor, call_cap)
        A_row[snow_mask, :] = coverage * collared + (1 - coverage) * raw
    active = (A_row.T @ x) - bench_annual_vec
    return float(np.std(active, ddof=1))


def collared_cvar_of(x, tickers_row, A_scen, bench_annual_vec, put_floor, call_cap):
    """Back-compat wrapper: full (100%-coverage) collar."""
    return blended_cvar_of(x, tickers_row, A_scen, bench_annual_vec, put_floor, call_cap,
                            coverage=1.0)


def load_all_data():
    """Loads every input collar_cvar_analysis.py and generate_collar_html.py both
    need: holdings, common-history returns, benchmark wealth curve, the
    dashboard's own path_metrics.json (for total_household_value and the
    cross-check), each path's post-trade weights, and the CVaR scenario
    matrix. Centralized here so the two scripts can never load these inputs
    two different ways and quietly drift apart.
    """
    holdings_all = pd.read_csv(os.path.join(DATA_DIR, "holdings_household_clean.csv"))
    returns_wide = pd.read_csv(
        os.path.join(DATA_DIR, "holdings_household_monthly_returns.csv"),
        parse_dates=["Date"]).set_index("Date")
    bench_wealth = pd.read_csv(
        os.path.join(DATA_DIR, "benchmark_wealth_curve.csv"),
        parse_dates=["Date"]).set_index("Date")
    with open(os.path.join(DATA_DIR, "path_metrics.json")) as f:
        path_metrics = json.load(f)

    snow_row = holdings_all[holdings_all["Ticker"] == "MSTR"].iloc[0]
    snow_price = float(snow_row["Market_Value"] / snow_row["Quantity"])

    path_weights = load_path_weights()
    all_tickers = sorted(set(
        t for df in path_weights.values() for t in df["Ticker"].unique()
    ))
    if "MSTR" not in all_tickers:
        raise RuntimeError("MSTR not found in any path's trade CSV -- nothing to collar.")

    A_scen, bench_annual = build_cvar_scenarios(returns_wide, bench_wealth, all_tickers)
    return {
        "path_metrics": path_metrics,
        "total_household_value": path_metrics["total_household_value"],
        "snow_price": snow_price,
        "path_weights": path_weights,
        "A_scen": A_scen,
        "bench_annual_vec": bench_annual.values,
    }


def main():
    print("Loading holdings, returns, benchmark history, and path_metrics.json "
          "(same inputs optimize_tax_cvar_mvo.py used)...")
    data = load_all_data()
    path_metrics = data["path_metrics"]
    total_household_value = data["total_household_value"]
    snow_price = data["snow_price"]
    path_weights = data["path_weights"]
    A_scen = data["A_scen"]
    bench_annual_vec = data["bench_annual_vec"]
    print(f"MSTR implied current price: ${snow_price:,.2f}/share "
          f"(Market_Value / Quantity from holdings_household_clean.csv)")

    n_scen = A_scen.shape[1]

    snow_raw = A_scen.loc["MSTR"].values
    snow_collared = np.clip(snow_raw, PUT_FLOOR, CALL_CAP)
    n_floored = int((snow_raw < PUT_FLOOR).sum())
    n_capped = int((snow_raw > CALL_CAP).sum())
    print(f"\nMSTR scenario returns: {n_scen} windows. "
          f"Collar (floor {PUT_FLOOR:+.0%} / cap {CALL_CAP:+.0%}) binds on "
          f"{n_floored} downside window(s) and {n_capped} upside window(s).")

    put_strike = snow_price * (1 + PUT_FLOOR)
    call_strike = snow_price * (1 + CALL_CAP)
    print(f"Collar strikes at current price: put ${put_strike:,.2f} ({PUT_FLOOR:+.0%}) / "
          f"call ${call_strike:,.2f} ({CALL_CAP:+.0%})")

    TOL = 1e-6  # tight -- these should match path_metrics.json exactly, same function/inputs
    rows = []
    for label, wdf in path_weights.items():
        tickers_row = wdf["Ticker"].values
        x = wdf["Weight"].values
        A_row_raw = A_scen.loc[tickers_row].values

        cvar_uncollared = cvar_of(x, A_row_raw, bench_annual_vec)

        # Cross-check against what the dashboard actually shows. "before"
        # isn't a keyed path in PATH_TRADE_FILES's dashboard sense but IS the
        # dashboard's paths["before"]["cvar"] -- check it too.
        dashboard_cvar = path_metrics["paths"].get(label, {}).get("cvar")
        if dashboard_cvar is not None:
            diff = cvar_uncollared - dashboard_cvar
            if abs(diff) > TOL:
                print(f"  [WARNING] Path '{label}': this script's uncollared CVaR "
                      f"({cvar_uncollared:.6%}) does NOT match path_metrics.json / dashboard "
                      f"({dashboard_cvar:.6%}), diff={diff:.6%} -- trade CSV or scenario data "
                      f"may be stale relative to path_metrics.json. Re-run "
                      f"optimize_tax_cvar_mvo.py before trusting this comparison.")
            else:
                print(f"  [OK] Path '{label}': uncollared CVaR matches dashboard "
                      f"({cvar_uncollared:.4%}), diff={diff:.2e}")

        cvar_collared = collared_cvar_of(x, tickers_row, A_scen, bench_annual_vec,
                                          PUT_FLOOR, CALL_CAP)
        snow_weight = float(wdf.loc[wdf["Ticker"] == "MSTR", "Weight"].sum())
        snow_notional = snow_weight * total_household_value
        snow_shares = snow_notional / snow_price
        snow_contracts = snow_shares / 100.0  # standard equity-option lot size

        rows.append({
            "path": label,
            "snow_weight": snow_weight,
            "snow_notional": snow_notional,
            "snow_shares": snow_shares,
            "snow_contracts": snow_contracts,
            "cvar_uncollared": cvar_uncollared,
            "cvar_collared": cvar_collared,
            "improvement_pp": (cvar_collared - cvar_uncollared) * 100,
            "dashboard_cvar_match": (
                None if dashboard_cvar is None else abs(cvar_uncollared - dashboard_cvar) <= TOL
            ),
        })

    result = pd.DataFrame(rows).set_index("path")
    result = result.reindex(["before"] + [l for l in PATH_TRADE_FILES if l in result.index])

    print(f"\n{'Path':<10} {'MSTR wt':>9} {'Notional':>14} {'Shares':>9} {'Contracts':>10} "
          f"{'CVaR (no collar)':>18} {'CVaR (collared)':>17} {'Improvement':>13}")
    for label, r in result.iterrows():
        print(f"{label:<10} {r['snow_weight']:>8.2%} ${r['snow_notional']:>12,.0f} "
              f"{r['snow_shares']:>9,.0f} {r['snow_contracts']:>9,.1f} "
              f"{r['cvar_uncollared']:>17.2%} {r['cvar_collared']:>16.2%} "
              f"{r['improvement_pp']:>+11.2f}pp")

    result.to_csv(OUT_CSV)
    print(f"\nWrote {OUT_CSV}")
    print(f"Put strike ${put_strike:,.2f} / Call strike ${call_strike:,.2f} "
          f"(off current implied price ${snow_price:,.2f}/share) apply to every path above -- "
          f"only the notional (shares/contracts) sized off each path's post-trade MSTR weight changes.")
    print(
        "\nReminder: this is a zero-cost costless-clip approximation (no premium modeled), "
        "one collar struck per 12-month scenario window (not rolled monthly), and only MSTR's "
        "row is collared -- every other ticker's scenario returns and every path's post-trade "
        "weights are exactly what optimize_tax_cvar_mvo.py already produced."
    )


if __name__ == "__main__":
    main()
