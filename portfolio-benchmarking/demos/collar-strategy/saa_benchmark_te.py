"""
saa_benchmark_te.py
=====================================================================
Answers: "What is every path's Tracking Error against the SAA sleeve/policy
benchmark, holding each path's actual optimized trade (Path C's own FF5
solve, Path D's own FF5 solve, SAA's own sleeve solve, etc.) exactly as-is?"

Context (see the earlier finding this script exists to fix): the dashboard's
"Tracking Error" numbers are NOT comparable across paths, because Path C/D
are measured against the FF5 factor-model max-Sharpe benchmark
(optimize_tax_cvar_mvo.py's te_of()) while SAA/SAA_3pct are measured against
a completely different benchmark -- the 13-sleeve Excel policy target in
all_equity_saa_covariance.xlsx (optimize_saa_mvo.py's te_total()). Comparing
those numbers side by side (as the dashboard and the collar report both do)
implicitly compares apples to oranges.

This script does NOT re-run any optimizer and does NOT change what any path
actually trades. It takes each path's already-optimized post-trade weights
(the same Proposed_Weight columns collar_cvar_analysis.py already reads) and
scores ALL of them -- including Path C and Path D, which were never solved
against this benchmark -- through the SAME sleeve-covariance TE formula
optimize_saa_mvo.py uses for its own path:

    active_sleeves = (sleeve-aggregated weight) - (sleeve policy target)
    sleeve_variance = active_sleeves' @ sleeve_covariance @ active_sleeves
    idio_variance   = sum_ticker (ticker weight)^2 * tau2[ticker]   (proxy-tracking term)
    TE = sqrt(sleeve_variance + idio_variance)

Implementation note: `load_saa_te_inputs`/`saa_te_of` are imported straight
from optimize_saa_mvo.py itself (added there so the rebalance dashboard can
share the exact same implementation) -- this file no longer keeps its own
copy of the formula, only a thin wrapper that derives the pre-trade
current_ticker_weight this package's DataFrame-shaped path_weights need.
optimize_saa_mvo.py's own te_total() (inside main()) builds the same formula
with fixed position-indexed matrices tied to one specific row ordering;
saa_te_of() reimplements it generically via groupby-sum on (ticker, weight)
pairs instead, so it can score ANY path's weight vector regardless of that
path's own row ordering -- verified below to reproduce optimize_saa_mvo.py's
own reported "te" for the SAA path exactly (see the [OK] check printed by
main()).

The calibration inputs (sleeve covariance, policy target, and the tau2 proxy-
tracking variances) are all fixed reference data -- none of them depend on
which path is being scored -- computed exactly once from the pre-trade
("before") book, identically to how optimize_saa_mvo.py computes them, then
reused to score every path.

Output: prints a comparison table and writes saa_benchmark_te_comparison.csv
in this folder.
"""

import os
import sys

import pandas as pd

# This script lives at portfolio-benchmarking/demos/collar-strategy/, three
# levels below portfolio-benchmarking/ itself -- walk up three dirnames to
# reach it and put it on sys.path so `optimize_tax_cvar_mvo`/`optimize_saa_mvo`
# resolve as top-level modules exactly as they do when run directly.
BENCHMARKING_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BENCHMARKING_DIR)

from collar_cvar_analysis import load_path_weights, PATH_TRADE_FILES  # noqa: E402
from optimize_tax_cvar_mvo import DATA_DIR  # noqa: E402
from optimize_saa_mvo import load_saa_te_inputs as _load_saa_te_inputs, saa_te_of  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(OUT_DIR, "saa_benchmark_te_comparison.csv")


def load_saa_te_inputs(path_weights):
    """Thin wrapper around optimize_saa_mvo.load_saa_te_inputs(): derives the
    pre-trade current_ticker_weight this collar-analysis package works with
    (a DataFrame per path, not optimize_saa_mvo.main()'s own row-indexed
    vectors) and hands it off to the CANONICAL implementation, so this
    package and the rebalance dashboard can never compute two different
    SAA-benchmark TE numbers for the same weights.
    """
    before = path_weights["before"]
    current_ticker_weight = before.groupby("Ticker")["Weight"].sum()
    return _load_saa_te_inputs(current_ticker_weight)


def main():
    print("Loading each path's post-trade weights (same trade CSVs collar_cvar_analysis.py reads)...")
    path_weights = load_path_weights()
    inputs = load_saa_te_inputs(path_weights)

    with open(os.path.join(DATA_DIR, "path_metrics.json")) as f:
        import json
        path_metrics = json.load(f)

    labels_order = ["before"] + [l for l in PATH_TRADE_FILES if l in path_weights]
    rows = []
    for label in labels_order:
        wdf = path_weights[label]
        te_total, te_sleeve_only = saa_te_of(
            wdf["Weight"].values, wdf["Ticker"].values,
            inputs["sleeves"], inputs["target_vec"], inputs["covariance_mat"], inputs["tau2"])
        rows.append({"path": label, "saa_te": te_total, "saa_te_sleeve_only": te_sleeve_only})

    result = pd.DataFrame(rows).set_index("path")

    # Sanity check: this generic reimplementation should reproduce
    # optimize_saa_mvo.py's OWN reported TE for the SAA path exactly (same
    # formula, same inputs, just row-order-independent construction).
    saa_reported = path_metrics["paths"].get("SAA", {}).get("te")
    if saa_reported is not None:
        diff = result.loc["SAA", "saa_te"] - saa_reported
        status = "OK" if abs(diff) < 1e-6 else "WARNING -- MISMATCH"
        print(f"\n[{status}] Reimplemented SAA-path TE = {result.loc['SAA','saa_te']:.6%}, "
              f"optimize_saa_mvo.py's own reported TE = {saa_reported:.6%}, diff={diff:.2e}")

    print(f"\n{'Path':<10} {'SAA-benchmark TE':>18} {'(sleeve-only)':>15}")
    for label, r in result.iterrows():
        print(f"{label:<10} {r['saa_te']:>17.2%} {r['saa_te_sleeve_only']:>14.2%}")

    result.to_csv(OUT_CSV)
    print(f"\nWrote {OUT_CSV}")
    print(
        "\nReminder: no optimizer was re-run and no path's trade changed -- this only re-SCORES each "
        "path's already-optimized post-trade weights against the SAA sleeve/policy benchmark, so every "
        "path's number here is now measured the same way and IS comparable, unlike the dashboard's "
        "per-path TE (FF5 benchmark for C/D, SAA sleeve benchmark for SAA/SAA_3pct)."
    )


if __name__ == "__main__":
    main()
