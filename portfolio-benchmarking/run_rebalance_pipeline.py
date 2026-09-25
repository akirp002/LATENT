r"""
run_rebalance_pipeline.py

The one entry point for the fabricated household portfolio track:
`generate_sample_real_holdings.py`'s own hardcoded fake holdings in,
`rebalance_dashboard.html` out. Runs the six scripts of that chain in
dependency order, each as its own subprocess, and refuses to run a step
whose inputs aren't there.

    python run_rebalance_pipeline.py --list        # the rundown, no execution
    python run_rebalance_pipeline.py --dry-run     # + what would run vs. skip
    python run_rebalance_pipeline.py               # run it
    python run_rebalance_pipeline.py --offline     # reuse cached prices
    python run_rebalance_pipeline.py --from optimize
    python run_rebalance_pipeline.py --only dashboard --force

=====================================================================
WHY THIS FILE EXISTS NOW (AND DIDN'T BEFORE)
=====================================================================

`docs/project-architecture-map.md` argued against an orchestrator, and drew
the line for when that would change: "if portfolio-benchmarking's pipeline A
ever gets pointed at a real, live portfolio, it's worth treating as closer to
the production track -- real account-level data and a number someone relies
on deserve (1) a single run_all.py so five ordered scripts don't have to be
remembered and run by hand, and (2) synthetic-data outputs should never
quietly get treated as real ones."

Both conditions now hold in spirit even though the household track's input
is fabricated: `generate_sample_real_holdings.py` writes a two-account
holdings file in exactly the shape a real brokerage export would take,
`optimize_tax_cvar_mvo.py` produces trade proposals against that fabricated
cost basis, and `rebalance_dashboard.html` is a deliverable in the same
shape a real one would be. Per `quant_agent_guidelines.md` §1 that is the
**production track**: idempotent runs, logging at pipeline boundaries, fail
fast at the boundary rather than deep inside. This file is that, and nothing
more -- it contains no modeling logic whatsoever. Every number still comes
from the step scripts; if this file disappeared, running the six scripts by
hand in STEPS order would produce byte-identical output.

It covers ONLY the household-holdings track. The pipeline-A sample-holdings
track (`generate_sample_holdings.py` -> `fetch_holdings_prices.py` -> ... ->
`portfolio_factor_exposures.py`) and the synthetic tracking-error prototype
(`benchmark_tracking_error_model.py`) are deliberately NOT wired in here --
each of the three fabricated-data tracks in this repo is kept on its own
button so none of their demo numbers is ever silently mixed with another's.

=====================================================================
THE CHAIN
=====================================================================

    generate_sample_real_holdings.py's own hardcoded fabricated data (no manual drop)
        |
        v
    [ingest]      generate_sample_real_holdings.py
                    fabricated two-account book -> one row per (account,
                    ticker), ST/LT unrealized gain, Is_Taxable per account
        |        -> holdings_household_clean.csv, holdings_household_tickers.txt
        v
    [prices]      fetch_household_holdings_prices.py            [NETWORK: yfinance]
        |        -> holdings_household_monthly_returns.csv
        v
    [benchmark]   construct_benchmark_mvo.py    <-- ..\factor-research\Data\
        |           max-Sharpe long-only over FF5, 60-70% Mkt-RF     FF_plus_
        |           -> 60/40 Mkt-RF/RMW policy mix, buy-and-hold     macro_
        |        -> benchmark_mvo_weights.csv, benchmark_wealth_curve.csv
        v
    [optimize]    optimize_tax_cvar_mvo.py
        |           min tracking error + L1 turnover penalty, s.t. realized-
        |           gain tax <= 2%/yr (MSTR excluded) and the ticker floor.
        |           No CVaR constraint -- Active CVaR is computed and
        |           reported, not enforced. Path C caps this year's MSTR
        |           sale at 70% of its current position, Path D at 30%.
        |        -> trade_proposals_path{C,D}.csv, path_metrics.json,
        |           optimizer_summary.txt
        v
    [attribution] portfolio_risk_attribution.py
        |           Euler split of TE by factor, tail-conditional CVaR
        |           attribution, cross-sectional beta z-scores
        |        -> risk_attribution.json
        v
    [dashboard]   generate_rebalance_dashboard.py
                 -> rebalance_dashboard.html

`construct_benchmark_mvo.py` sits in the middle of the chain but depends on
neither of the two steps above it -- it only needs the factor panel. It is
sequenced there because `optimize` needs its output, not because `prices`
feeds it.

=====================================================================
MANAGING THIS OVER A 20-25 YEAR HORIZON
=====================================================================

The point of the pipeline is that the answer is meant to CHANGE every year,
and the mechanism for that is re-running it, not re-deriving it. Cadence:

  - New holdings snapshot     -> full run. Weights, cost basis and the ST/LT
    gain split all move; every number downstream is stale the moment
    `generate_sample_real_holdings.py`'s fabricated inputs change.
  - Annually, at minimum      -> full run even with no other change. The 2%
    tax budget is an ANNUAL budget: it resets, and Paths C/D's MSTR sale caps
    (70% / 30% of the then-current position) are explicitly one year of a
    multi-year glide path -- same for the SAA path's own 2% budget and 50%
    MSTR cap. Last year's trade list is not this year's.
  - Rate/bracket changes      -> TAX_LT / TAX_ST in optimize_tax_cvar_mvo.py.
    A 25-year horizon will outlive the current rate schedule.
  - Policy or mandate change  -> CVAR_FLOOR, TAX_BUDGET, the reweighting-only
    scope (the optimizer introduces no ticker not already held), the 60-70%
    Mkt-RF band in construct_benchmark_mvo.py. All are constants in one file
    each, by design.
  - Upstream factor rebuild   -> if `factor-research`'s
    FF_plus_macro_workable.csv is rebuilt (new FF vintage, changed cleaning),
    the benchmark and every path silently change on the next run. There is no
    version pin between the two projects, just a live path reference. Re-run
    with --force after any such change rather than assuming insulation.

Structural caveats that a long horizon makes worse, recorded here because
this is the file someone opens first in five years:

  - The book is 16 tickers with one large single-stock position (see
    `generate_sample_real_holdings.py` for the exact fabricated weight).
    Both the CVaR scenarios and the z-score universe are built from THAT
    set. As the book diversifies, the z-scores' meaning drifts (they are
    "tilted vs. what this household owns," never Barra-style market-wide
    scores).
  - CVaR scenarios use overlapping 12-month windows over the tickers' COMMON
    history, so the newest holding truncates the sample for everything. One
    recent IPO or new ETF sleeve can cut the scenario count hard; `optimize`
    prints the window count and date range every run -- watch it.
  - `ingest` sets taxable vs. tax-advantaged per ACCOUNT_TAX_MAP in
    `generate_sample_real_holdings.py`. A new account (401k, 529, HSA) needs
    that map extended or the position is silently dropped.

=====================================================================
STALENESS TRACKS CODE, NOT JUST DATA
=====================================================================

Each step's own script file (plus anything it imports data-shaping logic
from -- see `code_deps` on the "attribution" step, which imports directly
from optimize_tax_cvar_mvo.py) counts as an input for the up-to-date check,
same as its data files. Editing a script's math -- the min-allocation
constraint added to `optimize_tax_cvar_mvo.py` alongside this note is the
concrete example -- is exactly as much a reason to rerun as a new holdings
snapshot, and this pipeline treats it that way rather than only noticing
when a Data\ file's timestamp moves.
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "Data")
FACTOR_DATA_DIR = os.path.join(HERE, "..", "factor-research", "Data")

FF_PANEL = os.path.join(FACTOR_DATA_DIR, "FF_plus_macro_workable.csv")
RUN_LOG = os.path.join(DATA_DIR, "pipeline_run_log.json")


def d(*parts):
    return os.path.join(DATA_DIR, *parts)


# Each step: what it runs, what it must be able to read, what it promises to
# write. `inputs` drives both the preflight check and the staleness check;
# `optional_outputs` exist only in some branches (the optimizer's path files
# depend on which constraint sets turn out feasible) so they must not count
# toward "is this step's output complete".
STEPS = [
    {
        "key": "ingest",
        "script": "generate_sample_real_holdings.py",
        "title": "Generate the fabricated household holdings",
        "why": "self-contained fabricated two-account book -> one row per (account, "
               "ticker) with ST/LT unrealized gain; taxable vs. IRA set per account label.",
        "inputs": [],
        "outputs": [d("holdings_household_clean.csv"), d("holdings_household_tickers.txt")],
        "optional_outputs": [],
        "network": False,
    },
    {
        "key": "prices",
        "script": "fetch_household_holdings_prices.py",
        "title": "Fetch monthly price history",
        "why": "yfinance monthly total returns for every held ticker, in percent, "
               "back to 1990 where available.",
        "inputs": [d("holdings_household_tickers.txt")],
        "outputs": [d("holdings_household_monthly_returns.csv")],
        "optional_outputs": [],
        "network": True,
    },
    {
        "key": "benchmark",
        "script": "construct_benchmark_mvo.py",
        "title": "Construct the policy benchmark",
        "why": "Max-Sharpe long-only weights over the 5 FF factors with a 60-70% Mkt-RF "
               "band; held buy-and-hold. This is the thing tracking error is measured "
               "against. Shared with the sample-holdings track -- it reads no real data.",
        "inputs": [FF_PANEL],
        "outputs": [d("benchmark_mvo_weights.csv"), d("benchmark_wealth_curve.csv"),
                    d("benchmark_effective_weights.csv"), d("benchmark_mvo_summary.txt")],
        "optional_outputs": [],
        "network": False,
    },
    {
        "key": "optimize",
        "script": "optimize_tax_cvar_mvo.py",
        "title": "Solve the tax-constrained rebalance (Paths C / D)",
        "why": "Minimum tracking error plus an L1 per-position turnover penalty, subject only "
               "to realized-gain tax <= 2% of household value (MSTR excluded from that budget) "
               "and the minimum ticker floor -- no CVaR constraint. Path C caps this year's "
               "MSTR sale at 70% of its current position, Path D at 30%. Active CVaR is still "
               "computed and reported for context on every path.",
        "inputs": [d("holdings_household_clean.csv"), d("holdings_household_monthly_returns.csv"),
                   d("benchmark_mvo_weights.csv"), d("benchmark_wealth_curve.csv"), FF_PANEL],
        "outputs": [d("path_metrics.json"), d("optimizer_summary.txt")],
        "optional_outputs": [d("trade_proposals_pathC.csv"), d("trade_proposals_pathD.csv")],
        "network": False,
    },
    {
        "key": "saa",
        "script": "optimize_saa_mvo.py",
        "title": "Solve the all-equity SAA covariance path(s)",
        "why": "Independent minimum-tracking-error solve against the workbook's all-equity "
               "policy target (0% single stock). Solves two variants: Path SAA (2% tax budget, "
               "MSTR excluded from it) and Path SAA 3% (3% tax budget, MSTR's own tax counts "
               "against it instead of being carved out). Both cap this year's MSTR sale at 50% "
               "of its current position.",
        "inputs": [d("holdings_household_clean.csv"), d("path_metrics.json"),
                   os.path.join(HERE, "..", "all_equity_saa_covariance.xlsx")],
        "outputs": [d("trade_proposals_pathSAA.csv"), d("saa_optimizer_summary.txt"),
                    d("trade_proposals_pathSAA_3pct.csv"), d("saa_optimizer_summary_3pct.txt")],
        "optional_outputs": [],
        "network": False,
    },
    {
        "key": "attribution",
        "script": "portfolio_risk_attribution.py",
        "title": "Attribute each path's risk to factors",
        "why": "Euler decomposition of TE by factor, tail-conditional CVaR attribution "
               "(the residual is the idiosyncratic single-stock risk a factor benchmark "
               "cannot hedge), and cross-sectional beta z-scores. Pure analysis -- "
               "re-reads the optimizer's saved proposals, never re-optimizes.",
        "inputs": [d("holdings_household_clean.csv"), d("holdings_household_monthly_returns.csv"),
                   d("benchmark_mvo_weights.csv"), d("benchmark_wealth_curve.csv"),
                   d("path_metrics.json"), FF_PANEL],
        "outputs": [d("risk_attribution.json")],
        "optional_outputs": [],
        "network": False,
        # Imports optimize_tax_cvar_mvo.py directly (regress_tickers, FF_FACTOR_COLS,
        # WINDOW, ASSET_CLASS, ...) rather than just reading its saved output --
        # an edit there can change this step's numbers with no data file touched,
        # so its mtime has to count as an input too. See step_code_paths().
        "code_deps": ["optimize_tax_cvar_mvo.py"],
    },
    {
        "key": "dashboard",
        "script": "generate_rebalance_dashboard.py",
        "title": "Render the rebalance dashboard",
        "why": "Self-contained HTML: path cards, side-by-side metrics, per-position trade "
               "blotter (every row, including unmoved holdings), risk attribution, "
               "asset-class views, methodology. Pure presentation -- computes no numbers.",
        "inputs": [d("holdings_household_clean.csv"), d("benchmark_mvo_weights.csv"),
                   d("path_metrics.json"), d("risk_attribution.json"),
                   d("trade_proposals_pathSAA.csv"), d("trade_proposals_pathSAA_3pct.csv")],
        "outputs": [os.path.join(HERE, "rebalance_dashboard.html")],
        "optional_outputs": [],
        "network": False,
    },
]

STEP_KEYS = [s["key"] for s in STEPS]


def rel(path):
    """Display path relative to this folder, so `Data/x.csv` stays short and the
    one cross-project dependency stands out as `../factor-research/...`."""
    try:
        return os.path.relpath(path, HERE).replace("\\", "/")
    except ValueError:  # different drive on Windows
        return path


def newest_mtime(paths):
    """Most recent mtime among paths that exist; None if none exist."""
    stamps = [os.path.getmtime(p) for p in paths if os.path.exists(p)]
    return max(stamps) if stamps else None


def oldest_mtime(paths):
    stamps = [os.path.getmtime(p) for p in paths if os.path.exists(p)]
    return min(stamps) if stamps else None


def step_code_paths(step):
    """This step's own script plus any other script it imports from
    (`code_deps`) -- both count as inputs for staleness purposes. Editing a
    script's logic can change its output with no data file touched at all
    (this session's own min-allocation-constraint edit to
    optimize_tax_cvar_mvo.py is exactly that case), so treating code as
    read-only for staleness -- as an earlier version of this file did --
    misses real reasons to rerun."""
    paths = [os.path.join(HERE, step["script"])]
    paths += [os.path.join(HERE, dep) for dep in step.get("code_deps", [])]
    return paths


def missing_inputs(step):
    return [p for p in step["inputs"] if not os.path.exists(p)]


def step_state(step):
    """(state, detail) where state is one of:
         'blocked'  -- a required input does not exist; cannot run
         'stale'    -- must run: output missing, or an input is newer
         'current'  -- every required output exists and post-dates every input
    Staleness ignores optional_outputs on purpose: the optimizer's path files
    depend on which constraint sets turn out feasible, so a missing pathD.csv
    is a legitimate result, not incomplete output.
    """
    absent = missing_inputs(step)
    if absent:
        return "blocked", f"missing input: {', '.join(rel(p) for p in absent)}"

    not_written = [p for p in step["outputs"] if not os.path.exists(p)]
    if not_written:
        return "stale", f"output missing: {', '.join(rel(p) for p in not_written)}"

    newest_in = newest_mtime(list(step["inputs"]) + step_code_paths(step))
    oldest_out = oldest_mtime(step["outputs"])
    if newest_in is not None and oldest_out is not None and newest_in > oldest_out:
        return "stale", "the script (or a data input) is newer than the output"
    return "current", "up to date"


def print_rundown(show_state):
    print("=" * 78)
    print("FABRICATED HOUSEHOLD PORTFOLIO -- construction & rebalance pipeline")
    print("=" * 78)
    print(f"  working dir : {rel(HERE)}")
    print(f"  interpreter : {sys.executable}")
    print(f"  factor panel: {rel(FF_PANEL)}"
          f"{'' if os.path.exists(FF_PANEL) else '   [MISSING]'}")
    print()
    for i, step in enumerate(STEPS, start=1):
        tags = []
        if step["network"]:
            tags.append("NETWORK")
        tag = f"  [{' '.join(tags)}]" if tags else ""
        print(f"{i}. {step['key']:<12} {step['script']}{tag}")
        print(f"   {step['title']}")
        for line in _wrap(step["why"], 70):
            print(f"     {line}")
        _print_paths("reads ", step["inputs"])
        _print_paths("writes", step["outputs"])
        if step["optional_outputs"]:
            _print_paths("or    ", step["optional_outputs"])
        if step.get("code_deps"):
            _print_paths("code  ", step_code_paths(step))
        if show_state:
            state, detail = step_state(step)
            print(f"     STATE : {state.upper()} -- {detail}")
        print()


def _print_paths(label, paths):
    """One indented, wrapped `label: a, b, c` block -- these lists get long
    enough (the optimizer reads five files) to be unreadable on one line."""
    if not paths:
        print(f"     {label}: (nothing)")
        return
    joined = ", ".join(rel(p) for p in paths)
    for i, line in enumerate(_wrap(joined, 64)):
        print(f"     {label}: {line}" if i == 0 else f"     {' ' * len(label)}  {line}")


def _wrap(text, width):
    """Tiny greedy wrapper -- avoids a textwrap import for three call sites."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        lines.append(cur)
    return lines


def run_step(step):
    """Run one step as a subprocess. Returns (ok, elapsed_seconds).

    Deliberately a subprocess and not an import: every step script does its
    work under `if __name__ == "__main__":`, so importing them would run
    nothing, and their module-level constants (TAX_LT, CVAR_FLOOR, ...) are
    meant to be read as one file's configuration rather than mutated from
    here. Output streams straight through to this console -- the step scripts
    already print the diagnostics that matter (dropped tickers, CVaR scenario
    count and date range, binding vs. slack constraints), and swallowing that
    into a log buffer would hide exactly what a human needs to see.
    """
    print("-" * 78)
    print(f"[{step['key']}] {step['script']} -- {step['title']}")
    print("-" * 78)
    start = time.time()
    result = subprocess.run([sys.executable, step["script"]], cwd=HERE)
    elapsed = time.time() - start
    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (exit {result.returncode})"
    print(f"\n[{step['key']}] {status} in {elapsed:.1f}s\n")
    return ok, elapsed


def write_run_log(records, started_at):
    """Append-only-ish run log. One entry per pipeline invocation, kept so a
    later reader can tell when the dashboard's numbers were last actually
    recomputed rather than inferring it from file mtimes."""
    history = []
    if os.path.exists(RUN_LOG):
        try:
            with open(RUN_LOG) as f:
                history = json.load(f).get("runs", [])
        except (ValueError, OSError):
            history = []  # corrupt or unreadable log is not worth failing a good run over
    history.append({
        "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started_at)),
        "interpreter": sys.executable,
        "steps": records,
    })
    with open(RUN_LOG, "w") as f:
        json.dump({"runs": history[-50:]}, f, indent=2)
    print(f"Run log: {rel(RUN_LOG)}")


def select_steps(args):
    """Resolve --only / --from / --offline into the list of steps to attempt."""
    steps = STEPS
    if args.only:
        steps = [s for s in steps if s["key"] in args.only]
    elif args.from_step:
        start = STEP_KEYS.index(args.from_step)
        steps = STEPS[start:]
    if args.offline:
        steps = [s for s in steps if not s["network"]]
    return steps


def main():
    parser = argparse.ArgumentParser(
        description="Run the real-household portfolio construction and rebalance-dashboard "
                    "pipeline end to end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="steps: " + " -> ".join(STEP_KEYS),
    )
    parser.add_argument("--list", action="store_true",
                        help="print the pipeline rundown and exit, without checking state")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the rundown plus what would run vs. skip, then exit")
    parser.add_argument("--force", action="store_true",
                        help="re-run steps even when their outputs look up to date")
    parser.add_argument("--offline", action="store_true",
                        help="skip the network step and reuse the cached price history")
    parser.add_argument("--only", action="append", choices=STEP_KEYS, metavar="STEP",
                        help="run only this step (repeatable)")
    parser.add_argument("--from", dest="from_step", choices=STEP_KEYS, metavar="STEP",
                        help="start at this step and run everything after it")
    args = parser.parse_args()

    if args.only and args.from_step:
        parser.error("--only and --from are mutually exclusive")

    if args.list:
        print_rundown(show_state=False)
        return 0

    print_rundown(show_state=True)

    selected = select_steps(args)
    if not selected:
        print("Nothing selected.")
        return 0

    # Once any step in this run is going to execute, every step after it also
    # runs -- regardless of what its OWN inputs' timestamps say right now.
    # Those timestamps describe the filesystem as it is before this run
    # starts; a step's real inputs may be about to change moments from now
    # because an earlier step in this same invocation is going to rewrite
    # them. Chasing the exact dependency graph to avoid an occasional
    # unneeded rerun isn't worth the risk of the reverse mistake (silently
    # skipping a step whose input just changed) -- this pipeline runs in
    # seconds, so the cascade is cheap and the simpler rule is the safer one.
    plan = []
    upstream_will_run = False
    for step in selected:
        state, detail = step_state(step)
        if state == "blocked" and not args.force:
            # Blocked may be transient: an earlier step in THIS run produces it.
            produced_earlier = {p for s in selected[:selected.index(step)]
                                for p in s["outputs"] + s["optional_outputs"]}
            if all(p in produced_earlier for p in missing_inputs(step)):
                plan.append((step, "run", "input produced by an earlier step in this run"))
                upstream_will_run = True
                continue
            plan.append((step, "blocked", detail))
        elif args.force or state == "stale":
            plan.append((step, "run", "--force" if args.force else detail))
            upstream_will_run = True
        elif upstream_will_run:
            plan.append((step, "run", "an earlier step in this run will refresh its inputs"))
        else:
            plan.append((step, "skip", detail))

    print("PLAN")
    for step, action, detail in plan:
        print(f"  {action.upper():<8} {step['key']:<12} {detail}")
    print()

    blocked = [(s, d) for s, a, d in plan if a == "blocked"]
    if blocked:
        print("Cannot proceed -- these steps have inputs that neither exist nor get")
        print("produced by an earlier selected step:")
        for step, detail in blocked:
            print(f"  {step['key']}: {detail}")
        if any(FF_PANEL in missing_inputs(s) for s, _ in blocked):
            print(f"\n  The factor panel comes from the other project: run")
            print(f"  factor-research/transform_data_ff.py to build")
            print(f"  {rel(FF_PANEL)}.")
        return 1

    if args.dry_run:
        print("Dry run -- nothing executed.")
        return 0

    started_at = time.time()
    records, failed = [], None
    for step, action, _ in plan:
        if action == "skip":
            records.append({"step": step["key"], "result": "skipped"})
            continue
        ok, elapsed = run_step(step)
        records.append({"step": step["key"], "result": "ok" if ok else "failed",
                        "seconds": round(elapsed, 1)})
        if not ok:
            failed = step
            break

    print("=" * 78)
    for record in records:
        secs = f"{record['seconds']:>6.1f}s" if "seconds" in record else "      --"
        print(f"  {record['result'].upper():<8} {record['step']:<12} {secs}")

    if failed:
        downstream = STEP_KEYS[STEP_KEYS.index(failed["key"]) + 1:]
        print(f"\nStopped at [{failed['key']}]. Nothing downstream ran"
              f"{' (' + ', '.join(downstream) + ')' if downstream else ''}.")
        print("The outputs still on disk are from the PREVIOUS run -- they are not")
        print("consistent with the current inputs. Fix the failure and re-run before")
        print("reading the dashboard.")
        write_run_log(records, started_at)
        return 1

    print(f"\nDone in {time.time() - started_at:.1f}s.")
    dash = os.path.join(HERE, "rebalance_dashboard.html")
    if os.path.exists(dash):
        print(f"Dashboard: {dash}")
    write_run_log(records, started_at)
    return 0


if __name__ == "__main__":
    sys.exit(main())
