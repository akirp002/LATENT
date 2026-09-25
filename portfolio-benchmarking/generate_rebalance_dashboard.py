r"""
generate_rebalance_dashboard.py

Builds `rebalance_dashboard.html` -- the side-by-side comparison of every
trade-proposal path `optimize_tax_cvar_mvo.py` produces -- entirely from
that script's saved outputs. No optimization happens here; this is a pure
presentation layer, same spirit as `scripts/generate_showcase_charts.py`
for the other pipeline.

Reads (Data\ folder, all produced by optimize_tax_cvar_mvo.py):
  - holdings_household_clean.csv        current household holdings
  - path_metrics.json              TE / CVaR / tax cost per path, feasibility
  - trade_proposals_path{A,B,C,D}.csv (or trade_proposals.csv if the plain
    joint problem was feasible) -- per-position trade detail, one row per
    (ticker, account) INCLUDING unchanged (HOLD) rows, one file per feasible
    path. A path marked infeasible in path_metrics.json has no trade file
    and is rendered as an infeasible card instead. The Trade Detail tab
    shows every row for the selected path, not just the ones that traded --
    seeing what did NOT change is part of reviewing a rebalance.
  - risk_attribution.json (portfolio_risk_attribution.py) -- per-factor TE/CVaR
    risk contribution and active-exposure z-scores per path. Optional: if
    absent, the Risk Attribution tab is simply omitted (older outputs still
    render everything else).

Also computes (not read from any file): each path's Tracking Error against
the SAA sleeve/policy benchmark, via optimize_saa_mvo.load_saa_te_inputs()/
saa_te_of(). path_metrics.json's own "te" field uses a DIFFERENT benchmark
per path -- the FF5 factor max-Sharpe benchmark for Current/C/D, but the
13-sleeve Excel policy target for SAA/SAA_3pct -- so those numbers are not
comparable to each other. This SAA-benchmark TE re-scores every path's
already-optimized post-trade weights (no re-optimization) against the ONE
SAA benchmark, giving a second, directly comparable TE number alongside the
dashboard's native per-path one. See PATH_INFO step 6 in the Methodology tab
for the full explanation shown in the page itself.

Writes: rebalance_dashboard.html (same folder), self-contained -- publish
it with the Artifact tool, or open it directly in a browser.

Run this after any re-run of optimize_tax_cvar_mvo.py to keep the dashboard
in sync with the underlying numbers.
"""

import json
import os

import pandas as pd

from optimize_saa_mvo import load_saa_te_inputs, saa_te_of

DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
OUT_PATH = os.path.join(os.path.dirname(__file__), "rebalance_dashboard.html")
SAA_WORKBOOK = os.path.join(os.path.dirname(__file__), "..", "all_equity_saa_covariance.xlsx")

BENCH_FACTOR_LABELS = {"Mkt-RF": "Mkt-RF", "SMB": "SMB", "HML": "HML", "RMW": "RMW", "CMA": "CMA"}


def benchmark_mix_str(weights, min_pct=0.005):
    """'60% Mkt-RF / 25% RMW / 5% SMB / 5% HML / 5% CMA' style description,
    largest weight first, so the dashboard never hardcodes a specific mix
    that the benchmark construction script might later change."""
    ordered = weights.sort_values(ascending=False)
    parts = [f"{w:.0%} {BENCH_FACTOR_LABELS.get(k, k)}" for k, w in ordered.items() if w >= min_pct]
    return " / ".join(parts)


def benchmark_allocation_table(weights, title):
    rows = "".join(
        f"<tr><td>{label}</td><td>{weight:.1%}</td></tr>"
        for label, weight in weights.items()
    )
    return f"""<div class="bench-block">
      <div class="bench-title">{title}</div>
      <table class="bench-table"><thead><tr><th>Sleeve / factor</th><th>Allocation</th></tr></thead>
      <tbody>{rows}<tr class="bench-total"><td>Total</td><td>{weights.sum():.1%}</td></tr></tbody></table>
    </div>"""


def load_saa_policy_weights():
    targets = pd.read_excel(SAA_WORKBOOK, sheet_name="SAA Targets", header=2)
    policy_col = "Policy Target (no single stock)"
    covariance = pd.read_excel(SAA_WORKBOOK, sheet_name="Covariance", header=2, index_col=0)
    valid_sleeves = set(covariance.index.dropna().astype(str))
    targets = targets[["Sleeve", policy_col]].copy()
    targets[policy_col] = pd.to_numeric(targets[policy_col], errors="coerce")
    targets = targets.dropna()
    targets = targets[targets["Sleeve"].isin(valid_sleeves)]
    return targets.set_index("Sleeve")[policy_col]

# Every path label the optimizer can emit, in display order. Four paths are
# solved: Path C and Path D (optimize_tax_cvar_mvo.py, no CVaR constraint,
# differing only in their MSTR sale cap), Path SAA (optimize_saa_mvo.py, its
# own independent objective, 2% tax budget, and MSTR carve-out), and Path
# SAA_3pct (the same SAA objective, but a 3% tax budget with MSTR's own tax
# counting against it instead of being carved out). "before" (added
# separately, see shown_labels below) is the do-nothing baseline, not an
# optimized path.
ALL_PATH_LABELS = ["C", "D", "SAA", "SAA_3pct"]

# Static per-path narrative; {cap} is filled in from path_metrics.json's sell_cap
# where relevant. Order here is the display order.
PATH_INFO = {
    "C": {
        "title": "PATH C &middot; MSTR EXCLUDED FROM BUDGET, SALE CAPPED AT 70%",
        "css": "pc",
        "desc": "Minimize tracking error subject only to the 2% annual tax budget (no CVaR "
                "constraint -- Active CVaR is computed and shown for context, not enforced). "
                "MSTR's realized tax is excluded from that budget entirely (a separate taxable "
                "strategy is assumed to handle it), and this year's MSTR sale is capped at "
                "{cap:.0%} of its current position -- a fast glide path.",
    },
    "D": {
        "title": "PATH D &middot; MSTR EXCLUDED FROM BUDGET, SALE CAPPED AT 30%",
        "css": "pd",
        "desc": "Same as Path C, but this year's MSTR sale is capped at only {cap:.0%} of "
                "its current position -- a slower glide path.",
    },
    "SAA": {
        "title": "PATH SAA &middot; ALL-EQUITY STRATEGIC ALLOCATION",
        "css": "psaa",
        "desc": "Independent optimizer minimizing covariance-based tracking error to "
                "the all-equity workbook's policy target with 0% single stock, subject to its own "
                "2% annual tax budget and the minimum ticker floors. MSTR's realized tax is excluded "
                "from that budget (a separate taxable strategy is assumed to handle it), and this "
                "year's MSTR sale is capped at 50% of its current position.",
    },
    "SAA_3pct": {
        "title": "PATH SAA 3% &middot; SAME TARGET, NO MSTR CARVE-OUT",
        "css": "psaa3",
        "desc": "Same all-equity policy-target objective and 50% MSTR sale cap as Path SAA, but with "
                "the tax budget raised to 3% of household value AND MSTR's own realized tax counting "
                "against that budget like any other position (no carve-out). MSTR's tax bill alone can "
                "consume most or all of the budget, leaving less room to trim everything else.",
    },
}


def load_metrics():
    with open(os.path.join(DATA_DIR, "path_metrics.json")) as f:
        return json.load(f)


def load_trades(label):
    """Path A/B use trade_proposals_path{L}.csv; the plain joint solve (if
    ever feasible) writes trade_proposals.csv instead -- check both."""
    for fname in (f"trade_proposals_path{label}.csv", "trade_proposals.csv"):
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            return pd.read_csv(path)
    return None


def path_card_html(label, metrics, cap=None):
    info = PATH_INFO[label]
    feasible = metrics.get("feasible", False)
    if not feasible:
        return f'''<div class="path {info["css"]} infeasible">
      <div class="path-title">{info["title"]}</div>
      <div class="path-sub">{info["desc"].format(cap=cap or 0)}</div>
      <div class="infeasible-badge">INFEASIBLE -- this constraint set has no solution this year</div>
    </div>'''
    te, cvar = metrics["te"], metrics["cvar"]
    saa_te = metrics.get("saa_te")
    tax = metrics.get("tax_cost", 0.0)
    desc = info["desc"].format(cap=cap or 0)
    extra = ""
    if "tax_cost_snow" in metrics:
        if metrics.get("snow_carveout", True):
            extra = (f'<div class="note">MSTR\'s own tax (excluded, separate strategy): '
                     f'${metrics["tax_cost_snow"]:,.0f}. Total realized tax (both): '
                     f'${metrics["tax_cost_total"]:,.0f}.</div>')
        else:
            extra = (f'<div class="note">MSTR\'s own tax (counts against budget, no carve-out): '
                     f'${metrics["tax_cost_snow"]:,.0f} of ${metrics["tax_cost_total"]:,.0f} total '
                     f'budgeted tax.</div>')
    return f'''<div class="path {info["css"]}">
      <div class="path-title">{info["title"]}</div>
      <div class="path-sub">{desc}</div>
      <div class="path-mets">
        <div class="pm"><div class="pm-l">Tax Cost</div><div class="pm-v">${tax:,.0f}</div>
          <div class="pm-d">{tax/metrics.get("_total",1):.2%}</div></div>
        <div class="pm"><div class="pm-l">Track. Error</div><div class="pm-v">{te:.2%}</div>
          <div class="pm-d">own benchmark</div></div>
        <div class="pm"><div class="pm-l">TE vs. SAA Bench.</div><div class="pm-v">{saa_te:.2%}</div>
          <div class="pm-d">comparable</div></div>
        <div class="pm"><div class="pm-l">Active CVaR</div><div class="pm-v">{cvar:.2%}</div></div>
        <div class="pm"><div class="pm-l">Turnover</div><div class="pm-v">{metrics.get("turnover", 0.0):.1%}</div>
          <div class="pm-d">one-way</div></div>
      </div>
      {extra}
    </div>'''


FACTOR_LABELS = {
    "Mkt-RF": "Market", "SMB": "Size", "HML": "Value", "RMW": "Profit.", "CMA": "Invest.",
}
PATH_SHORT = {"before": "Current", "C": "Path C", "D": "Path D", "SAA": "SAA target",
              "SAA_3pct": "SAA 3%"}


def diverging_row(label, value, scale, fmt, harm_sign, title):
    """One horizontal diverging bar anchored to a centered zero line.

    harm_sign: +1 when a POSITIVE value is the harmful/overweight direction,
    -1 when a NEGATIVE one is. Colour encodes that reading, not the raw sign --
    a negative CVaR contribution deepens the loss and must not be painted with
    the same hue as a negative TE contribution, which hedges.
    """
    pct = 0.0 if scale <= 0 else min(abs(value) / scale, 1.0) * 50.0
    warm = (value * harm_sign) > 0
    side = ("left:50%;border-radius:0 4px 4px 0" if value >= 0
            else f"right:50%;border-radius:4px 0 0 4px")
    cls = "add" if warm else "hedge"
    return (f'<div class="rb" title="{title}">'
            f'<div class="rb-lab">{label}</div>'
            f'<div class="rb-track"><div class="rb-zero"></div>'
            f'<div class="rb-bar {cls}" style="{side};width:{pct:.3f}%"></div></div>'
            f'<div class="rb-val {cls}">{fmt(value)}</div></div>')


def risk_section_html(risk, key, total_key, fmt, harm_sign, extra_rows=None):
    """Small multiples: one panel per path, a shared scale across all panels so
    the panels are actually comparable, one series per panel so no categorical
    palette is needed at all."""
    paths = risk["paths"]
    factors = risk["factors"]
    vals = []
    for p in paths.values():
        vals += [abs(p[key][f]) for f in factors]
        if extra_rows:
            vals += [abs(p[r[0]]) for r in extra_rows]
    scale = max(vals) if vals else 1.0

    panels = []
    for label, p in paths.items():
        rows = "".join(
            diverging_row(FACTOR_LABELS.get(f, f), p[key][f], scale, fmt, harm_sign,
                          f"{f}: {fmt(p[key][f])}")
            for f in factors
        )
        for rkey, rlabel in (extra_rows or []):
            rows += diverging_row(rlabel, p[rkey], scale, fmt, harm_sign,
                                  f"{rlabel}: {fmt(p[rkey])}")
        total = f"{p[total_key]:.2%}" if total_key else ""
        panels.append(
            f'<div class="sm" data-path="{label}"><div class="sm-h"><span>{PATH_SHORT.get(label, label)}</span>'
            f'<b>{total}</b></div>{rows}</div>')
    return f'<div class="sm-grid">{"".join(panels)}</div>', scale


def zscore_section_html(risk):
    """Active factor exposure in cross-sectional z-scores, one diverging-bar
    panel per path, fixed +/-2.5 scale (not data-max) so panels are readable
    on the units z-scores are actually interpreted in."""
    paths = risk["paths"]
    factors = risk["factors"]
    scale = 2.5
    fmt = lambda v: f"{v:+.2f}&sigma;"
    panels = []
    for label, p in paths.items():
        rows = "".join(
            diverging_row(FACTOR_LABELS.get(f, f), p["z_active"][f], scale, fmt, +1,
                          f"{f} active exposure: {fmt(p['z_active'][f])} vs. holdings universe")
            for f in factors
        )
        panels.append(f'<div class="sm" data-path="{label}"><div class="sm-h"><span>{PATH_SHORT.get(label, label)}</span></div>{rows}</div>')
    return f'<div class="sm-grid">{"".join(panels)}</div>'


def class_label(name, max_len=16):
    """Asset-class names run long ('Single Stock (Growth)'); the diverging-row
    label column is narrow and already ellipsis-truncates overflow via CSS, but
    a short prefix keeps the un-hovered label legible without relying on that."""
    return name if len(name) <= max_len else name[:max_len - 1] + "…"


def te_summary_table_html(risk):
    """Factor-only TE (what the optimizer's own objective sees) vs. total TE
    (factor + idiosyncratic) side by side, so the gap between them -- which
    can be large while MSTR is still a big position -- is the first thing
    read, not something to notice by subtracting two numbers on your own."""
    rows = []
    for label, p in risk["paths"].items():
        te, idio, tot = p["te"], p["te_idio"], p["te_total"]
        idio_share = (idio ** 2) / (tot ** 2) if tot > 0 else 0.0
        rows.append(
            f'<tr data-path="{label}"><td style="font-weight:700">{PATH_SHORT.get(label, label)}</td>'
            f'<td>{te:.2%}</td><td>{idio:.2%}</td>'
            f'<td style="font-weight:700">{tot:.2%}</td>'
            f'<td>{idio_share:.0%}</td></tr>')
    return (
        '<div class="tbl-wrap"><table><thead><tr>'
        '<th>Path</th><th>TE, factor-only</th><th>TE, idiosyncratic</th>'
        '<th>TE, total</th><th>% of variance idiosyncratic</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


def idio_te_class_section_html(risk):
    """Idiosyncratic (non-factor) TE contribution by asset class, one panel
    per path. Shares a scale across panels, and deliberately does NOT scale
    to the max of only the nonzero classes -- MSTR's bar dwarfing every
    diversified sleeve to a sliver IS the finding, not a rendering problem."""
    paths = risk["paths"]
    classes = risk["asset_classes"]
    fmt = lambda v: f"{v*100:+.2f}%"
    vals = [abs(p["te_idio_contrib_by_class"].get(c, 0.0)) for p in paths.values() for c in classes]
    scale = max(vals) if vals else 1.0
    panels = []
    for label, p in paths.items():
        rows = "".join(
            diverging_row(class_label(c), p["te_idio_contrib_by_class"].get(c, 0.0), scale, fmt, +1,
                          f"{c}: {fmt(p['te_idio_contrib_by_class'].get(c, 0.0))} idiosyncratic TE contribution")
            for c in classes if abs(p["te_idio_contrib_by_class"].get(c, 0.0)) > 1e-6
        )
        panels.append(
            f'<div class="sm" data-path="{label}"><div class="sm-h"><span>{PATH_SHORT.get(label, label)}</span>'
            f'<b>{p["te_idio"]:.2%}</b></div>{rows}</div>')
    return f'<div class="sm-grid">{"".join(panels)}</div>'


def class_weight_section_html(risk):
    """Asset-class weight CHANGE (proposed - current), one panel per path.
    "before" is skipped -- a path vs. itself is always zero and not useful."""
    paths = risk["paths"]
    classes = risk["asset_classes"]
    before = paths.get("before", {}).get("class_weight", {})
    fmt = lambda v: f"{v*100:+.1f}pp"
    shown = {l: p for l, p in paths.items() if l != "before"}
    vals = [abs(p["class_weight"].get(c, 0.0) - before.get(c, 0.0)) for p in shown.values() for c in classes]
    scale = max(vals) if vals else 1.0
    panels = []
    for label, p in shown.items():
        rows = "".join(
            diverging_row(class_label(c), p["class_weight"].get(c, 0.0) - before.get(c, 0.0), scale, fmt, +1,
                          f"{c}: {before.get(c, 0.0):.1%} &rarr; {p['class_weight'].get(c, 0.0):.1%}")
            for c in classes
        )
        panels.append(f'<div class="sm" data-path="{label}"><div class="sm-h"><span>{PATH_SHORT.get(label, label)}</span></div>{rows}</div>')
    return f'<div class="sm-grid">{"".join(panels)}</div>'


def path_filter_html(container_id, labels):
    """Toggle-button row (reusing the .tbtn/.toggle-row styling already used
    by the Trade Detail tab's path selector) that lets the viewer isolate one
    path's panels within a pane -- every generated panel/row in that pane
    carries a matching data-path attribute; see wirePathFilter() in the
    inline script."""
    buttons = ['<button class="tbtn on" data-path="all">All paths</button>']
    buttons += [f'<button class="tbtn" data-path="{l}">{PATH_SHORT.get(l, l)}</button>' for l in labels]
    return f'<div class="toggle-row" id="{container_id}">{"".join(buttons)}</div>'


def round_for_json(df):
    """Round dollar columns to cents, but keep weight/pct columns at full
    precision -- they are stored as fractions, so a blanket df.round(2)
    collapses 6.36% to 6.00% and every sub-1% IRA weight to 0.00%."""
    out = df.copy()
    for col in out.select_dtypes("number").columns:
        digits = 6 if ("Weight" in col or "Pct" in col) else 2
        out[col] = out[col].round(digits)
    return out


def build_trade_rows_js(trades_by_label):
    obj = {}
    for label, df in trades_by_label.items():
        if df is None:
            obj[label] = []
        else:
            obj[label] = round_for_json(df).to_dict(orient="records")
    return json.dumps(obj)


if __name__ == "__main__":
    holdings = pd.read_csv(os.path.join(DATA_DIR, "holdings_household_clean.csv"))
    bench_weights = pd.read_csv(os.path.join(DATA_DIR, "benchmark_mvo_weights.csv"), index_col=0)["Weight"]
    bench_mix = benchmark_mix_str(bench_weights)
    saa_weights = load_saa_policy_weights()
    benchmark_tables = (
        benchmark_allocation_table(bench_weights, "Factor benchmark · constrained max-Sharpe")
        + benchmark_allocation_table(saa_weights, "SAA benchmark · Excel policy target")
    )
    metrics_doc = load_metrics()
    paths = metrics_doc["paths"]
    total = metrics_doc["total_household_value"]
    for p in paths.values():
        p["_total"] = total

    trades_by_label = {label: load_trades(label) for label in ALL_PATH_LABELS if label in paths}

    # SAA-benchmark TE for every path (including "before"), re-scoring each
    # path's already-optimized post-trade weights against the ONE SAA
    # sleeve/policy benchmark instead of each path's own native benchmark --
    # see the module docstring's "Also computes" section for why this exists
    # and PATH_INFO step 6 in the Methodology tab for the page-facing
    # explanation. Calibrated once from the pre-trade book, same convention
    # optimize_saa_mvo.py itself uses.
    current_ticker_weight = (holdings["Market_Value"] / total).groupby(holdings["Ticker"]).sum()
    saa_te_inputs = load_saa_te_inputs(current_ticker_weight)
    paths["before"]["saa_te"], _ = saa_te_of(
        (holdings["Market_Value"] / total).to_numpy(), holdings["Ticker"].to_numpy(),
        saa_te_inputs["sleeves"], saa_te_inputs["target_vec"],
        saa_te_inputs["covariance_mat"], saa_te_inputs["tau2"])
    for label, trades in trades_by_label.items():
        if trades is None:
            continue
        real_rows = trades[trades["Ticker"] != "CASH_RESERVE"]
        paths[label]["saa_te"], _ = saa_te_of(
            real_rows["Proposed_Weight"].to_numpy(), real_rows["Ticker"].to_numpy(),
            saa_te_inputs["sleeves"], saa_te_inputs["target_vec"],
            saa_te_inputs["covariance_mat"], saa_te_inputs["tau2"])

    te_before, cvar_before = paths["before"]["te"], paths["before"]["cvar"]
    saa_te_before = paths["before"]["saa_te"]
    snow_row = holdings[holdings["Ticker"] == "MSTR"].iloc[0]
    mstr_w = snow_row["Market_Value"] / total
    mstr_gain_pct = snow_row["Pct_Gain"]
    joint_total = holdings.loc[holdings["Is_Taxable"], "Market_Value"].sum()
    ira_total = total - joint_total

    shown_labels = [label for label in ALL_PATH_LABELS if label in paths]
    path_cards = "\n".join(
        path_card_html(label, paths[label], cap=paths[label].get("sell_cap"))
        for label in shown_labels
    )
    n_words = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}
    is_plural = len(shown_labels) != 1
    paths_heading = f"{n_words.get(len(shown_labels), len(shown_labels))} Path{'s' if is_plural else ''}"
    banner_head = "&#9888; MSTR's concentration keeps Active CVaR well past the historical -15% comfort level"
    banner_tail = (
        f"{paths_heading} follow, none of them CVaR-constrained (Active CVaR is shown for context on "
        "each, not enforced): Path C and Path D solve the same tax-budget-only problem and differ only "
        "in how much of the current MSTR position may be sold this year; Path SAA is an independent "
        "solve against the all-equity policy target with its own tax budget and MSTR cap; Path SAA 3% "
        "is the same SAA solve with the tax budget raised to 3% and MSTR's own tax no longer carved out "
        "of it.")

    holdings_json = json.dumps(round_for_json(holdings).to_dict(orient="records"))
    trades_json = build_trade_rows_js(trades_by_label)
    # "before" (the do-nothing baseline) genuinely belongs on this chart --
    # the title says "Current / ..." so the bars should actually include it.
    metric_labels = ["before"] + shown_labels
    paths_side_by_side = " / ".join(PATH_SHORT.get(l, l) for l in metric_labels)
    # Tax-cost bar uses the path's TOTAL realized tax (tax_cost_total when a
    # path has one -- C/D/SAA, whose MSTR-carve-out means their plain
    # "tax_cost" is only the budget-counted ex-MSTR slice, near-zero for SAA
    # -- falling back to "tax_cost" for paths with no carve-out at all). This
    # is "what the household actually pays," not "what counts against a
    # particular budget," so every path's bar reflects its full tax bill.
    metric_bars_js = ",\n        ".join(
        f'{{label:"{PATH_SHORT.get(l, l)}", data:[{paths[l]["te"]*100 if paths[l].get("feasible") else 0}, '
        f'{abs(paths[l]["cvar"])*100 if paths[l].get("feasible") else 0}, '
        f'{paths[l].get("tax_cost_total", paths[l].get("tax_cost", 0))/total*100 if paths[l].get("feasible") else 0}], '
        f'backgroundColor: PATHCOLORS["{l}"], borderRadius:4}}'
        for l in metric_labels
    )

    risk_path = os.path.join(DATA_DIR, "risk_attribution.json")
    if os.path.exists(risk_path):
        with open(risk_path) as f:
            risk = json.load(f)
        te_summary = te_summary_table_html(risk)
        te_grid, te_scale = risk_section_html(
            risk, "te_contrib", "te", lambda v: f"{v*100:+.2f}%", +1)
        cv_grid, cv_scale = risk_section_html(
            risk, "cvar_contrib", "cvar", lambda v: f"{v*100:+.2f}%", -1,
            extra_rows=[("cvar_residual", "Idiosync.")])
        z_grid = zscore_section_html(risk)
        risk_path_filter = path_filter_html("riskPathFilter", list(risk["paths"].keys()))
        risk_tab_button = '<div class="tab" data-tab="risk">Risk Attribution</div>'
        risk_pane = f'''<div id="pane-risk" class="pane">
    <div class="card">
      <div class="card-h">Deep Dive &mdash; Filter to One Path</div>
      <div class="card-sub">Every table row and panel below is tagged by path. Pick one to isolate it across the
        factor-only/total TE table, the TE and CVaR contribution charts, and the z-score exposure chart on this
        tab &mdash; or leave "All paths" selected to compare them side by side, as before.</div>
      {risk_path_filter}
    </div>
    <div class="card">
      <div class="card-h">Factor-Only vs. Total Tracking Error</div>
      <div class="card-sub">The optimizer's own objective (Methodology, step 1) minimizes the FACTOR-ONLY column
        &mdash; it has no visibility into single-name risk. The "idiosyncratic" column is each ticker's own
        HAC-OLS residual variance (what its FF5 beta does NOT explain), rolled up across the book. While MSTR is
        still a large position, factor-only TE understates the household's actual tracking error substantially.</div>
      {te_summary}
    </div>
    <div class="legend-row">
      <span><i class="legend-sw" style="background:var(--red)"></i>Adds to risk / overweight</span>
      <span><i class="legend-sw" style="background:var(--accent)"></i>Hedges / underweight</span>
      <span class="note" style="margin:0">Scale is shared across panels within each chart so paths are directly comparable.</span>
    </div>
    <div class="card">
      <div class="card-h">Tracking-Error Contribution by Factor</div>
      <div class="card-sub">Euler decomposition &mdash; each factor's exact share of FACTOR-ONLY TE (bars sum to
        the panel header's TE, the "factor-only" column above, not total TE). Negative bars are factors whose
        active exposure currently offsets the rest of the book. See the Asset Classes tab for the idiosyncratic
        piece this chart excludes.</div>
      {te_grid}
    </div>
    <div class="card">
      <div class="card-h">Active CVaR Contribution by Factor</div>
      <div class="card-sub">Each factor's mean realized return over the SAME worst-5% scenario tail the optimizer's
        CVaR constraint uses, times this path's active exposure to it. "Idiosync." is the part of the tail loss the
        5 factors don't explain &mdash; overwhelmingly MSTR's single-name risk while it's still a large position.</div>
      {cv_grid}
    </div>
    <div class="card">
      <div class="card-h">Active Factor Exposure, Z-Scores</div>
      <div class="card-sub">Each factor's beta standardized across this household's own 16-ticker holdings universe
        (not a broad market cross-section), portfolio minus benchmark. &plusmn;1&sigma; means "tilted one
        cross-sectional standard deviation, relative to what this family already owns."</div>
      {z_grid}
    </div>
  </div>'''

        idio_grid = idio_te_class_section_html(risk)
        weight_chg_grid = class_weight_section_html(risk)
        class_alloc_labels = risk["asset_classes"]
        class_alloc_weights = [risk["paths"]["before"]["class_weight"].get(c, 0.0) for c in class_alloc_labels]
        class_alloc_js = json.dumps({"labels": class_alloc_labels, "weights": [w * 100 for w in class_alloc_weights]})
        classes_path_filter = path_filter_html("classPathFilter", list(risk["paths"].keys()))
        classes_tab_button = '<div class="tab" data-tab="classes">Asset Classes</div>'
        classes_pane = f'''<div id="pane-classes" class="pane">
    <div class="card">
      <div class="card-h">Deep Dive &mdash; Filter to One Path</div>
      <div class="card-sub">Isolate one path's panel in the idiosyncratic-TE and weight-change charts below.
        "Current Allocation" above isn't path-specific (it's the household's allocation today, before any trade)
        so this filter doesn't touch it. The weight-change chart has no "Current" panel by design &mdash; a path
        compared against itself is always zero &mdash; so selecting "Current" there shows an empty grid.</div>
      {classes_path_filter}
    </div>
    <div class="card">
      <div class="card-h">Current Allocation by Asset Class</div>
      <div class="card-sub">Household weight (Joint + IRA combined), rolled up from the 16-ticker holdings
        universe by the fixed labels in <code>ASSET_CLASS</code> (<code>optimize_tax_cvar_mvo.py</code>).</div>
      <div class="chart-wrap" style="height:300px"><canvas id="classAllocChart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-h">Idiosyncratic TE Contribution by Asset Class</div>
      <div class="card-sub">The single-name risk piece the factor-only chart on the Risk Attribution tab leaves
        out, rolled up by class instead of by ticker. Panel header is total idiosyncratic TE for that path
        (<code>sqrt(sum of class contributions)</code>). MSTR dwarfing every diversified sleeve IS the finding,
        not a chart-scaling issue &mdash; the y-axis is shared across panels on purpose.</div>
      {idio_grid}
    </div>
    <div class="card">
      <div class="card-h">Asset-Class Weight Change vs. Current</div>
      <div class="card-sub">Proposed minus current household weight, in percentage points, by class. Hover a bar
        for the exact current &rarr; proposed weight.</div>
      <div class="legend-row" style="margin-bottom:10px">
        <span><i class="legend-sw" style="background:var(--red)"></i>Increases</span>
        <span><i class="legend-sw" style="background:var(--accent)"></i>Decreases</span>
      </div>
      {weight_chg_grid}
    </div>
  </div>'''
    else:
        risk_tab_button = ""
        risk_pane = ""
        classes_tab_button = ""
        classes_pane = ""
        class_alloc_js = "null"

    html = r'''<meta charset="utf-8">
<title>Rebalance Options</title>
<meta name="description" content="Tax- and CVaR-constrained trade proposals for the Joint + IRA portfolio, generated from optimize_tax_cvar_mvo.py's saved outputs.">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
:root{
  --bg:#f5f6fa; --card:#ffffff; --border:#e1e5ef; --border2:#ccd3e2;
  --text:#1b2233; --muted:#68708a; --dim:#98a0b6;
  --accent:#2554e8; --accent-soft:#2554e814; --green:#0c8f6f; --red:#d93a3a; --amber:#c67c11; --purple:#7654d6; --teal:#0a8fa8; --pink:#c2298a;
  --mono:'IBM Plex Mono',monospace; --sans:'IBM Plex Sans',system-ui,sans-serif;
  --shadow:0 1px 2px rgba(20,30,60,.05),0 2px 8px rgba(20,30,60,.05);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#0f131c; --card:#161b27; --border:#262e42; --border2:#333d57;
    --text:#e7eaf3; --muted:#8b93ab; --dim:#5d6580;
    --accent:#5b86ff; --accent-soft:#5b86ff1c; --green:#28b98d; --red:#f0605d; --amber:#e0a53c; --purple:#a68bef; --teal:#3fb8d1; --pink:#e560ae;
    --shadow:0 1px 2px rgba(0,0,0,.3),0 2px 10px rgba(0,0,0,.25);
  }
}
:root[data-theme="dark"]{
  --bg:#0f131c; --card:#161b27; --border:#262e42; --border2:#333d57;
  --text:#e7eaf3; --muted:#8b93ab; --dim:#5d6580;
  --accent:#5b86ff; --accent-soft:#5b86ff1c; --green:#28b98d; --red:#f0605d; --amber:#e0a53c; --purple:#a68bef; --teal:#3fb8d1; --pink:#e560ae;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 2px 10px rgba(0,0,0,.25);
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--sans);line-height:1.5;padding-inline:0}
.head{background:var(--card);border-bottom:1px solid var(--border);padding:14px 20px;position:sticky;top:env(safe-area-inset-top,0px);z-index:50;display:flex;flex-wrap:wrap;gap:14px 28px;align-items:center;box-shadow:var(--shadow)}
.head-id{font-family:var(--mono);font-size:12px;font-weight:600;color:var(--text);padding-right:20px;border-right:1px solid var(--border)}
.head-id span{display:block;font-size:9px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;font-weight:500;margin-bottom:2px}
.hm{font-family:var(--mono);white-space:nowrap}
.hm b{display:block;font-size:15px;font-weight:600;letter-spacing:-.01em}
.hm i{display:block;font-style:normal;font-size:8.5px;color:var(--muted);letter-spacing:.09em;text-transform:uppercase;margin-bottom:1px}
.wrap{max-width:1280px;margin:0 auto;padding:20px 20px 48px}
.tabs{display:flex;gap:4px;margin-bottom:18px;border-bottom:1px solid var(--border);overflow-x:auto}
.tab{font-family:var(--mono);font-size:11.5px;font-weight:600;color:var(--muted);padding:10px 16px;cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap;user-select:none}
.tab:hover{color:var(--text)}
.tab.on{color:var(--accent);border-color:var(--accent)}
.pane{display:none}.pane.on{display:block}
.banner{background:linear-gradient(135deg,#fef4e8,#fff);border:1px solid var(--amber)55;border-radius:10px;padding:16px 18px;margin-bottom:16px;font-size:12.5px;line-height:1.65}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .banner{background:linear-gradient(135deg,#2a2210,#161b27);}}
:root[data-theme="dark"] .banner{background:linear-gradient(135deg,#2a2210,#161b27);}
.banner b{color:var(--amber)}
.banner-head{font-family:var(--mono);font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--amber);margin-bottom:6px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:16px}
.stat{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:12px 14px;box-shadow:var(--shadow)}
.stat-l{font-family:var(--mono);font-size:8.5px;color:var(--muted);letter-spacing:.09em;text-transform:uppercase;margin-bottom:5px}
.stat-v{font-family:var(--mono);font-size:19px;font-weight:600;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.stat-s{font-size:10px;color:var(--muted);margin-top:3px;font-family:var(--mono)}
.g{color:var(--green)}.r{color:var(--red)}.a{color:var(--amber)}.ac{color:var(--accent)}.p{color:var(--purple)}.t{color:var(--teal)}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px 18px;box-shadow:var(--shadow);margin-bottom:16px}
.card-h{font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:.09em;text-transform:uppercase;margin-bottom:12px;display:flex;align-items:center;gap:8px}
.card-h::after{content:'';flex:1;height:1px;background:var(--border)}
.card-sub{font-size:11.5px;color:var(--muted);margin-top:-6px;margin-bottom:12px}
.path-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin-bottom:16px}
@media (max-width:820px){.path-grid{grid-template-columns:1fr}}
.path{border-radius:12px;padding:16px 18px;border:1px solid var(--border);background:var(--card);box-shadow:var(--shadow);position:relative;overflow:hidden;display:flex;flex-direction:column}
.path::before{content:'';position:absolute;top:0;left:0;right:0;height:3px}
.path.pc::before{background:var(--green)}.path.pd::before{background:var(--purple)}.path.psaa::before{background:var(--pink)}.path.psaa3::before{background:var(--teal)}
.path.infeasible{opacity:.75}
.path.infeasible::before{background:var(--red)}
.infeasible-badge{font-family:var(--mono);font-size:10px;font-weight:700;color:var(--red);background:var(--red)14;border-radius:8px;padding:10px 12px;margin-top:auto}
.path-title{font-family:var(--mono);font-weight:700;font-size:12px;margin-bottom:6px;letter-spacing:.02em}
.path-sub{font-size:11px;color:var(--muted);margin-bottom:12px;line-height:1.55;flex:1}
.path-mets{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:6px}
@media (max-width:420px){.path-mets{grid-template-columns:repeat(2,1fr)}}
.pm{background:var(--bg);border-radius:8px;padding:8px 6px;text-align:center}
.pm-l{font-family:var(--mono);font-size:7.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:3px}
.pm-v{font-family:var(--mono);font-size:13.5px;font-weight:700;font-variant-numeric:tabular-nums}
.pm-d{font-size:8.5px;color:var(--muted);font-family:var(--mono);margin-top:1px}
.tag{display:inline-block;font-family:var(--mono);font-size:8px;font-weight:700;padding:2px 6px;border-radius:4px;letter-spacing:.03em}
.tag-buy{background:var(--green)18;color:var(--green)}.tag-sell{background:var(--red)18;color:var(--red)}.tag-hold{background:var(--dim)18;color:var(--muted)}
.tbl-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:11px}
thead tr{background:var(--bg)}
th{padding:7px 10px;text-align:left;font-size:8px;font-weight:700;color:var(--muted);letter-spacing:.07em;text-transform:uppercase;border-bottom:2px solid var(--border2);white-space:nowrap}
td{padding:6px 10px;border-bottom:1px solid var(--border);white-space:nowrap;font-variant-numeric:tabular-nums}
tbody tr:hover td{background:var(--accent-soft)}
.chart-wrap{position:relative}
.toggle-row{display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap}
.tbtn{font-family:var(--mono);font-size:10.5px;padding:6px 14px;border-radius:7px;border:1px solid var(--border2);background:var(--card);color:var(--muted);cursor:pointer;font-weight:600}
.tbtn.on{border-color:var(--accent);background:var(--accent-soft);color:var(--accent)}
.tbtn:disabled{opacity:.4;cursor:not-allowed}
.mono{font-family:var(--mono)}
.note{font-size:10.5px;color:var(--muted);line-height:1.6;margin-top:4px}
.method-step{display:flex;gap:12px;margin-bottom:16px}
.method-num{font-family:var(--mono);font-weight:700;color:var(--accent);font-size:13px;flex-shrink:0;width:22px}
.method-body h4{font-size:13px;margin-bottom:4px}
.method-body p{font-size:12px;color:var(--muted);line-height:1.65}
.method-body code{font-family:var(--mono);font-size:11px;background:var(--bg);padding:1px 5px;border-radius:4px;border:1px solid var(--border)}
.sm-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}
.sm{background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.sm-h{display:flex;justify-content:space-between;align-items:baseline;font-family:var(--mono);font-size:10.5px;font-weight:700;margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid var(--border2)}
.sm-h b{font-size:12.5px;font-variant-numeric:tabular-nums}
.rb{display:grid;grid-template-columns:44px 1fr 58px;align-items:center;gap:8px;margin-bottom:5px;font-family:var(--mono);font-size:9.5px}
.rb-lab{color:var(--muted);text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rb-track{position:relative;height:12px;background:var(--card);border-radius:3px}
.rb-zero{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--border2)}
.rb-bar{position:absolute;top:1px;bottom:1px;min-width:1px}
.rb-bar.add{background:var(--red)}
.rb-bar.hedge{background:var(--accent)}
.rb-val{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
.rb-val.add{color:var(--red)}
.rb-val.hedge{color:var(--accent)}
.legend-row{display:flex;gap:18px;align-items:center;margin-bottom:14px;font-size:11px;color:var(--muted);flex-wrap:wrap}
.legend-sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.bench-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:18px;margin-top:14px}
.bench-title{font-weight:700;margin-bottom:7px}
.bench-table{margin:0;font-size:12px}
.bench-table td:last-child,.bench-table th:last-child{text-align:right;font-variant-numeric:tabular-nums}
.bench-total td{font-weight:700;border-top:2px solid var(--border2)}
::-webkit-scrollbar{width:7px;height:7px}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:4px}
</style>

<header class="head">
  <div class="head-id"><span>Household Portfolio</span>Account Holder A &amp; Account Holder B &middot; Joint + IRA</div>
  <div class="hm"><i>Total AUM</i><b class="ac">$__TOTAL_FMT__</b></div>
  <div class="hm"><i>MSTR Weight</i><b class="r">__MSTR_W__%</b></div>
  <div class="hm"><i>TE (current)</i><b class="a">__TE_BEFORE__%</b></div>
  <div class="hm"><i>Active CVaR 5% (current)</i><b class="r">__CVAR_BEFORE__%</b></div>
  <div class="hm"><i>Tax Budget</i><b class="t">2.0%/yr</b></div>
</header>

<div class="wrap">
  <div class="tabs">
    <div class="tab on" data-tab="overview">Overview</div>
    <div class="tab" data-tab="compare">Path Comparison</div>
    <div class="tab" data-tab="trades">Trade Detail</div>
    __RISK_TAB_BUTTON__
    __CLASSES_TAB_BUTTON__
    <div class="tab" data-tab="method">Methodology &amp; Data</div>
  </div>

  <div id="pane-overview" class="pane on">
    <div class="banner">
      <div class="banner-head">__BANNER_HEAD__</div>
      Active CVaR (5th-percentile annual shortfall vs. the policy benchmark) is currently
      <b>__CVAR_BEFORE__%</b>, past the <b>-15%</b> reference level (no longer an enforced constraint on any
      path below) &mdash; driven almost entirely by MSTR's __MSTR_W__% weight and __MSTR_GAIN__% embedded gain. __BANNER_TAIL__
    </div>

    <div class="grid">
      <div class="stat"><div class="stat-l">Household Value</div><div class="stat-v ac">$__TOTAL_FMT__</div>
        <div class="stat-s">Joint: $__JOINT_FMT__ &middot; IRA: $__IRA_FMT__</div></div>
      <div class="stat"><div class="stat-l">Tracking Error (current)</div><div class="stat-v a">__TE_BEFORE__%</div>
        <div class="stat-s">vs. __BENCH_MIX__ benchmark</div></div>
      <div class="stat"><div class="stat-l">TE vs. SAA Benchmark (current)</div><div class="stat-v a">__SAA_TE_BEFORE__%</div>
        <div class="stat-s">comparable across every path below &mdash; see Methodology step 6</div></div>
      <div class="stat"><div class="stat-l">Active CVaR 5% (current)</div><div class="stat-v r">__CVAR_BEFORE__%</div>
        <div class="stat-s">Reference: -15.0% &middot; informational, not enforced</div></div>
      <div class="stat"><div class="stat-l">MSTR Weight</div><div class="stat-v r">__MSTR_W__%</div>
        <div class="stat-s">of household value</div></div>
      <div class="stat"><div class="stat-l"># Positions</div><div class="stat-v t">__N_POS__</div>
        <div class="stat-s">__N_TICKERS__ unique tickers, 2 accounts</div></div>
    </div>

    <div class="card">
      <div class="card-h">Current Allocation by Weight</div>
      <div class="card-sub">Household weight (Joint + IRA combined)</div>
      <div class="chart-wrap" style="height:340px"><canvas id="allocChart"></canvas></div>
    </div>
  </div>

  <div id="pane-compare" class="pane">
    <div class="card-h" style="margin-bottom:14px">__PATHS_HEADING__</div>
    <div class="path-grid">
      __PATH_CARDS__
    </div>
    <div class="card">
      <div class="card-h">__PATHS_SIDE_BY_SIDE__ &mdash; Side by Side</div>
      <div class="chart-wrap" style="height:300px"><canvas id="metricChart"></canvas></div>
    </div>
  </div>

  <div id="pane-trades" class="pane">
    <div class="toggle-row" id="pathToggle"></div>
    <div class="card" style="padding:0">
      <div class="tbl-wrap">
        <table>
          <thead><tr><th>Account</th><th>Ticker</th><th>Description</th><th>Dir</th><th>Current Wt</th><th>Proposed Wt</th><th>Trade $</th><th>Realized Gain</th><th>Tax Cost</th></tr></thead>
          <tbody id="tradeBody"></tbody>
        </table>
      </div>
    </div>
  </div>

  __RISK_PANE__

  __CLASSES_PANE__

  <div id="pane-method" class="pane">
    <div class="card">
      <div class="card-h">Benchmark Allocations</div>
      <div class="card-sub">These are the policy benchmarks used by the dashboard. The factor allocation is
        created by constrained max-Sharpe optimization; the SAA allocation is read directly from
        <code>all_equity_saa_covariance.xlsx</code> and is not optimized at benchmark creation.</div>
      <div class="bench-grid">__BENCHMARK_TABLES__</div>
    </div>
    <div class="card">
      <div class="card-h">What This Model Actually Solves</div>
      <div class="method-step"><div class="method-num">1</div><div class="method-body">
        <h4>Objective &mdash; minimize tracking error, net of turnover</h4>
        <p>Minimum-tracking-error mean-variance optimization against the repo's existing policy benchmark
        (__BENCH_MIX__ Fama-French factors -- a max-Sharpe long-only mix with a floor under every factor so
        none get zeroed by a noisy sample mean, see <code>construct_benchmark_mvo.py</code>), plus a
        per-position L1 turnover penalty
        <code>&lambda; &middot; &Sigma;<sub>i</sub> |x<sub>i</sub> &minus; x0<sub>i</sub>|</code> charged on each
        (ticker, account) row separately. No return forecast is used.</p>
      </div></div>
      <div class="method-step"><div class="method-num">2</div><div class="method-body">
        <h4>Active CVaR &mdash; computed, not a constraint</h4>
        <p>5th-percentile Active CVaR is still estimated for every path from overlapping 12-month windows over
        the common price history of all holdings, and shown for context (Risk Attribution tab, path cards) &mdash;
        but no path below is required to satisfy any floor on it. An earlier version of this model enforced a
        -15% CVaR floor; that constraint has been removed from Paths C, D, and SAA.</p>
      </div></div>
      <div class="method-step"><div class="method-num">3</div><div class="method-body">
        <h4>Constraint &mdash; annual tax budget</h4>
        <p>Only net sales in the Joint (taxable) account cost anything. Taxed at a blended __TAXLT__% long-term
        rate and an assumed __TAXST__% ordinary+NIIT rate on any short-term portion. Capped at 2% of household
        value per year.</p>
      </div></div>
      <div class="method-step"><div class="method-num">4</div><div class="method-body">
        <h4>Constraint &mdash; minimum per-ticker allocation</h4>
        <p>Every currently-held ticker keeps at least __MIN_TICKER_WEIGHT__% of household value, summed across
        accounts if held in both. A structural floor, not a tax or risk rule &mdash; its only job is to stop the
        optimizer from fully zeroing out a small sleeve as an incidental side-effect of chasing tracking error or
        the tax budget. Applies to every path below, EXCEPT tickers in no-add status (next step).</p>
      </div></div>
      <div class="method-step"><div class="method-num">5</div><div class="method-body">
        <h4>Constraint &mdash; no-add tickers (SHV)</h4>
        <p>SHV may be trimmed or sold, but never bought further, in every path below and in both optimizer
        scripts &mdash; a plain per-row upper bound (<code>x &le; x0</code>) rather than a tax or risk rule.
        SHV's current combined weight (0.298%) sits just under the 0.3% allocation floor in step 4, so it is
        exempted from that floor entirely: requiring both "no more than its current weight" and "at least 0.3%"
        on the same ticker would be infeasible by construction.</p>
      </div></div>
      <div class="method-step"><div class="method-num">6</div><div class="method-body">
        <h4>Path C vs. Path D vs. Path SAA vs. Path SAA 3%</h4>
        <p>Path C and Path D solve the identical tax-budget-only problem (<code>optimize_tax_cvar_mvo.py</code>,
        no CVaR constraint): MSTR's realized tax is carved out of the 2% budget's accounting entirely (a separate
        taxable strategy for MSTR is assumed handled elsewhere), and each caps how much of MSTR's current
        position may be sold this year &mdash; 70% for C, 30% for D &mdash; modeling a faster or slower
        multi-year glide path instead of a one-shot exit. Path SAA is a fully independent optimizer
        (<code>optimize_saa_mvo.py</code>) targeting the all-equity sleeve policy instead of the FF5 benchmark,
        with its own 2% tax budget and a 50% MSTR sale cap, using the identical carve-out device. Path SAA 3%
        is the same SAA optimizer and objective, but with the tax budget raised to 3% AND the MSTR carve-out
        removed &mdash; MSTR's own realized tax now counts against that 3% budget like any other position, so a
        larger MSTR sale competes directly with every other trade for the same dollars, rather than being
        exempted from the constraint entirely. All four are solved as convex quadratic programs via
        <code>scipy.optimize.minimize(SLSQP)</code>.</p>
        <p><b>"Track. Error" vs. "TE vs. SAA Bench."</b> on every path card above are deliberately two different
        numbers, not a rounding artifact. "Track. Error" is each path's own NATIVE benchmark &mdash; the FF5
        factor max-Sharpe benchmark for Current/Path C/Path D, but a completely different 13-sleeve Excel policy
        target for Path SAA/SAA 3% &mdash; so that column is <b>not comparable path to path</b>: Path C's 9.00%
        and Path SAA's 4.46% are two different rulers, not two points on the same one. "TE vs. SAA Bench."
        re-scores every path's SAME already-optimized post-trade weights (no re-optimization, no trade changes)
        against the ONE SAA sleeve/policy benchmark, so that column IS directly comparable across every path
        &mdash; on it, Path C and Path SAA turn out to be much closer to each other than "Track. Error" implies.</p>
      </div></div>
      <div class="method-step"><div class="method-num">7</div><div class="method-body">
        <h4>Turnover penalty &mdash; &lambda; = __TURN_LAMBDA__</h4>
        <p>Without a trading penalty, minimum-tracking-error MVO is indifferent between a small trim spread across
        the book and a wholesale liquidation into whichever holding hedges best &mdash; the classic corner-solution
        failure mode. The L1 penalty in step 1 prices that: it leaves genuinely worthwhile trades near full size
        while zeroing out the long tail of cosmetic few-hundred-dollar trades. <code>|x &minus; x0|</code> is
        linearized with a two-sided slack per position, so the problem stays a convex QP. At
        <code>&lambda; = __TURN_LAMBDA__</code> this cuts the trade list by roughly a third and the realized tax
        bill along with it, for a few tens of basis points of tracking error. Note per-position size limits still
        aren't imposed beyond what was asked for &mdash; the penalty discourages concentration, it doesn't cap it.</p>
      </div></div>
      <div class="method-step"><div class="method-num">8</div><div class="method-body">
        <h4>Risk attribution &mdash; TE, CVaR and exposure z-scores by factor</h4>
        <p>The Risk Attribution tab (<code>portfolio_risk_attribution.py</code>) decomposes each path's headline
        numbers onto the 5 FF factors. Tracking error uses an exact Euler decomposition
        (<code>TE = &Sigma;<sub>k</sub> a<sub>k</sub>(&Sigma;a)<sub>k</sub> / TE</code>, which sums to TE exactly);
        CVaR is attributed by averaging each factor's realized return over the SAME worst-5% scenario tail used
        to compute it (informational, not a constraint on any path), leaving a residual for what the 5 factors
        don't explain (overwhelmingly MSTR's single-name risk). SAA's FF5-factor risk numbers are attribution
        of what its sleeve-covariance-optimized weights happen to expose the household to on the FF5 benchmark,
        not something its own objective targets. Exposure z-scores standardize each ticker's factor betas across
        this household's own
        16-ticker holdings universe, not a broad market cross-section &mdash; they read as "tilted relative to
        what this family already owns," which is the right question for a reweighting-only mandate.</p>
      </div></div>
      <div class="method-step"><div class="method-num">9</div><div class="method-body">
        <h4>Idiosyncratic tracking error and asset-class rollups</h4>
        <p>The factor-only TE above is exactly what the optimizer's objective sees, and it's a real
        simplification: every ticker's FF5 regression leaves a residual, and for a position the size of MSTR
        that residual dominates, not rounds away. A structural ("Barra-style") split adds it back without
        touching the optimizer itself: each ticker's annualized OLS residual variance, times the SQUARE of its
        total household weight (summed across accounts &mdash; the benchmark holds no individual stocks, so a
        ticker's total weight IS its idiosyncratic exposure), summed across tickers under a no-cross-correlation
        assumption, gives <code>Var<sub>idio</sub></code>; <code>TE<sub>total</sub> = &radic;(TE<sub>factor</sub>&sup2;
        + Var<sub>idio</sub>)</code>. Because factor and idiosyncratic variance are orthogonal blocks of one
        quadratic form, Euler's theorem still applies exactly to the combined total &mdash; verified as an
        assertion in the script, same as the factor-only split. Per-ticker idiosyncratic contributions roll up
        to the fixed asset-class labels in <code>ASSET_CLASS</code> by plain summation, which is exact (not an
        approximation) under that same no-cross-correlation assumption. The Active-CVaR residual on the Risk
        Attribution tab is NOT further split this way &mdash; that would need each ticker's own idiosyncratic
        return summed over the same tail scenarios, not just its residual variance, a separate piece of work
        this pass didn't add.</p>
      </div></div>
      <div class="method-step"><div class="method-num">10</div><div class="method-body">
        <h4>Data used</h4>
        <p>Holdings, cost basis, and ST/LT unrealized gains are read directly from the brokerage export. Monthly
        price history for all tickers is real, pulled via yfinance. This dashboard itself is generated purely
        from <code>optimize_tax_cvar_mvo.py</code>'s saved CSV/JSON outputs by
        <code>generate_rebalance_dashboard.py</code> &mdash; no numbers are computed here.</p>
      </div></div>
    </div>
  </div>
</div>

<script>
const HOLDINGS = __HOLDINGS_JSON__;
const TRADES = __TRADES_JSON__;
const CLASS_ALLOC = __CLASS_ALLOC_JSON__;
const C = {accent:'#2554e8',green:'#0c8f6f',red:'#d93a3a',amber:'#c67c11',purple:'#7654d6',teal:'#0a8fa8',pink:'#c2298a',muted:'#68708a',border:'#e1e5ef'};
const PATHCOLORS = {before:C.muted, C:C.green, D:C.purple, SAA:C.pink, SAA_3pct:C.teal};
function themeColor(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim()||C.muted;}
Chart.defaults.color = themeColor('--muted');
Chart.defaults.borderColor = themeColor('--border');
Chart.defaults.font.family = "'IBM Plex Mono', monospace";
Chart.defaults.font.size = 10;
function fmtUSD(x){return '$'+Math.round(x).toLocaleString();}

document.querySelectorAll('.tab').forEach(el=>{
  el.addEventListener('click',()=>{
    document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
    document.querySelectorAll('.pane').forEach(p=>p.classList.remove('on'));
    el.classList.add('on');
    document.getElementById('pane-'+el.dataset.tab).classList.add('on');
  });
});

function wirePathFilter(filterId, paneId){
  const container = document.getElementById(filterId);
  if(!container) return;
  const buttons = container.querySelectorAll('.tbtn');
  const targets = document.querySelectorAll('#'+paneId+' [data-path]');
  buttons.forEach(b=>{
    b.addEventListener('click', ()=>{
      buttons.forEach(x=>x.classList.remove('on'));
      b.classList.add('on');
      const sel = b.dataset.path;
      targets.forEach(el=>{
        el.style.display = (sel==='all' || el.dataset.path===sel) ? '' : 'none';
      });
    });
  });
}
wirePathFilter('riskPathFilter', 'pane-risk');
wirePathFilter('classPathFilter', 'pane-classes');

(function(){
  const sorted = [...HOLDINGS].sort((a,b)=>b.Weight_Household-a.Weight_Household);
  new Chart(document.getElementById('allocChart'), {
    type:'bar',
    data:{labels: sorted.map(h=>h.Ticker+' ('+h.Account.split(' ')[0]+')'),
      datasets:[{data: sorted.map(h=>h.Weight_Household*100),
        backgroundColor: sorted.map(h=>h.Ticker==='MSTR'?C.red:(h.Account.includes('IRA')?C.teal:C.accent)),
        borderRadius:3}]},
    options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>c.raw.toFixed(2)+'%'}}},
      scales:{x:{ticks:{callback:v=>v.toFixed(0)+'%'},grid:{color:themeColor('--border')}},
               y:{grid:{display:false},ticks:{autoSkip:false,font:{size:9}}}}}
  });
})();

if (CLASS_ALLOC) (function(){
  const el = document.getElementById('classAllocChart');
  if (!el) return;
  new Chart(el, {
    type:'bar',
    data:{labels: CLASS_ALLOC.labels,
      datasets:[{data: CLASS_ALLOC.weights,
        backgroundColor: CLASS_ALLOC.labels.map(l=>l.startsWith('Single Stock')?C.red:C.accent),
        borderRadius:3}]},
    options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>c.raw.toFixed(2)+'%'}}},
      scales:{x:{ticks:{callback:v=>v.toFixed(0)+'%'},grid:{color:themeColor('--border')}},
               y:{grid:{display:false},ticks:{autoSkip:false,font:{size:9}}}}}
  });
})();

(function(){
  new Chart(document.getElementById('metricChart'), {
    type:'bar',
    data:{labels:['Tracking Error','Active CVaR 5% (abs)','Tax Cost (% of AUM)'],
      datasets:[
        __METRIC_BARS__
      ]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{position:'top',labels:{boxWidth:10,padding:14}},
        tooltip:{callbacks:{label:c=>c.dataset.label+': '+c.raw.toFixed(2)+'%'}}},
      scales:{y:{ticks:{callback:v=>v.toFixed(0)+'%'},grid:{color:themeColor('--border')}},x:{grid:{display:false}}}}
  });
})();

const PATHLABELS = {before:'Current', joint:'Joint solve', A:'Path A', B:'Path B', C:'Path C', D:'Path D', SAA:'SAA target', SAA_3pct:'SAA 3%'};
const AVAILABLE_PATHS = Object.keys(TRADES).filter(l=>TRADES[l] && TRADES[l].length);
document.getElementById('pathToggle').innerHTML = AVAILABLE_PATHS.map((l,i)=>
  `<button class="tbtn${i===0?' on':''}" data-path="${l}">${PATHLABELS[l]||l} trades</button>`).join('');
function renderTrades(path){
  // Every position is shown, including HOLD rows (Trade_Dollars == 0) --
  // seeing what did NOT change is as much a part of reviewing a rebalance
  // as seeing what did.
  const rows = (TRADES[path] || []).slice().sort((a,b)=>a.Trade_Dollars-b.Trade_Dollars);
  document.getElementById('tradeBody').innerHTML = rows.map(r=>{
    const tagClass = r.Direction==='BUY'?'tag-buy':(r.Direction==='SELL'?'tag-sell':'tag-hold');
    return `<tr>
      <td>${r.Account}</td>
      <td style="font-weight:700;color:${r.Ticker==='MSTR'?C.red:C.accent}">${r.Ticker}</td>
      <td style="color:var(--muted);font-size:10px;white-space:normal;min-width:160px">${r.Description||'&mdash;'}</td>
      <td><span class="tag ${tagClass}">${r.Direction}</span></td>
      <td>${(r.Current_Weight*100).toFixed(2)}%</td>
      <td>${(r.Proposed_Weight*100).toFixed(2)}%</td>
      <td style="color:${r.Trade_Dollars<0?C.red:(r.Trade_Dollars>0?C.green:'inherit')}">${fmtUSD(r.Trade_Dollars)}</td>
      <td>${r.Realized_Gain_Dollars?fmtUSD(r.Realized_Gain_Dollars):'&mdash;'}</td>
      <td>${r.Tax_Cost_Dollars?fmtUSD(r.Tax_Cost_Dollars):'&mdash;'}</td>
    </tr>`;
  }).join('');
}
document.querySelectorAll('#pathToggle .tbtn').forEach(b=>{
  b.addEventListener('click',()=>{
    document.querySelectorAll('#pathToggle .tbtn').forEach(x=>x.classList.remove('on'));
    b.classList.add('on');
    renderTrades(b.dataset.path);
  });
});
if(AVAILABLE_PATHS.length) renderTrades(AVAILABLE_PATHS[0]);
</script>
'''

    html = html.replace("__TOTAL_FMT__", f"{total:,.0f}")
    html = html.replace("__JOINT_FMT__", f"{joint_total:,.0f}")
    html = html.replace("__IRA_FMT__", f"{ira_total:,.0f}")
    html = html.replace("__MSTR_W__", f"{mstr_w*100:.1f}")
    html = html.replace("__MSTR_GAIN__", f"{mstr_gain_pct:,.0f}")
    html = html.replace("__TE_BEFORE__", f"{te_before*100:.2f}")
    html = html.replace("__SAA_TE_BEFORE__", f"{saa_te_before*100:.2f}")
    html = html.replace("__CVAR_BEFORE__", f"{cvar_before*100:.1f}")
    html = html.replace("__N_POS__", str(len(holdings)))
    html = html.replace("__N_TICKERS__", str(holdings["Ticker"].nunique()))
    html = html.replace("__TAXLT__", f"{metrics_doc['tax_lt_rate']*100:.1f}")
    html = html.replace("__TAXST__", f"{metrics_doc['tax_st_rate']*100:.1f}")
    html = html.replace("__RISK_TAB_BUTTON__", risk_tab_button)
    html = html.replace("__RISK_PANE__", risk_pane)
    html = html.replace("__CLASSES_TAB_BUTTON__", classes_tab_button)
    html = html.replace("__CLASSES_PANE__", classes_pane)
    html = html.replace("__CLASS_ALLOC_JSON__", class_alloc_js)
    html = html.replace("__PATHS_HEADING__", paths_heading)
    html = html.replace("__PATHS_SIDE_BY_SIDE__", paths_side_by_side)
    html = html.replace("__BANNER_HEAD__", banner_head)
    html = html.replace("__BANNER_TAIL__", banner_tail)
    html = html.replace("__BENCH_MIX__", bench_mix)
    html = html.replace("__BENCHMARK_TABLES__", benchmark_tables)
    html = html.replace("__TURN_LAMBDA__", str(metrics_doc.get("turnover_lambda", 0.0)))
    html = html.replace("__MIN_TICKER_WEIGHT__", f"{metrics_doc.get('min_ticker_weight', 0.003)*100:.1f}")
    html = html.replace("__PATH_CARDS__", path_cards)
    html = html.replace("__HOLDINGS_JSON__", holdings_json)
    html = html.replace("__TRADES_JSON__", trades_json)
    html = html.replace("__METRIC_BARS__", metric_bars_js)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Wrote {OUT_PATH} ({len(html):,} bytes)")
    print(f"Paths included: {list(paths.keys())}")
    for label in ALL_PATH_LABELS:
        if label in paths:
            status = "feasible" if paths[label].get("feasible") else "INFEASIBLE"
            print(f"  Path {label}: {status}")
