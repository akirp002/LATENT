"""
generate_collar_html.py
=====================================================================
Runs the MSTR collar (see collar_cvar_analysis.py for the mechanic -- one
zero-cost collar per 12-month CVaR scenario window, put floor -10% / call
cap +20%, off MSTR's current implied price) at four COVERAGE levels for
every optimized trade path:

    0%   -- no collar (uncollared baseline, cross-checked against the
             rebalance dashboard's own Active CVaR numbers)
    30%  -- partial collar, 30% of the post-trade MSTR position covered
    50%  -- partial collar, 50% of the post-trade MSTR position covered
    100% -- full collar, the entire post-trade MSTR position covered

Coverage is modeled as a linear blend of MSTR's raw and collared scenario
returns (see blended_cvar_of's docstring in collar_cvar_analysis.py) --
mathematically identical to actually collaring `coverage` fraction of the
shares and leaving the rest naked.

Writes post_collar_trade_CVAR_Analytics.html in this folder: a single
self-contained page (Chart.js via CDN, same library/version the rebalance
dashboard already uses) with one grouped bar chart (CVaR by path x coverage)
and one detail table per path (CVaR, notional collared, contracts, and the
pp improvement over uncollared) at all four coverage levels.
"""

import json
import os

import numpy as np
import pandas as pd

from collar_cvar_analysis import (
    CALL_CAP,
    PATH_TRADE_FILES,
    PUT_FLOOR,
    blended_cvar_of,
    blended_scenario_te_of,
    load_all_data,
)
from optimize_tax_cvar_mvo import cvar_of
from saa_benchmark_te import load_saa_te_inputs, saa_te_of

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_HTML = os.path.join(OUT_DIR, "post_collar_trade_CVAR_Analytics.html")
OUT_CSV = os.path.join(OUT_DIR, "collar_scenarios_comparison.csv")

COVERAGE_LEVELS = [0.0, 0.30, 0.50, 1.00]
COVERAGE_LABELS = {0.0: "No Collar", 0.30: "30% Collar", 0.50: "50% Collar", 1.00: "Full 100% Collar"}
COVERAGE_COLORS = {0.0: "#8a8f98", 0.30: "#f2a341", 0.50: "#4c8bf5", 1.00: "#2fae62"}

PATH_SHORT = {"before": "Current", "C": "Path C", "D": "Path D", "SAA": "SAA target",
              "SAA_3pct": "SAA 3%"}


def build_results():
    data = load_all_data()
    path_metrics = data["path_metrics"]
    total_household_value = data["total_household_value"]
    snow_price = data["snow_price"]
    path_weights = data["path_weights"]
    A_scen = data["A_scen"]
    bench_annual_vec = data["bench_annual_vec"]

    put_strike = snow_price * (1 + PUT_FLOOR)
    call_strike = snow_price * (1 + CALL_CAP)

    labels_order = ["before"] + [l for l in PATH_TRADE_FILES if l in path_weights]

    # SAA-benchmark TE: re-scores every path's already-optimized post-trade
    # weights (no re-optimization) against the SAME sleeve/policy benchmark
    # optimize_saa_mvo.py itself uses, so Path C/D (normally measured against
    # the FF5 benchmark) become directly comparable to SAA/SAA_3pct. See
    # saa_benchmark_te.py for the full explanation and its own [OK] check
    # against optimize_saa_mvo.py's own reported number.
    saa_te_inputs = load_saa_te_inputs(path_weights)

    rows = []
    warnings = []
    TOL = 1e-6
    for label in labels_order:
        wdf = path_weights[label]
        tickers_row = wdf["Ticker"].values
        x = wdf["Weight"].values
        A_row_raw = A_scen.loc[tickers_row].values
        snow_weight = float(wdf.loc[wdf["Ticker"] == "MSTR", "Weight"].sum())
        snow_notional_full = snow_weight * total_household_value

        cvar_uncollared = cvar_of(x, A_row_raw, bench_annual_vec)
        dashboard_cvar = path_metrics["paths"].get(label, {}).get("cvar")
        if dashboard_cvar is not None and abs(cvar_uncollared - dashboard_cvar) > TOL:
            warnings.append(
                f"Path '{label}': uncollared CVaR ({cvar_uncollared:.4%}) does not match "
                f"path_metrics.json / dashboard ({dashboard_cvar:.4%})."
            )

        # Dashboard TE is NOT scenario-based -- it's the whole-distribution,
        # parametric factor-covariance + idio-residual-variance quadratic form
        # optimize_tax_cvar_mvo.py's te_of() already solved for and saved to
        # path_metrics.json. It never reads A_scen, so a MSTR collar (which
        # only edits A_scen) structurally cannot move it -- it's one fixed
        # number per path, read straight from path_metrics.json, not
        # recomputed here.
        te = path_metrics["paths"].get(label, {}).get("te")
        te_factor_only = path_metrics["paths"].get(label, {}).get("te_factor_only")

        # Same idea as the dashboard TE above -- one fixed number per path,
        # not a function of collar coverage (this formula never reads A_scen
        # either) -- but unlike the dashboard TE, this one uses the SAME
        # benchmark for every path, so it IS comparable across paths.
        saa_te, saa_te_sleeve_only = saa_te_of(
            x, tickers_row, saa_te_inputs["sleeves"], saa_te_inputs["target_vec"],
            saa_te_inputs["covariance_mat"], saa_te_inputs["tau2"])

        for coverage in COVERAGE_LEVELS:
            cvar_c = blended_cvar_of(x, tickers_row, A_scen, bench_annual_vec,
                                      PUT_FLOOR, CALL_CAP, coverage=coverage)
            # Empirical "scenario TE" -- std dev of the SAME 58 scenario active
            # returns CVaR is built from, with MSTR blended by this coverage.
            # Unlike the dashboard's parametric TE above, this DOES respond to
            # the collar -- it's the literal "std of active returns" a collar
            # mechanically shrinks by capping MSTR's scenario outcomes.
            scenario_te_c = blended_scenario_te_of(x, tickers_row, A_scen, bench_annual_vec,
                                                    PUT_FLOOR, CALL_CAP, coverage=coverage)
            covered_notional = coverage * snow_notional_full
            covered_shares = covered_notional / snow_price
            covered_contracts = covered_shares / 100.0
            rows.append({
                "path": label,
                "path_label": PATH_SHORT.get(label, label),
                "coverage": coverage,
                "coverage_label": COVERAGE_LABELS[coverage],
                "snow_weight": snow_weight,
                "snow_notional_full": snow_notional_full,
                "covered_notional": covered_notional,
                "covered_shares": covered_shares,
                "covered_contracts": covered_contracts,
                "cvar": cvar_c,
                "cvar_uncollared": cvar_uncollared,
                "improvement_pp": (cvar_c - cvar_uncollared) * 100,
                "te": te,
                "te_factor_only": te_factor_only,
                "saa_te": saa_te,
                "saa_te_sleeve_only": saa_te_sleeve_only,
                "scenario_te": scenario_te_c,
            })

    result = pd.DataFrame(rows)
    meta = {
        "snow_price": snow_price,
        "put_strike": put_strike,
        "call_strike": call_strike,
        "total_household_value": total_household_value,
        "labels_order": labels_order,
        "warnings": warnings,
    }
    return result, meta


def fmt_pct(v):
    return f"{v * 100:.2f}%"


def fmt_money(v):
    return f"${v:,.0f}"


def build_chart_datasets(result):
    datasets = []
    for coverage in COVERAGE_LEVELS:
        sub = result[result["coverage"] == coverage]
        data_by_path = {row["path"]: row["cvar"] * 100 for _, row in sub.iterrows()}
        datasets.append({
            "coverage": coverage,
            "label": COVERAGE_LABELS[coverage],
            "color": COVERAGE_COLORS[coverage],
            "data": data_by_path,
        })
    return datasets


def overview_table_html(result, path_order):
    rows_html = []
    for label in path_order:
        sub = result[result["path"] == label]
        te = sub.iloc[0]["te"]
        saa_te = sub.iloc[0]["saa_te"]
        scenario_te_none = sub.loc[sub["coverage"] == 0.0, "scenario_te"].iloc[0]
        scenario_te_full = sub.loc[sub["coverage"] == 1.0, "scenario_te"].iloc[0]
        cvar_none = sub.loc[sub["coverage"] == 0.0, "cvar"].iloc[0]
        cvar_full = sub.loc[sub["coverage"] == 1.0, "cvar"].iloc[0]
        improvement = sub.loc[sub["coverage"] == 1.0, "improvement_pp"].iloc[0]
        rows_html.append(f"""
        <tr>
          <td>{PATH_SHORT.get(label, label)}</td>
          <td class="num">{fmt_pct(te) if te is not None else '&mdash;'}</td>
          <td class="num">{fmt_pct(saa_te)}</td>
          <td class="num">{fmt_pct(scenario_te_none)}</td>
          <td class="num">{fmt_pct(scenario_te_full)}</td>
          <td class="num">{fmt_pct(cvar_none)}</td>
          <td class="num">{fmt_pct(cvar_full)}</td>
          <td class="num">{improvement:+.2f}pp</td>
        </tr>""")
    return f"""
    <div class="card">
      <div class="card-h">Overview &mdash; Tracking Error vs. Active CVaR</div>
      <div class="card-sub">Three different TE numbers, deliberately: <b>Tracking Error (dashboard)</b> is each
        path's OWN native benchmark &mdash; the FF5 max-Sharpe benchmark for Current/C/D, but a completely
        different 13-sleeve Excel policy benchmark for SAA/SAA 3% (<code>optimize_saa_mvo.py</code>'s
        <code>te_total()</code>) &mdash; so these numbers are <b>NOT comparable across paths</b>.
        <b>SAA-benchmark TE</b> fixes that: it re-scores every path's SAME already-optimized post-trade weights
        (no re-optimization, no path's trade changes) against the one SAA sleeve/policy benchmark, so this column
        IS directly comparable path to path (see <code>saa_benchmark_te.py</code>). <b>Scenario TE</b> is a third,
        separate thing again &mdash; the empirical std dev of the same 58 scenario active returns Active CVaR is
        built from, which is the only one of the three that responds to the MSTR collar (compare its "no collar"
        vs. "100% collar" columns).</div>
      <table>
        <thead><tr><th>Path</th><th>Tracking Error (dashboard, per-path benchmark)</th>
          <th>SAA-benchmark TE (comparable)</th>
          <th>Scenario TE (no collar)</th><th>Scenario TE (100% collar)</th>
          <th>Active CVaR (no collar)</th><th>Active CVaR (100% collar)</th><th>CVaR improvement</th></tr></thead>
        <tbody>{''.join(rows_html)}</tbody>
      </table>
    </div>"""


def path_section_html(label, sub, meta):
    sub = sub.sort_values("coverage")
    rows_html = []
    for _, r in sub.iterrows():
        rows_html.append(f"""
        <tr>
          <td>{r['coverage_label']}</td>
          <td class="num">{fmt_pct(r['cvar'])}</td>
          <td class="num">{r['improvement_pp']:+.2f}pp</td>
          <td class="num">{fmt_pct(r['scenario_te'])}</td>
          <td class="num">{fmt_money(r['covered_notional'])}</td>
          <td class="num">{r['covered_shares']:,.0f}</td>
          <td class="num">{r['covered_contracts']:,.1f}</td>
        </tr>""")
    snow_weight = sub.iloc[0]["snow_weight"]
    snow_notional_full = sub.iloc[0]["snow_notional_full"]
    te = sub.iloc[0]["te"]
    te_txt = f" &middot; Tracking Error (dashboard) <b>{fmt_pct(te)}</b> (unaffected by the collar)" if te is not None else ""
    return f"""
    <div class="card">
      <div class="card-h">{PATH_SHORT.get(label, label)}</div>
      <div class="card-sub">Post-trade MSTR weight <b>{fmt_pct(snow_weight)}</b>
        ({fmt_money(snow_notional_full)} notional) &mdash; the full position available to collar.{te_txt}</div>
      <table>
        <thead><tr><th>Scenario</th><th>Active CVaR</th><th>vs. no collar</th><th>Scenario TE</th>
          <th>Notional collared</th><th>Shares collared</th><th>Contracts</th></tr></thead>
        <tbody>{''.join(rows_html)}</tbody>
      </table>
    </div>"""


def main():
    result, meta = build_results()
    result.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")

    datasets = build_chart_datasets(result)
    path_order = meta["labels_order"]
    chart_labels = [PATH_SHORT.get(p, p) for p in path_order]
    chart_datasets_js = ",\n        ".join(
        "{label:%r, backgroundColor:%r, borderRadius:4, data:[%s]}" % (
            ds["label"], ds["color"],
            ",".join(f"{ds['data'].get(p, 0):.4f}" for p in path_order)
        )
        for ds in datasets
    )

    te_by_path = {
        label: result.loc[result["path"] == label, "te"].iloc[0] for label in path_order
    }
    dashboard_te_data_js = ",".join(
        f"{(te_by_path[p] or 0) * 100:.4f}" for p in path_order
    )
    scenario_te_datasets_js = ",\n        ".join(
        "{type:'bar', label:%r, backgroundColor:%r, borderRadius:4, data:[%s]}" % (
            f"Scenario TE ({COVERAGE_LABELS[coverage]})", COVERAGE_COLORS[coverage],
            ",".join(
                f"{result.loc[(result['path'] == p) & (result['coverage'] == coverage), 'scenario_te'].iloc[0] * 100:.4f}"
                for p in path_order
            )
        )
        for coverage in COVERAGE_LEVELS
    )

    overview_html = overview_table_html(result, path_order)
    sections_html = "\n".join(
        path_section_html(label, result[result["path"] == label], meta)
        for label in path_order
    )

    warnings_html = ""
    if meta["warnings"]:
        items = "".join(f"<li>{w}</li>" for w in meta["warnings"])
        warnings_html = f"""
    <div class="banner warn">
      <b>&#9888; Cross-check against the dashboard failed for some paths:</b>
      <ul>{items}</ul>
      Re-run optimize_tax_cvar_mvo.py before trusting this report.
    </div>"""
    else:
        warnings_html = """
    <div class="banner ok">
      &#10003; Every path's uncollared CVaR in this report matches Data/path_metrics.json
      exactly (the same number the rebalance dashboard displays) &mdash; the collar scenarios
      below are layered on top of that verified baseline, not a separately-derived number.
    </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Post-Collar Trade CVaR Analytics</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f1115; --card: #171a21; --border: #2a2f3a; --text: #e6e9ef; --sub: #9aa3b2;
    --accent: #2fae62; --red: #e5484d;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--text); font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; padding: 24px 16px 60px; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .sub {{ color: var(--sub); font-size: 14px; margin-bottom: 20px; max-width: 900px; line-height:1.5; }}
  .banner {{ border-radius: 10px; padding: 14px 18px; margin-bottom: 20px; font-size: 13.5px; line-height:1.5; }}
  .banner.ok {{ background: rgba(47,174,98,0.12); border: 1px solid rgba(47,174,98,0.4); }}
  .banner.warn {{ background: rgba(229,72,77,0.12); border: 1px solid rgba(229,72,77,0.4); }}
  .banner.warn ul {{ margin: 6px 0 0 18px; padding:0; }}
  .meta-row {{ display:flex; gap:14px; flex-wrap:wrap; margin-bottom: 22px; }}
  .stat {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:12px 16px; min-width:150px; }}
  .stat-l {{ font-size:11px; color:var(--sub); text-transform:uppercase; letter-spacing:.04em; }}
  .stat-v {{ font-size:18px; font-weight:600; margin-top:2px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:18px 20px; margin-bottom:18px; }}
  .card-h {{ font-size:15px; font-weight:600; margin-bottom:4px; }}
  .card-sub {{ font-size:12.5px; color:var(--sub); margin-bottom:12px; }}
  table {{ width:100%; border-collapse: collapse; font-size:13px; }}
  th, td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--border); }}
  th {{ color: var(--sub); font-weight:500; font-size:11.5px; text-transform:uppercase; letter-spacing:.03em; }}
  td.num, th.num {{ text-align:right; font-variant-numeric: tabular-nums; }}
  .chart-grid {{ display:grid; grid-template-columns: 1.4fr 1fr; gap:18px; margin-bottom: 18px; }}
  .chart-grid .card {{ margin-bottom: 0; }}
  @media (max-width: 820px) {{ .chart-grid {{ grid-template-columns: 1fr; }} }}
  .chart-wrap {{ height: 340px; }}
  .footnote {{ color: var(--sub); font-size: 11.5px; margin-top: 26px; line-height:1.6; max-width: 900px; }}
  code {{ background:#20242e; padding:1px 5px; border-radius:4px; font-size:12px; }}
</style>
</head>
<body>

  <h1>Post-Collar Trade CVaR Analytics</h1>
  <div class="sub">
    Active CVaR (5th-percentile annual shortfall vs. the policy benchmark, same 58 overlapping
    12-month scenarios and Oct-2020&ndash;Jun-2026 common history as the rebalance dashboard) for
    each optimized trade path, with a zero-cost MSTR collar (put floor {PUT_FLOOR:+.0%} / call cap
    {CALL_CAP:+.0%} per 12-month scenario window, no premium modeled) applied at four coverage
    levels: none, 30%, 50%, and a full 100% collar on the post-trade MSTR position.
  </div>

  {warnings_html}

  <div class="meta-row">
    <div class="stat"><div class="stat-l">MSTR implied price</div><div class="stat-v">{fmt_money(meta['snow_price'])}</div></div>
    <div class="stat"><div class="stat-l">Put strike (&minus;10%)</div><div class="stat-v">{fmt_money(meta['put_strike'])}</div></div>
    <div class="stat"><div class="stat-l">Call strike (+20%)</div><div class="stat-v">{fmt_money(meta['call_strike'])}</div></div>
    <div class="stat"><div class="stat-l">Total household value</div><div class="stat-v">{fmt_money(meta['total_household_value'])}</div></div>
  </div>

  <div class="chart-grid">
    <div class="card">
      <div class="card-h">Active CVaR by Path &times; Collar Coverage</div>
      <div class="card-sub">Bars are raw Active CVaR (not absolute value) &mdash; more negative is worse. Coverage blends MSTR's
        raw and collared scenario return linearly (30% coverage = 30% of the MSTR position collared, 70% left naked).</div>
      <div class="chart-wrap"><canvas id="cvarChart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-h">Tracking Error by Path &mdash; Dashboard (line) vs. Scenario (bars)</div>
      <div class="card-sub">The purple line is the dashboard's parametric factor-model TE (from
        <code>path_metrics.json</code>) &mdash; fixed, does not move with the collar. The bars are the empirical
        Scenario TE (std dev of the same 58 scenario active returns Active CVaR uses) at each coverage level &mdash;
        this one DOES fall as more of MSTR is collared.</div>
      <div class="chart-wrap"><canvas id="teChart"></canvas></div>
    </div>
  </div>

  {overview_html}

  {sections_html}

  <div class="footnote">
    Methodology: one collar struck per 12-month CVaR scenario window (not rolled monthly), zero-cost costless clip
    (no Black-Scholes / implied-vol premium modeled), only MSTR's scenario row is touched &mdash; every other ticker's
    returns and every path's post-trade weights (<code>Proposed_Weight</code> from each
    <code>trade_proposals_path*.csv</code>) are exactly what <code>optimize_tax_cvar_mvo.py</code> already produced.
    Strikes are off MSTR's current implied price (Market_Value / Quantity); shares/contracts are sized off each
    path's <em>post-trade</em> MSTR notional, not today's full pre-trade position. Generated by
    <code>generate_collar_html.py</code> in <code>portfolio-benchmarking/demos/collar-strategy/</code>.
  </div>

<script>
new Chart(document.getElementById('cvarChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(chart_labels)},
    datasets: [
        {chart_datasets_js}
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'top', labels: {{ color: '#e6e9ef', boxWidth: 10, padding: 14 }} }},
      tooltip: {{ callbacks: {{ label: c => c.dataset.label + ': ' + c.raw.toFixed(2) + '%' }} }}
    }},
    scales: {{
      y: {{ ticks: {{ color: '#9aa3b2', callback: v => v.toFixed(0) + '%' }}, grid: {{ color: '#2a2f3a' }} }},
      x: {{ ticks: {{ color: '#9aa3b2' }}, grid: {{ display: false }} }}
    }}
  }}
}});

new Chart(document.getElementById('teChart'), {{
  data: {{
    labels: {json.dumps(chart_labels)},
    datasets: [
        {scenario_te_datasets_js},
        {{type:'line', label:'Tracking Error (dashboard, fixed)', borderColor:'#7c5cff',
          backgroundColor:'#7c5cff', borderWidth:2, pointRadius:4, fill:false, tension:0,
          data:[{dashboard_te_data_js}]}}
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'top', labels: {{ color: '#e6e9ef', boxWidth: 10, padding: 14, font:{{size:10}} }} }},
      tooltip: {{ callbacks: {{ label: c => c.dataset.label + ': ' + c.raw.toFixed(2) + '%' }} }}
    }},
    scales: {{
      y: {{ ticks: {{ color: '#9aa3b2', callback: v => v.toFixed(0) + '%' }}, grid: {{ color: '#2a2f3a' }} }},
      x: {{ ticks: {{ color: '#9aa3b2' }}, grid: {{ display: false }} }}
    }}
  }}
}});
</script>
</body>
</html>
"""

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
