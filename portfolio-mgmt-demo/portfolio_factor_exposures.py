r"""
portfolio_factor_exposures.py

The core analysis: returns-based FF5 factor regression for every sample
holding, aggregated up to portfolio-level factor exposure (full portfolio,
and ex-concentrated-position), compared against the benchmark's factor
exposure (from construct_benchmark_mvo.py), and turned into tracking error
two ways.

Method
------
1.  Per-holding regression: for each ticker, regress its own excess return
    (Return - RF) on the 5 FF factors over whatever history yfinance
    returned for it (the concentrated position typically has much less
    history than the older core holdings, since it's modeled as a
    recent-IPO name -- see generate_sample_holdings.py). HAC (Newey-West)
    SEs, same convention as quant_model_macro_3.py.
2.  Portfolio-level exposure: weighted sum of the per-holding betas, using
    current market-value weights. Two versions:
      - Full portfolio (concentrated position included)
      - Ex-concentrated-position (dropped, remaining holdings re-normalized to 100%)
3.  Benchmark exposure: the fixed weight vector from construct_benchmark_mvo.py
    (60% Mkt-RF / 40% RMW), i.e. beta_bench = [0.60, 0, 0, 0.40, 0] exactly,
    since the benchmark IS defined by those weights on those factors (no
    regression needed there -- it's constructed, not estimated).
4.  Active exposure: db = beta_portfolio - beta_benchmark.
5.  Ex-ante tracking error: TE = sqrt(db' Sigma db), annualized, where Sigma
    is the FULL-SAMPLE (unshrunk) annualized FF5 covariance matrix -- same
    covariance convention as the benchmark construction (no Ledoit-Wolf in
    this pipeline).
6.  Ex-post tracking error: reconstruct a "current weights, held constant
    back through time" portfolio return series (i.e. what these exact
    weights would have earned each month, common-history overlap across
    holdings), compare it to the benchmark's realized buy-and-hold return
    series from construct_benchmark_mvo.py, and take the annualized std of
    the difference. This is a real, non-trivial check on the regression-
    based ex-ante number.

*** SAMPLE DATA ***
Every number this script prints or saves comes from the fabricated demo
portfolio in generate_sample_holdings.py -- tickers, weights, and dollar
values are all made up, not a real client's real holdings.

Inputs:
  - Data\holdings_clean.csv             (from generate_sample_holdings.py)
  - Data\holdings_monthly_returns.csv   (from fetch_holdings_prices.py)
  - Data\benchmark_mvo_weights.csv      (from construct_benchmark_mvo.py)
  - Data\benchmark_wealth_curve.csv     (from construct_benchmark_mvo.py)
  - ..\python-projects\Data\FF_plus_macro_workable.csv  (FF5 + RF)

Outputs (Data\ folder):
  - holding_betas.csv                 per-ticker FF5 betas, t-stats, R2, N_Obs
  - portfolio_factor_exposures.csv    full vs ex-concentrated vs benchmark betas, and deltas
  - tracking_error_vs_benchmark.txt   ex-ante + ex-post TE, full vs ex-concentrated
  - portfolio_active_returns.csv      monthly portfolio/benchmark/active return series, both versions
"""

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROJECTS_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "python-projects", "Data")
FF_PATH = os.path.join(PROJECTS_DATA_DIR, "FF_plus_macro_workable.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "Data")

FF_FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
MIN_OBS = 24
ANNUALIZE = np.sqrt(12)
CONCENTRATED_TICKER = "PLTR"  # matches generate_sample_holdings.py


def load_ff():
    raw = pd.read_csv(FF_PATH, parse_dates=["Date"]).set_index("Date")
    ff = raw[FF_FACTOR_COLS] / 100.0
    rf = raw["RF"] / 100.0
    return ff, rf


def hac_maxlags(n_obs):
    return int(np.floor(4 * (n_obs / 100) ** (2 / 9)))


def regress_holdings(returns_wide, ff, rf, tickers):
    rows = []
    betas = {}
    for t in tickers:
        if t not in returns_wide.columns:
            continue
        ticker_ret = returns_wide[t].dropna() / 100.0
        data = ff.join(rf.rename("RF")).join(ticker_ret.rename("Ret"), how="inner").dropna()
        n_obs = len(data)
        if n_obs < MIN_OBS:
            print(f"  skipping {t}: only {n_obs} aligned obs (< {MIN_OBS})")
            continue

        excess = data["Ret"] - data["RF"]
        x = sm.add_constant(data[FF_FACTOR_COLS])
        result = sm.OLS(excess, x).fit(cov_type="HAC", cov_kwds={"maxlags": hac_maxlags(n_obs)})

        betas[t] = result.params.drop("const")
        for term in ["const"] + FF_FACTOR_COLS:
            rows.append({
                "Ticker": t, "Term": "Alpha" if term == "const" else term,
                "Beta": result.params[term], "T_Stat": result.tvalues[term],
                "R2": result.rsquared, "N_Obs": n_obs,
                "Start": data.index.min().date(), "End": data.index.max().date(),
            })
    return pd.DataFrame(rows), pd.DataFrame(betas).T  # betas: index=ticker, cols=FF5


def portfolio_exposure(holdings, betas_df, weight_col):
    merged = holdings.merge(betas_df.reset_index().rename(columns={"index": "Ticker"}),
                             on="Ticker", how="inner")
    w = merged[weight_col]
    w_norm = w / w.sum()  # in case some tickers were dropped for insufficient history
    exposure = (merged[FF_FACTOR_COLS].values * w_norm.values[:, None]).sum(axis=0)
    coverage = w.sum()
    return pd.Series(exposure, index=FF_FACTOR_COLS), coverage


if __name__ == "__main__":
    holdings = pd.read_csv(os.path.join(OUT_DIR, "holdings_clean.csv"))
    returns_wide = pd.read_csv(os.path.join(OUT_DIR, "holdings_monthly_returns.csv"),
                                parse_dates=["Date"]).set_index("Date")
    ff, rf = load_ff()
    bench_weights = pd.read_csv(os.path.join(OUT_DIR, "benchmark_mvo_weights.csv"), index_col=0)["Weight"]
    bench_wealth = pd.read_csv(os.path.join(OUT_DIR, "benchmark_wealth_curve.csv"),
                                parse_dates=["Date"]).set_index("Date")

    tickers = sorted(holdings["Ticker"].unique())
    print(f"Regressing {len(tickers)} holdings on FF5 (HAC SEs)...\n")
    beta_long, beta_wide = regress_holdings(returns_wide, ff, rf, tickers)
    beta_long.to_csv(os.path.join(OUT_DIR, "holding_betas.csv"), index=False)

    print("Per-holding FF5 betas:")
    print(beta_wide.round(3))

    # --- Portfolio-level exposure: full vs ex-concentrated-position ---
    exp_full, cov_full = portfolio_exposure(holdings, beta_wide, "Weight_Full")
    exp_ex_conc, cov_ex_conc = portfolio_exposure(
        holdings[holdings["Ticker"] != CONCENTRATED_TICKER], beta_wide, "Weight_ExConcentrated")

    beta_bench = pd.Series(0.0, index=FF_FACTOR_COLS)
    beta_bench.update(bench_weights)

    exposures = pd.DataFrame({
        f"Portfolio (Full, w/ {CONCENTRATED_TICKER})": exp_full,
        "Portfolio (Ex-Concentrated)": exp_ex_conc,
        "Benchmark": beta_bench,
    })
    exposures["Delta_Full"] = exposures[f"Portfolio (Full, w/ {CONCENTRATED_TICKER})"] - exposures["Benchmark"]
    exposures["Delta_ExConcentrated"] = exposures["Portfolio (Ex-Concentrated)"] - exposures["Benchmark"]
    exposures.to_csv(os.path.join(OUT_DIR, "portfolio_factor_exposures.csv"))

    print(f"\nRegression weight coverage: full={cov_full:.1%}, ex-concentrated={cov_ex_conc:.1%} "
          f"(should be ~100% unless a holding had too little history)")
    print("\nFactor exposures (portfolio vs benchmark):")
    print(exposures.round(4))

    # --- Ex-ante tracking error: TE = sqrt(db' Sigma db), full-sample covariance ---
    sigma = (ff.cov() * 12).values  # annualized, unshrunk -- consistent with benchmark build
    db_full = exposures["Delta_Full"].values
    db_ex_conc = exposures["Delta_ExConcentrated"].values

    te_ante_full = np.sqrt(db_full @ sigma @ db_full)
    te_ante_ex_conc = np.sqrt(db_ex_conc @ sigma @ db_ex_conc)

    print(f"\nEx-ante tracking error (regression-based, full-sample FF5 covariance):")
    print(f"  Full portfolio (w/ {CONCENTRATED_TICKER}):  {te_ante_full:.4%}")
    print(f"  Ex-concentrated portfolio:         {te_ante_ex_conc:.4%}")

    # --- Ex-post tracking error: current weights held constant, common history ---
    monthly_ret = returns_wide[tickers] / 100.0

    def realized_portfolio_return(weight_col, exclude_concentrated=False):
        h = holdings[holdings["Ticker"] != CONCENTRATED_TICKER] if exclude_concentrated else holdings
        # Holdings has one row per (account, ticker) -- the same ticker can
        # appear in both accounts, so aggregate by ticker first (monthly_ret
        # is per-ticker, not per-account).
        w = h.groupby("Ticker")[weight_col].sum()
        w = w / w.sum()
        avail = [t for t in w.index if t in monthly_ret.columns]
        w = w[avail] / w[avail].sum()
        # Restrict to months where EVERY weighted ticker has actual data --
        # summing over only the tickers available in a given month would
        # silently treat a missing ticker's weight as if it earned 0% that
        # month instead of properly shrinking the comparison window (the
        # concentrated position only starts trading partway through history).
        sub = monthly_ret[avail].dropna()
        return (sub * w).sum(axis=1)

    port_ret_full = realized_portfolio_return("Weight_Full", exclude_concentrated=False).rename("Portfolio_Full")
    port_ret_ex_conc = realized_portfolio_return("Weight_ExConcentrated", exclude_concentrated=True).rename("Portfolio_ExConcentrated")
    bench_ret = bench_wealth["Benchmark_Return"].rename("Benchmark")

    combined = pd.concat([port_ret_full, port_ret_ex_conc, bench_ret], axis=1).dropna()
    combined["Active_Full"] = combined["Portfolio_Full"] - combined["Benchmark"]
    combined["Active_ExConcentrated"] = combined["Portfolio_ExConcentrated"] - combined["Benchmark"]
    combined.to_csv(os.path.join(OUT_DIR, "portfolio_active_returns.csv"))

    te_post_full = combined["Active_Full"].std() * ANNUALIZE
    te_post_ex_conc = combined["Active_ExConcentrated"].std() * ANNUALIZE

    print(f"\nEx-post (realized) tracking error over common overlap "
          f"({combined.index.min().date()} -> {combined.index.max().date()}, {len(combined)} months, "
          f"limited by the concentrated position's shorter history):")
    print(f"  Full portfolio (w/ {CONCENTRATED_TICKER}):  {te_post_full:.4%}")
    print(f"  Ex-concentrated portfolio:         {te_post_ex_conc:.4%}")

    port_cagr_full = (1 + combined["Portfolio_Full"]).prod() ** (12 / len(combined)) - 1
    port_cagr_ex_conc = (1 + combined["Portfolio_ExConcentrated"]).prod() ** (12 / len(combined)) - 1
    bench_cagr = (1 + combined["Benchmark"]).prod() ** (12 / len(combined)) - 1

    print(f"\nCAGR over the same overlap window:")
    print(f"  Full portfolio (w/ {CONCENTRATED_TICKER}):  {port_cagr_full:.4%}")
    print(f"  Ex-concentrated portfolio:         {port_cagr_ex_conc:.4%}")
    print(f"  Benchmark:                 {bench_cagr:.4%}")

    summary_lines = [
        "Tracking Error vs. Constructed Benchmark (SAMPLE DATA)",
        "========================================================",
        "",
        "Benchmark: 60% Mkt-RF / 40% RMW, fixed weights, no rebalancing "
        "(construct_benchmark_mvo.py)",
        "Covariance: full-sample (unshrunk) annualized FF5 covariance matrix",
        "",
        "Factor exposures (beta):",
        exposures.round(4).to_string(),
        "",
        f"Ex-ante TE (sqrt(db' Sigma db)):",
        f"  Full portfolio (w/ {CONCENTRATED_TICKER}):  {te_ante_full:.4%}",
        f"  Ex-concentrated portfolio:         {te_ante_ex_conc:.4%}",
        "",
        f"Ex-post (realized) TE, common overlap "
        f"{combined.index.min().date()} -> {combined.index.max().date()} "
        f"({len(combined)} months, capped by the concentrated position's shorter history):",
        f"  Full portfolio (w/ {CONCENTRATED_TICKER}):  {te_post_full:.4%}",
        f"  Ex-concentrated portfolio:         {te_post_ex_conc:.4%}",
        "",
        f"CAGR over the same overlap window:",
        f"  Full portfolio (w/ {CONCENTRATED_TICKER}):  {port_cagr_full:.4%}",
        f"  Ex-concentrated portfolio:         {port_cagr_ex_conc:.4%}",
        f"  Benchmark:                 {bench_cagr:.4%}",
    ]
    with open(os.path.join(OUT_DIR, "tracking_error_vs_benchmark.txt"), "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\nSaved outputs to {OUT_DIR}:")
    for fname in [
        "holding_betas.csv", "portfolio_factor_exposures.csv",
        "tracking_error_vs_benchmark.txt", "portfolio_active_returns.csv",
    ]:
        print(f"  {fname}")
