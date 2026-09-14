r"""
construct_benchmark_mvo.py

Constructs the STRATEGIC BENCHMARK: a single, fixed (never rebalanced),
long-only weight vector over the 5 Fama-French factor return streams
(Mkt-RF, SMB, HML, RMW, CMA), chosen by mean-variance (max-Sharpe)
optimization, with:
  - long-only (all weights >= 0)
  - weights sum to 1
  - market-factor (Mkt-RF) weight constrained to [0.60, 0.70]
  - full-sample covariance matrix (NOT Ledoit-Wolf / shrunk) -- this step is
    about picking one fixed policy weight set, and the sample covariance of
    the same 5 well-identified factors over the full ~20yr history is a
    defensible, simple, transparent choice for that. (Ledoit-Wolf comes back
    in the later tracking-error model, which needs a covariance estimate
    that behaves well against a live/rolling portfolio.)

Why only the 5 FF factors: Mkt-RF/SMB/HML/RMW/CMA are the only factors in
this project's factor set that are actual investable return streams (they're
long-short spread portfolios you can, in principle, replicate). The 9 macro
PCA factors (US_Growth_Factor, Inflation_Factor, etc.) are standardized
z-scores of macro indicator levels -- there's no security or spread that
pays you "the Inflation Factor return," so they're excluded from this
long-only weight-based benchmark. They stay useful downstream for explaining
*why* the benchmark or portfolio moved, just not as something to hold.

Method
------
1.  mu    = annualized mean of the 5 FF factor returns (monthly mean * 12)
2.  Sigma = annualized full-sample covariance matrix (monthly cov * 12)
3.  rf    = annualized mean risk-free rate (from the RF column)
4.  Maximize Sharpe ratio (w'mu - rf) / sqrt(w'Sigma w) subject to:
        sum(w) == 1
        0 <= w_i <= 1 for all i
        0.60 <= w_Mkt-RF <= 0.70
    via scipy.optimize.minimize (SLSQP).
5.  This ONE weight vector is then applied as a true buy-and-hold portfolio
    -- no rebalancing. Each factor's $1 wealth index is compounded
    independently from the start date; the benchmark's aggregate wealth is
    the weighted sum of those independently-compounding sleeves, so the
    *effective* weights drift over time exactly as a real unrebalanced
    portfolio's would. The benchmark return series used everywhere downstream
    (tracking error, etc.) is derived from that aggregate wealth curve, not
    from a constant-weight-rebalanced sum of returns.
6.  Reports CAGR, annualized vol, Sharpe, max drawdown -- the "does this look
    like a sane long-only benchmark" gut check (rule of thumb: ~8-12% CAGR
    for a mostly-equity long-only benchmark over this sample).

Inputs:
  - ..\factor-research\Data\FF_plus_macro_workable.csv  (FF5 + RF)

Outputs (Data\ folder next to this script):
  - benchmark_mvo_weights.csv       the single fixed weight vector
  - benchmark_wealth_curve.csv      $1 buy-and-hold wealth index + derived monthly returns
  - benchmark_effective_weights.csv how the buy-and-hold weights drifted over time
  - benchmark_mvo_summary.txt       CAGR / vol / Sharpe / max drawdown / weights
"""

import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PROJECTS_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "factor-research", "Data")
FF_PATH = os.path.join(PROJECTS_DATA_DIR, "FF_plus_macro_workable.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "Data")

FF_FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
MARKET_COL = "Mkt-RF"
MARKET_MIN, MARKET_MAX = 0.60, 0.70
ANNUALIZE = 12


def load_ff(path=FF_PATH):
    raw = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    ff = raw[FF_FACTOR_COLS] / 100.0
    rf = raw["RF"] / 100.0
    data = ff.join(rf.rename("RF")).dropna()
    return data[FF_FACTOR_COLS], data["RF"]


def max_sharpe_weights(mu, sigma, rf_annual, market_col=MARKET_COL,
                        market_bounds=(MARKET_MIN, MARKET_MAX)):
    n = len(mu)
    cols = list(mu.index)
    mkt_idx = cols.index(market_col)

    def neg_sharpe(w):
        # mu holds mean EXCESS (of RF) factor returns, so w @ mu is already
        # the portfolio's expected excess-of-RF return -- do not subtract
        # rf_annual again here.
        port_ret_excess = w @ mu.values
        port_vol = np.sqrt(w @ sigma.values @ w)
        return -port_ret_excess / port_vol

    bounds = [(0.0, 1.0)] * n
    bounds[mkt_idx] = market_bounds
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    w0 = np.full(n, (1.0 - np.mean(market_bounds)) / (n - 1))
    w0[mkt_idx] = np.mean(market_bounds)

    result = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds,
                       constraints=constraints,
                       options={"maxiter": 1000, "ftol": 1e-12})
    if not result.success:
        raise RuntimeError(f"MVO optimization failed: {result.message}")

    return pd.Series(result.x, index=cols)


def buy_and_hold_wealth(returns, weights, rf, start_value=1.0):
    """
    True (no-rebalance) buy-and-hold, with RF correctly layered in.

    All 5 FF factors here are EXCESS-of-RF returns (Mkt-RF explicitly so;
    SMB/HML/RMW/CMA are self-financing zero-investment spreads that need no
    RF added to themselves). Since the whole $1 of capital is always fully
    invested -- whether directly in the market or sitting as cash/collateral
    behind the self-financing spread overlays -- RF applies to the ENTIRE
    book every period, once, not weighted by any single sleeve's size:

        Total_Return_t = RF_t + sum_i active_weight_i,t-1 * Factor_i,t

    The active_weight_i,t are the RELATIVE sizes of the 5 factor tilts
    among themselves (ignoring RF), drifting via each tilt's own
    cumulative excess-return compounding -- this is the "no rebalancing"
    part. Returns (wealth_series, effective_weights_df) where
    effective_weights_df is that relative (RF-excluded) drift, useful for
    seeing how the style mix has moved, not literal dollar shares of a
    3-way (equity/spread/cash) split.
    """
    active_notional = weights.values * (1.0 + returns).cumprod()
    active_notional = pd.DataFrame(active_notional, index=returns.index, columns=returns.columns)
    active_weight = active_notional.div(active_notional.sum(axis=1), axis=0)
    active_weight_lag = active_weight.shift(1)
    active_weight_lag.iloc[0] = weights.values

    port_ret = rf + (active_weight_lag[returns.columns] * returns).sum(axis=1)
    wealth = start_value * (1.0 + port_ret).cumprod()
    wealth = wealth.rename("Benchmark_Wealth")
    return wealth, active_weight


def performance_stats(wealth, monthly_returns, rf_annual):
    n_months = len(wealth)
    cagr = (wealth.iloc[-1] / wealth.iloc[0]) ** (ANNUALIZE / n_months) - 1.0
    ann_vol = monthly_returns.std() * np.sqrt(ANNUALIZE)
    ann_arith_mean = monthly_returns.mean() * ANNUALIZE
    sharpe = (ann_arith_mean - rf_annual) / ann_vol
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    max_dd = drawdown.min()
    return {
        "CAGR": cagr,
        "Annualized Volatility": ann_vol,
        "Annualized Arithmetic Mean Return": ann_arith_mean,
        "Sharpe Ratio (vs avg RF)": sharpe,
        "Max Drawdown": max_dd,
        "N_Months": n_months,
    }


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    ff, rf = load_ff()
    print(f"Sample: {ff.index.min().date()} -> {ff.index.max().date()} ({len(ff)} monthly obs)")
    print(f"FF factors: {FF_FACTOR_COLS}\n")

    mu = ff.mean() * ANNUALIZE
    sigma = ff.cov() * ANNUALIZE  # full sample covariance, NOT shrunk
    rf_annual = rf.mean() * ANNUALIZE

    print("Annualized mean returns (mu):")
    print(mu.round(4))
    print(f"\nAnnualized risk-free rate: {rf_annual:.4%}")
    print("\nAnnualized full-sample covariance matrix (Sigma):")
    print(sigma.round(5))

    weights = max_sharpe_weights(mu, sigma, rf_annual)
    print(f"\n=== MVO benchmark weights (market constrained to "
          f"[{MARKET_MIN:.0%}, {MARKET_MAX:.0%}]) ===")
    print(weights.round(4))
    weights.to_frame("Weight").to_csv(os.path.join(OUT_DIR, "benchmark_mvo_weights.csv"))

    port_excess_ret_at_weights = weights @ mu
    port_total_ret_at_weights = rf_annual + port_excess_ret_at_weights
    port_vol_at_weights = np.sqrt(weights @ sigma @ weights)
    sharpe_at_weights = port_excess_ret_at_weights / port_vol_at_weights
    print(f"\nAt these weights (static, rebalanced-return math):")
    print(f"  Expected annualized TOTAL return (RF + excess): {port_total_ret_at_weights:.4%}")
    print(f"  Expected annualized excess return (vs RF):       {port_excess_ret_at_weights:.4%}")
    print(f"  Expected annualized vol:                         {port_vol_at_weights:.4%}")
    print(f"  Expected Sharpe:                                 {sharpe_at_weights:.4f}")

    wealth, effective_weights = buy_and_hold_wealth(ff, weights, rf)
    benchmark_returns = wealth.pct_change().dropna().rename("Benchmark_Return")

    wealth_out = pd.concat([wealth, benchmark_returns], axis=1)
    wealth_out.to_csv(os.path.join(OUT_DIR, "benchmark_wealth_curve.csv"))
    effective_weights.to_csv(os.path.join(OUT_DIR, "benchmark_effective_weights.csv"))

    stats = performance_stats(wealth, benchmark_returns, rf_annual)
    print("\n=== Realized buy-and-hold benchmark performance (no rebalancing) ===")
    for k, v in stats.items():
        if k in ("N_Months",):
            print(f"  {k}: {v}")
        elif k.startswith("Sharpe"):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v:.4%}")

    print("\nActive-tilt (RF-excluded) relative weights, start vs. end of sample:")
    print(pd.DataFrame({
        "Start": effective_weights.iloc[0],
        "End": effective_weights.iloc[-1],
    }).round(4))

    gut_check = "PASSES" if 0.08 <= stats["CAGR"] <= 0.12 else "OUTSIDE"
    print(f"\nGut-check (expected 8-12% CAGR): {gut_check} ({stats['CAGR']:.2%})")

    summary_lines = [
        "Benchmark MVO Construction Summary",
        "===================================",
        "",
        f"Sample: {ff.index.min().date()} -> {ff.index.max().date()} ({len(ff)} monthly obs)",
        f"Factors: {FF_FACTOR_COLS}",
        f"Constraints: long-only, sum(w)=1, {MARKET_COL} in [{MARKET_MIN:.0%}, {MARKET_MAX:.0%}]",
        "Objective: maximize Sharpe ratio, full-sample (unshrunk) covariance",
        "",
        "Optimized weights (fixed at inception, never rebalanced):",
        weights.round(4).to_string(),
        "",
        f"Expected (static) annualized TOTAL return: {port_total_ret_at_weights:.4%}",
        f"Expected (static) annualized excess return: {port_excess_ret_at_weights:.4%}",
        f"Expected (static) annualized vol:    {port_vol_at_weights:.4%}",
        f"Expected (static) Sharpe:            {sharpe_at_weights:.4f}",
        "",
        "Realized buy-and-hold performance (no rebalancing):",
        *[f"  {k}: {v}" if k == "N_Months"
          else (f"  {k}: {v:.4f}" if k.startswith("Sharpe") else f"  {k}: {v:.4%}")
          for k, v in stats.items()],
        "",
        f"Gut-check vs. expected 8-12% CAGR: {gut_check}",
        "",
        "Active-tilt (RF-excluded) relative weights, start vs end:",
        pd.DataFrame({"Start": effective_weights.iloc[0], "End": effective_weights.iloc[-1]}).round(4).to_string(),
    ]
    with open(os.path.join(OUT_DIR, "benchmark_mvo_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\nSaved outputs to {OUT_DIR}:")
    for fname in [
        "benchmark_mvo_weights.csv",
        "benchmark_wealth_curve.csv",
        "benchmark_effective_weights.csv",
        "benchmark_mvo_summary.txt",
    ]:
        print(f"  {fname}")
