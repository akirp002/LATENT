r"""
benchmark_tracking_error_model.py

Constructs a factor-replicating benchmark for a portfolio and models tracking
error (TE) against it, using a Ledoit-Wolf shrinkage covariance matrix of the
factor returns.

*** PORTFOLIO DATA IS CURRENTLY SYNTHETIC / A PLACEHOLDER ***
There was no real portfolio return series available when this was built, so
a synthetic portfolio is generated from the factor set itself (see
`build_synthetic_portfolio` below) with a deliberate style drift (rising HML
and a macro-factor tilt over time) so the rolling-beta / tracking-error
machinery has something realistic to pick up. Swap in your own portfolio by
replacing `load_portfolio_returns()` with a read of an actual monthly return
series (see the commented-out alternative inside that function) -- nothing
else in the script needs to change.

Method
------
1.  Factors: reuse the FF5 + macro PCA factor set from
    quant_model_macro_2.py (Data/FF_plus_macro_workable.csv +
    Data/macro_pca_factors.csv in the "python-projects" folder).
2.  Benchmark construction (returns-based style regression):
    fit  Portfolio_t = alpha + sum_k beta_k * Factor_k,t + eps_t
    by full-sample OLS (HAC/Newey-West SEs). The fitted, alpha-stripped part
    -- sum_k beta_k * Factor_k,t -- is the "systematic benchmark": a passive
    combination of the factors that replicates the portfolio's average factor
    exposure. Active_t = Portfolio_t - Benchmark_t is the tracking-difference
    series (ex-post).
3.  Rolling betas: the SAME regression re-fit on a rolling 36-month window,
    but restricted to just the 5 FF factors, to get time-varying exposures
    b(t) so style drift is visible. The rolling fit deliberately drops the
    9 macro PCA factors: with only 36-60 monthly observations per window and
    14 highly collinear regressors, per-window OLS betas are not identified
    well enough to mean anything (verified empirically -- the "current vs.
    policy" delta below becomes dominated by pure estimation noise, not real
    drift, once all 14 factors are put in a short rolling window). FF5 is
    the canonical, low-dimensional style/benchmark factor set and is well
    conditioned at this window length; the full 14-factor set is still used
    for the static full-sample benchmark/ex-post TE above, where 244 monthly
    observations are enough to identify it.
4.  Ledoit-Wolf covariance: shrinkage-estimated covariance matrix of the FF5
    factor returns (sklearn.covariance.LedoitWolf), used as the risk model
    Sigma for ex-ante tracking error.
5.  Ex-ante (predicted) tracking error:
        TE_ante = sqrt(db' * Sigma_LW * db),  annualized *= sqrt(12)
    where db = current exposure (latest rolling-window beta) minus policy
    exposure (the long-run average rolling beta) -- i.e. how much the
    portfolio's current factor tilts have drifted from its own long-run
    policy. Swap `policy_beta` for an externally supplied target/benchmark
    exposure vector if you have one.
6.  Ex-post (realized) tracking error: annualized std of the full-sample
    Active_t series, plus a rolling 12-month realized TE series for
    comparison against the ex-ante number. Note the two are not expected to
    match exactly: ex-ante only prices systematic FF5 exposure drift, while
    ex-post also picks up idiosyncratic noise and the macro-factor-driven
    part of the residual.

Inputs (from the "python-projects" pipeline):
  - ..\python-projects\Data\FF_plus_macro_workable.csv
  - ..\python-projects\Data\macro_pca_factors.csv

Outputs (Data\ folder next to this script):
  - synthetic_portfolio_returns.csv     the placeholder portfolio return series
  - benchmark_regression_summary.txt    full-sample OLS summary (HAC SEs)
  - benchmark_static_betas.csv          full-sample beta/t-stat/R2 per factor
  - benchmark_fitted_vs_active.csv      Portfolio_t, Benchmark_t, Active_t
  - rolling_betas.csv                   36-month rolling FF5 factor exposures
  - ledoit_wolf_covariance.csv          shrinkage covariance matrix (FF5 factors)
  - tracking_error_summary.txt          ex-ante + ex-post TE numbers
  - rolling_realized_te.csv             rolling 12-month realized TE series
"""

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.covariance import LedoitWolf

PROJECTS_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "python-projects", "Data")
FF_PATH = os.path.join(PROJECTS_DATA_DIR, "FF_plus_macro_workable.csv")
MACRO_PATH = os.path.join(PROJECTS_DATA_DIR, "macro_pca_factors.csv")

OUT_DIR = os.path.join(os.path.dirname(__file__), "Data")

FF_FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
ROLL_WINDOW = 36          # months, standard style-analysis window
REALIZED_TE_WINDOW = 12   # months, rolling realized TE window
ANNUALIZE = np.sqrt(12)
SEED = 42


def load_factors():
    # FF factors are stored in percentage points (e.g. 3.03 == 3.03%); convert
    # to decimal fractions so they're on the same scale as everything else
    # here (macro PCA factors are unitless z-scores, already scale-consistent).
    ff = pd.read_csv(FF_PATH, parse_dates=["Date"]).set_index("Date")[FF_FACTOR_COLS] / 100.0
    macro = pd.read_csv(MACRO_PATH, parse_dates=["Date"]).set_index("Date")
    data = ff.join(macro, how="inner").dropna()
    return data


def build_synthetic_portfolio(factors, seed=SEED):
    r"""
    *** PLACEHOLDER ***. Replace this whole function with:

        def load_portfolio_returns(factors):
            port = pd.read_csv(r"path\to\your\portfolio_returns.csv",
                                parse_dates=["Date"]).set_index("Date")["Return"]
            return port.reindex(factors.index).dropna()

    once you have a real monthly return series. What follows fabricates one
    from the factors themselves: a mostly-market-beta equity book with a
    small, deliberately drifting HML/macro tilt (so rolling betas and the
    ex-ante TE model have real style drift to detect) plus idiosyncratic
    noise.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(len(factors))
    frac = t / max(t[-1], 1)

    macro_cols = [c for c in factors.columns if c not in FF_FACTOR_COLS]
    tilt_macro_col = macro_cols[0] if macro_cols else None

    base_weights = pd.Series(0.0, index=factors.columns)
    base_weights["Mkt-RF"] = 0.90
    base_weights["SMB"] = 0.10
    base_weights["HML"] = 0.05
    base_weights["RMW"] = 0.05
    base_weights["CMA"] = 0.00
    if tilt_macro_col is not None:
        base_weights[tilt_macro_col] = 0.05

    # Time-varying weights: HML tilt rises from 0.05 -> 0.35 (value drift),
    # macro tilt fades from 0.05 -> -0.05 over the sample.
    weights_t = pd.DataFrame(
        np.tile(base_weights.values, (len(factors), 1)),
        index=factors.index, columns=factors.columns,
    )
    weights_t["HML"] = 0.05 + 0.30 * frac
    if tilt_macro_col is not None:
        weights_t[tilt_macro_col] = 0.05 - 0.10 * frac

    alpha_monthly = 0.001  # 10 bps/mo manager skill, purely illustrative
    noise = rng.normal(0.0, 0.015, size=len(factors))  # 1.5%/mo idiosyncratic

    port_ret = alpha_monthly + (weights_t.values * factors.values).sum(axis=1) + noise
    return pd.Series(port_ret, index=factors.index, name="Portfolio_Return")


def hac_maxlags(n_obs):
    return int(np.floor(4 * (n_obs / 100) ** (2 / 9)))


def fit_static_benchmark(port, factors):
    x = sm.add_constant(factors)
    maxlags = hac_maxlags(len(port))
    result = sm.OLS(port, x).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

    betas = result.params.drop("const")
    benchmark = factors[betas.index].values @ betas.values
    benchmark = pd.Series(benchmark, index=factors.index, name="Benchmark_Return")
    active = (port - benchmark).rename("Active_Return")
    return result, betas, benchmark, active


def rolling_betas(port, factors, window=ROLL_WINDOW):
    cols = list(factors.columns)
    idx, rows = [], []
    for end in range(window, len(port) + 1):
        sl = slice(end - window, end)
        x = sm.add_constant(factors.iloc[sl])
        y = port.iloc[sl]
        res = sm.OLS(y, x).fit()
        rows.append(res.params.drop("const").reindex(cols).values)
        idx.append(port.index[end - 1])
    return pd.DataFrame(rows, index=idx, columns=cols)


def rolling_realized_te(active, window=REALIZED_TE_WINDOW):
    return (active.rolling(window).std() * ANNUALIZE).rename("Rolling_Realized_TE")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Load factors + (synthetic, for now) portfolio ---
    factors = load_factors()
    factor_cols = list(factors.columns)
    port = build_synthetic_portfolio(factors)
    print(f"Aligned sample: {factors.index.min().date()} -> {factors.index.max().date()} "
          f"({len(factors)} monthly observations)")
    print(f"Factors ({len(factor_cols)}): {factor_cols}\n")

    port.to_frame().to_csv(os.path.join(OUT_DIR, "synthetic_portfolio_returns.csv"))

    # --- 1) Static full-sample benchmark ---
    static_result, static_betas, benchmark, active = fit_static_benchmark(port, factors)
    print("=== Static (full-sample) benchmark regression ===")
    print(static_result.summary())

    with open(os.path.join(OUT_DIR, "benchmark_regression_summary.txt"), "w") as f:
        f.write(str(static_result.summary()))

    static_out = pd.DataFrame({
        "Beta": static_result.params,
        "T_Stat": static_result.tvalues,
    })
    static_out["R2"] = static_result.rsquared
    static_out.to_csv(os.path.join(OUT_DIR, "benchmark_static_betas.csv"))

    fitted_vs_active = pd.concat([port, benchmark, active], axis=1)
    fitted_vs_active.to_csv(os.path.join(OUT_DIR, "benchmark_fitted_vs_active.csv"))

    ex_post_te_monthly = active.std()
    ex_post_te_annual = ex_post_te_monthly * ANNUALIZE
    print(f"\nEx-post (realized, full-sample) tracking error: "
          f"{ex_post_te_monthly:.4%} monthly / {ex_post_te_annual:.4%} annualized")

    # --- 2) Rolling betas (style drift) -- FF5 only, see module docstring
    # for why the 9 macro factors are excluded from the rolling fit ---
    roll_betas = rolling_betas(port, factors[FF_FACTOR_COLS], ROLL_WINDOW)
    roll_betas.to_csv(os.path.join(OUT_DIR, "rolling_betas.csv"))
    print(f"\nRolling {ROLL_WINDOW}-month FF5 betas computed: {len(roll_betas)} windows")

    current_beta = roll_betas.iloc[-1]
    policy_beta = roll_betas.mean()  # long-run average exposure = "policy" benchmark
    db = (current_beta - policy_beta).reindex(FF_FACTOR_COLS)

    print("\nCurrent (latest window) vs policy (long-run average) factor exposure:")
    print(pd.DataFrame({"Current": current_beta, "Policy": policy_beta, "Delta": db}).round(4))

    # --- 3) Ledoit-Wolf shrinkage covariance of the FF5 factor returns ---
    lw = LedoitWolf().fit(factors[FF_FACTOR_COLS].values)
    sigma_lw = pd.DataFrame(lw.covariance_, index=FF_FACTOR_COLS, columns=FF_FACTOR_COLS)
    sigma_lw.to_csv(os.path.join(OUT_DIR, "ledoit_wolf_covariance.csv"))
    print(f"\nLedoit-Wolf shrinkage intensity: {lw.shrinkage_:.4f} "
          f"(0 = pure sample covariance, 1 = pure shrinkage target)")

    # --- 4) Ex-ante tracking error: TE = sqrt(db' Sigma_LW db) ---
    db_vec = db.values
    te_ante_monthly = np.sqrt(db_vec @ sigma_lw.values @ db_vec)
    te_ante_annual = te_ante_monthly * ANNUALIZE
    print(f"\nEx-ante (Ledoit-Wolf) tracking error given current style drift: "
          f"{te_ante_monthly:.4%} monthly / {te_ante_annual:.4%} annualized")

    # For comparison: sample (non-shrunk) covariance version of the same calc
    sigma_sample = factors[FF_FACTOR_COLS].cov().values
    te_ante_sample_annual = np.sqrt(db_vec @ sigma_sample @ db_vec) * ANNUALIZE

    # --- 5) Rolling realized TE series, for comparison against ex-ante ---
    roll_te = rolling_realized_te(active, REALIZED_TE_WINDOW)
    roll_te.to_csv(os.path.join(OUT_DIR, "rolling_realized_te.csv"))

    summary_lines = [
        "Tracking Error Summary",
        "=======================",
        "",
        "*** Portfolio return series is SYNTHETIC/PLACEHOLDER -- see the ",
        "module docstring in benchmark_tracking_error_model.py to swap in ",
        "a real one. ***",
        "",
        f"Sample: {factors.index.min().date()} -> {factors.index.max().date()} "
        f"({len(factors)} monthly obs)",
        f"Factors: {factor_cols}",
        "",
        f"Ex-post realized TE (full sample):        "
        f"{ex_post_te_monthly:.4%} monthly / {ex_post_te_annual:.4%} annualized",
        f"Ex-ante TE (Ledoit-Wolf, FF5 style drift vs. policy):  "
        f"{te_ante_monthly:.4%} monthly / {te_ante_annual:.4%} annualized",
        f"  (sample-covariance version, for comparison):   {te_ante_sample_annual:.4%} annualized",
        f"Ledoit-Wolf shrinkage intensity: {lw.shrinkage_:.4f}",
        "",
        f"Rolling {REALIZED_TE_WINDOW}-mo realized TE: "
        f"min {roll_te.min():.4%}, max {roll_te.max():.4%}, "
        f"latest {roll_te.iloc[-1]:.4%}",
        "",
        "Current vs policy (long-run average) factor exposure delta (db):",
        pd.DataFrame({"Current": current_beta, "Policy": policy_beta, "Delta": db}).round(4).to_string(),
    ]
    summary_text = "\n".join(summary_lines)
    with open(os.path.join(OUT_DIR, "tracking_error_summary.txt"), "w") as f:
        f.write(summary_text)

    print(f"\nSaved outputs to {OUT_DIR}:")
    for fname in [
        "synthetic_portfolio_returns.csv",
        "benchmark_regression_summary.txt",
        "benchmark_static_betas.csv",
        "benchmark_fitted_vs_active.csv",
        "rolling_betas.csv",
        "ledoit_wolf_covariance.csv",
        "tracking_error_summary.txt",
        "rolling_realized_te.csv",
    ]:
        print(f"  {fname}")
