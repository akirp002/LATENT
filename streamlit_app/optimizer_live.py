"""
Live-recompute wrapper around portfolio-benchmarking/optimize_tax_cvar_mvo.py
and portfolio_risk_attribution.py, for the "Optimizer Lab" page.

Unlike data_loader.py (which only reads pre-computed Data/*.csv|json|txt),
this module imports and CALLS the real optimizer's solve() function with
user-adjustable parameters, so a visitor can sweep e.g. the MSTR sell cap,
tax budget, turnover penalty, or idio shrinkage and see the resulting
trade-off live, instead of only the two fixed Path C / Path D runs already
baked into Data/path_metrics.json.

Nothing here changes the optimizer's math -- solve() and the Euler risk
decomposition are called exactly as optimize_tax_cvar_mvo.py's own __main__
and portfolio_risk_attribution.py's own __main__ call them. The only new
code is:
  - monkeypatching the handful of module-level constants solve() reads
    directly (TAX_BUDGET, CVAR_FLOOR, TURNOVER_LAMBDA, MIN_TICKER_WEIGHT)
    around each call, since solve() itself doesn't take them as arguments
  - a block-bootstrap of the factor covariance to turn the single TE number
    into a distribution (for the TE box plot), since TE itself has no
    natural sampling distribution otherwise
  - a scenario-return distribution per configuration (for the CVaR box
    plot), which is exactly the same 12-month overlapping-scenario active
    returns build_cvar_scenarios() / active_cvar() already compute --
    reused, not recomputed differently

All numbers here are still on the same fabricated sample household as the
rest of this app (see the repo README / docs/HOW_IT_WORKS.md).
"""

import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
PB_DIR = os.path.join(os.path.dirname(ROOT), "portfolio-benchmarking")
FR_DATA = os.path.join(os.path.dirname(ROOT), "factor-research", "Data")

# The 9 macro PCA factors factor-research/quant_model_macro_*.py already
# built from FRED series (growth, inflation, rates, policy, currency,
# credit, commodities) -- see factor-research/Data/macro_pca_factors.csv.
# These are macro-THEME factors, distinct from the 5 Fama-French factors the
# optimizer itself minimizes tracking error against.
MACRO_FACTOR_COLS = [
    "US_Growth_Factor", "Global_Growth_Factor", "Inflation_Factor", "Rates_Yields_Factor",
    "US_Monetary_Policy_Factor", "Global_Monetary_Policy_Factor", "Currency_Factor",
    "Credit_Risk_Factor", "Commodities_Factor",
]


@st.cache_resource
def get_optimizer_modules():
    """Imports the real optimizer + risk-attribution modules from the
    sibling portfolio-benchmarking/ package. Cached (not reloaded) so the
    module-level constant monkeypatching in run_config() below always
    targets the one loaded instance."""
    if PB_DIR not in sys.path:
        sys.path.insert(0, PB_DIR)
    import optimize_tax_cvar_mvo as opt
    import portfolio_risk_attribution as attr

    return opt, attr


@st.cache_resource
def load_base_data():
    """Everything a solve() call needs that does NOT depend on the swept
    parameters: betas (HAC OLS, fixed), the blended factor covariance, and
    the CVaR/factor scenario matrices. Computed once per Streamlit session."""
    opt, attr = get_optimizer_modules()

    holdings_all = pd.read_csv(os.path.join(opt.DATA_DIR, "holdings_household_clean.csv"))
    returns_wide = pd.read_csv(
        os.path.join(opt.DATA_DIR, "holdings_household_monthly_returns.csv"), parse_dates=["Date"]
    ).set_index("Date")
    ff, rf = opt.load_ff()
    bench_weights = pd.read_csv(os.path.join(opt.DATA_DIR, "benchmark_mvo_weights.csv"), index_col=0)["Weight"]
    bench_wealth = pd.read_csv(
        os.path.join(opt.DATA_DIR, "benchmark_wealth_curve.csv"), parse_dates=["Date"]
    ).set_index("Date")

    tickers = sorted(holdings_all["Ticker"].unique())
    beta_wide, idio_var = opt.regress_tickers(returns_wide, ff, rf, tickers)
    usable_tickers = list(beta_wide.index)
    holdings = holdings_all[holdings_all["Ticker"].isin(usable_tickers)].copy()
    holdings["Weight_Household"] = holdings["Market_Value"] / holdings_all["Market_Value"].sum()

    beta_bench = pd.Series(0.0, index=opt.FF_FACTOR_COLS)
    beta_bench.update(bench_weights)
    sigma = opt.blended_factor_covariance(ff, annualize=12)
    A_scen, bench_annual = opt.build_cvar_scenarios(returns_wide, bench_wealth, usable_tickers)
    _, _, F_mat = attr.build_aligned_scenarios(returns_wide, bench_wealth, ff, usable_tickers)

    joint = holdings[holdings["Is_Taxable"]]
    class_of = {t: opt.ASSET_CLASS.get(t, t) for t in usable_tickers}

    return dict(
        holdings_all=holdings_all, returns_wide=returns_wide, ff=ff, rf=rf,
        beta_wide=beta_wide, idio_var=idio_var, usable_tickers=usable_tickers,
        holdings=holdings, beta_bench=beta_bench, sigma=sigma,
        A_scen=A_scen, bench_annual=bench_annual, F_mat=F_mat,
        joint=joint, class_of=class_of,
        total_household_value=float(holdings_all["Market_Value"].sum()),
    )


@st.cache_data(show_spinner=False)
def _compute_tax_per_dollar(joint_ticker, joint_lt_gain, joint_st_gain, joint_mv, tax_lt, tax_st):
    tax_per_dollar = pd.Series(
        (np.array(joint_lt_gain) * tax_lt + np.array(joint_st_gain) * tax_st) / np.array(joint_mv),
        index=joint_ticker,
    )
    return tax_per_dollar


def compute_tax_per_dollar(joint_df, tax_lt, tax_st):
    return _compute_tax_per_dollar(
        tuple(joint_df["Ticker"]), tuple(joint_df["LT_Gain"]), tuple(joint_df["ST_Gain"]),
        tuple(joint_df["Market_Value"]), tax_lt, tax_st,
    )


@st.cache_data(show_spinner="Estimating FF5 residual covariance (idiosyncratic risk)...")
def compute_omega(returns_wide, ff, rf, tickers, shrink):
    opt, _ = get_optimizer_modules()
    return opt.residual_covariance(returns_wide, ff, rf, tickers, shrink=shrink)


@dataclass
class OptConfig:
    label: str
    mstr_sell_cap: float = 0.70
    exclude_mstr_from_budget: bool = True
    tax_budget: float = 0.02
    enforce_cvar: bool = False
    cvar_floor: float = -0.15
    turnover_lambda: float = 0.02
    min_ticker_weight: float = 0.003
    idio_shrinkage: float = 0.4
    tax_lt: float = 0.295
    tax_st: float = 0.408


def run_config(cfg: OptConfig):
    """Runs the real solve() once for one parameter configuration and scores
    the result with the same te_of/cvar_of/tax_cost_of logic __main__ uses,
    plus the Euler factor+idio TE decomposition and the tail-conditional
    CVaR decomposition from portfolio_risk_attribution.py."""
    opt, attr = get_optimizer_modules()
    base = load_base_data()

    tax_per_dollar = compute_tax_per_dollar(base["joint"], cfg.tax_lt, cfg.tax_st)
    omega = compute_omega(base["returns_wide"], base["ff"], base["rf"], base["usable_tickers"], cfg.idio_shrinkage)

    saved = dict(
        TAX_BUDGET=opt.TAX_BUDGET, CVAR_FLOOR=opt.CVAR_FLOOR,
        TURNOVER_LAMBDA=opt.TURNOVER_LAMBDA, MIN_TICKER_WEIGHT=opt.MIN_TICKER_WEIGHT,
    )
    try:
        opt.TAX_BUDGET = cfg.tax_budget
        opt.CVAR_FLOOR = cfg.cvar_floor
        opt.TURNOVER_LAMBDA = cfg.turnover_lambda
        opt.MIN_TICKER_WEIGHT = cfg.min_ticker_weight
        exclude = frozenset({"MSTR"}) if cfg.exclude_mstr_from_budget else frozenset()
        outcome = opt.solve(
            base["holdings"], base["beta_wide"], base["beta_bench"], base["sigma"],
            base["A_scen"], base["bench_annual"], tax_per_dollar,
            enforce_cvar=cfg.enforce_cvar, enforce_tax_budget=True,
            exclude_from_tax_budget=exclude, max_sell_frac={"MSTR": cfg.mstr_sell_cap},
            omega=omega,
        )
    finally:
        opt.TAX_BUDGET = saved["TAX_BUDGET"]
        opt.CVAR_FLOOR = saved["CVAR_FLOOR"]
        opt.TURNOVER_LAMBDA = saved["TURNOVER_LAMBDA"]
        opt.MIN_TICKER_WEIGHT = saved["MIN_TICKER_WEIGHT"]

    if not outcome["feasible"]:
        return {"label": cfg.label, "config": cfg, "feasible": False, "message": outcome["message"]}

    x = outcome["x"]
    rows, x0 = outcome["rows"], outcome["x0"]
    beta_matrix, beta_bench_vec, sigma_mat = outcome["beta_matrix"], outcome["beta_bench_vec"], outcome["sigma_mat"]
    A_row, bench_annual_vec = outcome["A_row"], outcome["bench_annual_vec"]
    omega_G, omega_mat = outcome["omega_G"], outcome["omega_mat"]
    tickers_row = rows["Ticker"].values

    active_exposure = beta_matrix.T @ x - beta_bench_vec
    te_factor = float(np.sqrt(active_exposure @ sigma_mat @ active_exposure))
    w_ticker_corr = omega_G @ x
    te_idio_var_corr = float(w_ticker_corr @ omega_mat @ w_ticker_corr)
    te_total = float(np.sqrt(active_exposure @ sigma_mat @ active_exposure + te_idio_var_corr))

    cvar = opt.cvar_of(x, A_row, bench_annual_vec)
    active_scenarios = bench_annual_vec - A_row.T @ x  # the same distribution CVaR is the tail mean of

    tax_vec_full = np.array([tax_per_dollar.get(t, 0.0) for t in tickers_row])
    joint_mask = rows["Is_Taxable"].values
    sell = np.maximum(x0 - x, 0.0) * joint_mask
    tax_cost = float((sell * tax_vec_full).sum()) * base["total_household_value"]
    turnover = 0.5 * float(np.abs(x - x0).sum())

    # Euler TE risk contribution: factor (rescaled) + idiosyncratic-by-class,
    # exactly portfolio_risk_attribution.py's own decomposition (diagonal
    # idio model, for interpretability -- distinct from the correlated Omega
    # the optimizer itself minimizes; both are legitimate, see that module's
    # docstring item 1).
    te_c, te_factor_check = attr.te_contributions(active_exposure, sigma_mat, opt.FF_FACTOR_COLS)
    w_by_ticker = attr.weight_by_ticker(x, tickers_row)
    te_c_resc, idio_c, te_total_attr, idio_var_total = attr.idio_te_contributions(
        te_factor_check, te_c, w_by_ticker, base["idio_var"]
    )
    idio_c_by_class = idio_c.rename(index=base["class_of"]).groupby(level=0).sum()

    cv_c, cvar_residual, _cvar_check, k_tail = attr.cvar_contributions(
        x, A_row, bench_annual_vec, base["F_mat"], active_exposure, opt.FF_FACTOR_COLS
    )

    return dict(
        label=cfg.label, config=cfg, feasible=True,
        x=x, x0=x0, rows=rows, tickers_row=tickers_row, tax_vec_full=tax_vec_full,
        active_exposure=active_exposure, sigma_mat=sigma_mat,
        te_factor=te_factor, te_idio=float(np.sqrt(max(te_idio_var_corr, 0.0))), te_total=te_total,
        cvar=cvar, active_scenarios=active_scenarios, k_tail=k_tail,
        tax_cost=tax_cost, turnover=turnover,
        te_contrib_factor=te_c_resc, te_contrib_idio_total=float(idio_c.sum()),
        te_idio_contrib_by_class=idio_c_by_class, te_total_attr=te_total_attr,
        cvar_contrib_factor=cv_c, cvar_residual=cvar_residual,
    )


@st.cache_data(show_spinner="Loading macro PCA factors...")
def load_macro_pca_factors():
    return pd.read_csv(os.path.join(FR_DATA, "macro_pca_factors.csv"), parse_dates=["Date"]).set_index("Date")


@st.cache_data(show_spinner="Regressing household tickers on macro PCA factors...")
def regress_tickers_macro(returns_wide, macro_pca, tickers):
    """HAC-OLS beta of each ACTUAL held ticker's own monthly return on the 9
    macro PCA factors -- same HAC-OLS convention as optimize_tax_cvar_mvo's
    regress_tickers() (just swapping the FF5 regressors for the macro PCA
    ones), run directly on this household's own 16 tickers. Deliberately NOT
    the generic 32-asset-class proxy universe
    asset_full14_betas_tstats_long.csv already has betas for (US_LC_Blend,
    Intl_LC_Value, etc.) -- proxying e.g. MSTR onto "US_LC_Growth" would
    misrepresent a concentrated single-stock's actual macro sensitivity, so
    this regresses each real ticker's own price history instead.
    Returns (betas: DataFrame index=ticker cols=macro factors, r2: Series).
    """
    opt, _ = get_optimizer_modules()
    betas, r2 = {}, {}
    for t in tickers:
        if t not in returns_wide.columns:
            continue
        ret = returns_wide[t].dropna() / 100.0
        data = macro_pca.join(ret.rename("Ret"), how="inner").dropna()
        n_obs = len(data)
        if n_obs < opt.MIN_OBS:
            continue
        x = sm.add_constant(data[MACRO_FACTOR_COLS])
        result = sm.OLS(data["Ret"], x).fit(cov_type="HAC", cov_kwds={"maxlags": opt.hac_maxlags(n_obs)})
        betas[t] = result.params.drop("const")
        r2[t] = result.rsquared
    return pd.DataFrame(betas).T, pd.Series(r2)


def weight_by_class(x, tickers_row, class_of):
    """Aggregate a weight vector (per (ticker,account) row) up to total
    weight per ASSET CLASS -- the same class_of map the optimizer's own
    idiosyncratic-risk-by-class rollup uses."""
    _, attr = get_optimizer_modules()
    w_by_ticker = attr.weight_by_ticker(x, tickers_row)
    return w_by_ticker.rename(index=lambda t: class_of.get(t, t)).groupby(level=0).sum()


def macro_exposure_by_class(x, tickers_row, macro_beta_wide, class_of):
    """Exact, additive decomposition of the portfolio's aggregate macro-factor
    exposure: exposure_k = sum_ticker weight_ticker * beta[ticker, k]. Rolled
    up to asset class (summed, not averaged) so it stacks cleanly on a bar
    chart -- rows sum to each factor's total portfolio exposure by
    construction, since this is a plain weighted sum, not a risk model."""
    _, attr = get_optimizer_modules()
    w_by_ticker = attr.weight_by_ticker(x, tickers_row)
    common = [t for t in w_by_ticker.index if t in macro_beta_wide.index]
    contrib = macro_beta_wide.loc[common].multiply(w_by_ticker.loc[common], axis=0)
    contrib["_class"] = [class_of.get(t, t) for t in common]
    return contrib.groupby("_class").sum()


def herfindahl_index(weights):
    """Sum of squared weights (HHI) -- a simple, standard concentration
    metric. 1/n for an equal-weighted n-position book; 1.0 for a single
    position. Scale-free, so it's comparable across configs regardless of
    household size."""
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w ** 2))


def bootstrap_te_distribution(result, n_boot=150, block=12, seed=42):
    """Block-bootstrap the monthly FF5 panel (preserving short-run
    autocorrelation better than an iid resample) to turn this configuration's
    single TE number into a distribution: resample the factor history,
    rebuild the same blended (long-run + 12mo-EWMA) covariance, and re-score
    the SAME optimized weights x against it. The idiosyncratic term is held
    fixed at its point estimate -- this bootstrap targets factor-covariance
    estimation uncertainty specifically, not the idio model.

    Caveat: resampling reorders months, so the EWMA half-life component is
    only an approximate robustness check under resampling, not exact --
    the long-run covariance component (half the blend) is order-independent
    and unaffected."""
    opt, _ = get_optimizer_modules()
    base = load_base_data()
    ff = base["ff"].sort_index().reset_index(drop=True)
    n = len(ff)
    db = result["active_exposure"]
    te_idio_var = result["te_idio"] ** 2

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    tes = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max(n - block, 1) + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        ff_boot = ff.iloc[idx].reset_index(drop=True)
        sigma_boot = opt.blended_factor_covariance(ff_boot, annualize=12).loc[opt.FF_FACTOR_COLS, opt.FF_FACTOR_COLS].values
        te_sq = float(db @ sigma_boot @ db) + te_idio_var
        tes[b] = np.sqrt(max(te_sq, 0.0))
    return tes
