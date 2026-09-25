r"""
portfolio_risk_attribution.py

Decomposes each rebalance path's RISK -- not just its headline number -- onto
the five Fama-French factors, and expresses the portfolio's factor tilts in
cross-sectional z-scores. Pure analysis: reads the optimizer's saved trade
proposals and recomputes attribution from the same betas and scenarios the
optimizer itself used. Nothing here re-optimizes.

Three decompositions, one per question the headline metrics can't answer:

1. TRACKING-ERROR RISK CONTRIBUTION (Euler decomposition, factor + idiosyncratic)
   --------------------------------------------------------------------------
   TE(x) = sqrt(a' Sigma a),  a = beta_portfolio(x) - beta_benchmark
   TE is positively homogeneous of degree 1 in a, so Euler's theorem gives an
   exact, additive split across factors:

       CTR_k = a_k * (Sigma a)_k / TE       and       sum_k CTR_k = TE

   CTR_k is factor k's contribution to total tracking error, in the same units
   as TE itself. This answers "WHICH factor bet is our tracking error actually
   made of?" -- a portfolio can cut total TE while making a single factor's
   contribution worse, and the headline number hides that.

   Note CTR_k can be NEGATIVE: a factor whose active exposure hedges the rest
   of the book subtracts from total TE. That is a real effect, not an artifact.

   THIS IS FACTOR-ONLY. It is also exactly what optimize_tax_cvar_mvo.py
   minimizes (see that module's docstring) -- it is a deliberate SIMPLIFICATION
   that assumes each ticker's return is fully explained by its FF5 beta, which
   is false: every ticker's HAC OLS regression leaves a residual, and for a
   35%-weight single stock like MSTR that residual is most of its risk, not a
   rounding error. So a SECOND, structural ("Barra-style") decomposition is
   added on top, without touching the optimizer's own objective (a bigger,
   separate design change this script doesn't make):

       idio_var_t   = each ticker's annualized OLS residual variance
                       (mse_resid * 12 -- see regress_tickers), i.e. the
                       variance the 5 FF factors do NOT explain
       w_t(x)       = ticker t's TOTAL household weight (summed across
                       accounts) under path x -- the benchmark holds zero
                       idiosyncratic exposure, so this IS the ticker's active
                       idiosyncratic exposure
       Var_idio(x)  = sum_t  w_t(x)^2 * idio_var_t        (diagonal --
                       residuals assumed uncorrelated across tickers, the
                       standard structural-risk-model simplifying assumption)
       TE_total(x)  = sqrt(TE_factor(x)^2 + Var_idio(x))

   Because Var_factor and Var_idio are the two orthogonal blocks of one
   quadratic form (z'Mz with z=[a; w], M=block_diag(Sigma, diag(idio_var))),
   Euler's theorem applies to the COMBINED total exactly the same way,
   rescaled onto TE_total instead of TE_factor:

       CTR_k(factor, rescaled) = CTR_k * (TE_factor / TE_total)
       CTR_t(idio)             = w_t(x)^2 * idio_var_t / TE_total
       sum_k CTR_k(factor,resc.) + sum_t CTR_t(idio)  =  TE_total   exactly

   This is verified as an assertion in the script, same as the factor-only
   split was. Per-ticker idio contributions are also rolled up to
   ASSET_CLASS (optimize_tax_cvar_mvo.ASSET_CLASS) by plain summation --
   exact, not an approximation, because Euler contributions under a diagonal
   idio block are already additive across tickers with zero cross terms.

2. CVaR RISK CONTRIBUTION (tail-conditional factor attribution)
   -------------------------------------------------------------
   Active CVaR is the mean active return over the worst 5% of the overlapping
   12-month scenarios -- the same scenario set and tail convention
   optimize_tax_cvar_mvo.py constrains. Within those tail scenarios the active
   return is linear in the active factor exposure:

       Active_s = sum_k a_k * F_k,s + idiosyncratic_s

   where F_k,s is factor k's summed return over window s. Averaging over just
   the tail scenarios T gives an additive split:

       CVaR_contribution_k = a_k * mean_{s in T}( F_k,s )
       residual            = CVaR - sum_k CVaR_contribution_k

   The residual is what the five factors do NOT explain about the tail:
   idiosyncratic risk (here, overwhelmingly MSTR's single-stock risk), which
   is exactly the part a factor-based benchmark cannot hedge. A path that cuts
   CVaR by shrinking the residual is doing something categorically different
   from one that cuts it by neutralizing a factor bet.

   NOTE: unlike the TE idiosyncratic split above, this residual is NOT further
   broken out per ticker/asset-class here. Doing that rigorously would need
   each ticker's own idiosyncratic return summed over the SAME 12-month tail
   windows (not just its annualized residual variance, which is all the TE
   split needs) -- a separate scenario-construction step this pass doesn't
   add. Stated as a limitation, not silently skipped.

   IMPORTANT: this is a TAIL-conditional attribution, so it does not match the
   TE decomposition above and is not supposed to. TE is a whole-distribution
   (symmetric, second-moment) measure; CVaR looks only at the left tail. A
   factor can be a large TE contributor and a small CVaR contributor if its
   active exposure simply did not move against the book when the book was
   losing.

3. FACTOR EXPOSURE Z-SCORES (cross-sectional standardization)
   ------------------------------------------------------------
   Raw betas are not comparable across factors -- a 0.9 Mkt-RF beta and a 0.9
   CMA beta are wildly different statements. Each factor's betas are therefore
   standardized across the HOLDINGS UNIVERSE (the tickers actually in this
   book):

       z_ticker[t,k] = (beta[t,k] - mean_t beta[.,k]) / std_t beta[.,k]
       z_portfolio[k] = sum_rows x_row * z_ticker[ticker(row), k]

   So z = +1.0 means "this book is tilted one cross-sectional standard
   deviation more toward factor k than the average thing it holds." The
   benchmark's own exposure is standardized on the same scale for comparison,
   and the ACTIVE z-score (portfolio - benchmark) is what the tracking-error
   objective is actually pushing toward zero.

   Caveat, stated plainly: the universe is the ~16 tickers in this household's
   book, not a broad market cross-section. These z-scores say "tilted relative
   to what this family already owns," which is the relevant question for a
   reweighting-only mandate, but they are NOT Barra-style market-wide scores
   and should not be read as such.

Inputs (Data\, all produced upstream):
  - holdings_household_clean.csv, holdings_household_monthly_returns.csv
  - benchmark_mvo_weights.csv, benchmark_wealth_curve.csv
  - trade_proposals_path{A,B,C,D}.csv / trade_proposals.csv
  - ..\factor-research\Data\FF_plus_macro_workable.csv

Output:
  - Data\risk_attribution.json   consumed by generate_rebalance_dashboard.py
"""

import json
import os

import numpy as np
import pandas as pd

from optimize_tax_cvar_mvo import (
    ASSET_CLASS,
    CVAR_ALPHA_TAIL,
    DATA_DIR,
    FF_FACTOR_COLS,
    WINDOW,
    load_ff,
    regress_tickers,
)

OUT_PATH = os.path.join(DATA_DIR, "risk_attribution.json")


def build_aligned_scenarios(returns_wide, bench_wealth, ff, tickers, window=WINDOW):
    """Overlapping `window`-month scenarios for tickers, the benchmark AND the
    factors, all on one common index so the CVaR tail identified from the
    portfolio's active return can be used to average the factors over exactly
    the same windows.

    Returns (A_tick, bench_annual, F_fac) where A_tick is ticker x scenario,
    F_fac is factor x scenario, and bench_annual is a scenario-indexed Series.
    """
    ret = returns_wide[tickers].dropna(how="any") / 100.0
    bench = bench_wealth["Benchmark_Return"]
    combined = ret.join(bench, how="inner").join(ff, how="inner").dropna()
    if len(combined) < window:
        raise RuntimeError(
            f"Only {len(combined)} months of common history across tickers + benchmark "
            f"+ factors (< {window}) -- cannot build a scenario."
        )

    n_scen = len(combined) - window + 1
    A = np.zeros((len(tickers), n_scen))
    F = np.zeros((len(FF_FACTOR_COLS), n_scen))
    bench_annual = np.zeros(n_scen)
    for s in range(n_scen):
        chunk = combined.iloc[s: s + window]
        A[:, s] = chunk[tickers].sum().values
        F[:, s] = chunk[FF_FACTOR_COLS].sum().values
        bench_annual[s] = chunk["Benchmark_Return"].sum()

    print(f"  [scenarios] {n_scen} overlapping {window}-month windows, "
          f"{combined.index.min().date()} -> {combined.index.max().date()}")
    return (pd.DataFrame(A, index=tickers),
            pd.Series(bench_annual),
            pd.DataFrame(F, index=FF_FACTOR_COLS))


def te_contributions(active_exposure, sigma_mat, factor_cols):
    """Euler decomposition of TE. Returns (contrib Series, total TE).
    sum(contrib) == TE exactly (to floating point)."""
    a = np.asarray(active_exposure, dtype=float)
    var = float(a @ sigma_mat @ a)
    te = float(np.sqrt(max(var, 0.0)))
    if te <= 0:
        return pd.Series(0.0, index=factor_cols), 0.0
    contrib = a * (sigma_mat @ a) / te
    return pd.Series(contrib, index=factor_cols), te


def idio_te_contributions(te_factor, te_factor_contrib, weight_by_ticker, idio_var_by_ticker):
    """Extends the factor-only Euler split to a combined factor+idiosyncratic
    TE_total. See the module docstring's item 1 for the derivation. Returns
    (te_factor_contrib_rescaled, idio_contrib_by_ticker, te_total, idio_var_total).
    sum(te_factor_contrib_rescaled) + sum(idio_contrib_by_ticker) == te_total exactly.
    """
    idio_var_total = float((weight_by_ticker ** 2 * idio_var_by_ticker.reindex(
        weight_by_ticker.index).fillna(0.0)).sum())
    te_total = float(np.sqrt(te_factor ** 2 + idio_var_total))
    if te_total <= 0:
        return te_factor_contrib * 0.0, weight_by_ticker * 0.0, 0.0, 0.0
    scale = te_factor / te_total
    idio_contrib = (weight_by_ticker ** 2 * idio_var_by_ticker.reindex(
        weight_by_ticker.index).fillna(0.0)) / te_total
    return te_factor_contrib * scale, idio_contrib, te_total, idio_var_total


def weight_by_ticker(x, tickers_row):
    """Aggregate a path's per-(ticker,account) weight vector up to one total
    weight per unique ticker -- the benchmark holds zero idiosyncratic
    exposure, so this total IS each ticker's active idiosyncratic exposure."""
    return pd.Series(x, index=tickers_row).groupby(level=0).sum()


def cvar_contributions(x, A_row, bench_annual_vec, F_mat, active_exposure,
                       factor_cols, alpha_tail=CVAR_ALPHA_TAIL):
    """Tail-conditional factor attribution of Active CVaR.

    The tail is identified from the ACTUAL scenario active returns (so it is
    the same tail the optimizer's constraint sees), then each factor's mean
    realized return over just those scenarios is multiplied by the active
    exposure to that factor. The leftover is idiosyncratic.
    """
    active = (A_row.T @ x) - bench_annual_vec  # scenario active returns
    n_scen = len(active)
    k = max(1, int(np.ceil(alpha_tail * n_scen)))
    tail_idx = np.argsort(active)[:k]  # k worst active outcomes
    cvar = float(active[tail_idx].mean())  # as a RETURN; negative = bad

    f_tail_mean = F_mat.values[:, tail_idx].mean(axis=1)  # (5,)
    contrib = np.asarray(active_exposure, dtype=float) * f_tail_mean
    residual = cvar - float(contrib.sum())
    return pd.Series(contrib, index=factor_cols), residual, cvar, int(k)


def zscore_frame(beta_wide, factor_cols):
    """Cross-sectional standardization of each factor's betas over the
    holdings universe. Returns (z_by_ticker, mean, std)."""
    mean = beta_wide[factor_cols].mean()
    std = beta_wide[factor_cols].std(ddof=1).replace(0.0, np.nan)
    return (beta_wide[factor_cols] - mean) / std, mean, std


def load_path_weights(rows, label):
    """Align a saved trade-proposal file's Proposed_Weight back onto `rows`
    order. Returns None when that path has no trade file (infeasible)."""
    for fname in (f"trade_proposals_path{label}.csv", "trade_proposals.csv"):
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        merged = rows[["Account", "Ticker"]].merge(
            df[["Account", "Ticker", "Proposed_Weight"]], on=["Account", "Ticker"], how="left")
        if merged["Proposed_Weight"].isna().any():
            raise RuntimeError(f"{fname} is missing rows present in holdings -- refusing to guess.")
        return merged["Proposed_Weight"].values
    return None


if __name__ == "__main__":
    holdings_all = pd.read_csv(os.path.join(DATA_DIR, "holdings_household_clean.csv"))
    returns_wide = pd.read_csv(os.path.join(DATA_DIR, "holdings_household_monthly_returns.csv"),
                                parse_dates=["Date"]).set_index("Date")
    ff, rf = load_ff()
    bench_weights = pd.read_csv(os.path.join(DATA_DIR, "benchmark_mvo_weights.csv"), index_col=0)["Weight"]
    bench_wealth = pd.read_csv(os.path.join(DATA_DIR, "benchmark_wealth_curve.csv"),
                                parse_dates=["Date"]).set_index("Date")

    tickers = sorted(holdings_all["Ticker"].unique())
    print(f"Regressing {len(tickers)} tickers on FF5 (HAC SEs)...")
    beta_wide, idio_var_by_ticker = regress_tickers(returns_wide, ff, rf, tickers)
    usable = list(beta_wide.index)

    holdings = holdings_all[holdings_all["Ticker"].isin(usable)].copy()
    holdings["Weight_Household"] = holdings["Market_Value"] / holdings_all["Market_Value"].sum()
    rows = holdings.reset_index(drop=True)
    x0 = rows["Weight_Household"].values
    tickers_row = rows["Ticker"].values

    beta_matrix = beta_wide.loc[tickers_row].values  # n x 5
    beta_bench_vec = pd.Series(0.0, index=FF_FACTOR_COLS)
    beta_bench_vec.update(bench_weights)
    beta_bench_vec = beta_bench_vec.reindex(FF_FACTOR_COLS).fillna(0.0).values
    sigma_mat = (ff.cov() * 12).loc[FF_FACTOR_COLS, FF_FACTOR_COLS].values

    print("Building aligned ticker / benchmark / factor scenarios...")
    A_scen, bench_annual, F_mat = build_aligned_scenarios(returns_wide, bench_wealth, ff, usable)
    A_row = A_scen.loc[tickers_row].values
    bench_annual_vec = bench_annual.values

    z_by_ticker, z_mean, z_std = zscore_frame(beta_wide, FF_FACTOR_COLS)
    z_row = z_by_ticker.loc[tickers_row].values  # n x 5
    z_bench = (beta_bench_vec - z_mean.values) / z_std.values

    # Path C and Path D are the two MSTR-glide-path trade proposals from
    # optimize_tax_cvar_mvo.py; SAA is optimize_saa_mvo.py's independent
    # sleeve-covariance solve. FF5 risk attribution applies to ANY proposed
    # weight vector regardless of which objective produced it, so SAA's
    # proposed weights get the same factor/CVaR/z-score decomposition as C
    # and D -- this is attribution of what the SAA path's weights happen to
    # expose the household to on the FF5 benchmark, not a claim that SAA's
    # own objective optimizes for any of these numbers.
    labels = ["before"] + [l for l in ["C", "D", "SAA", "SAA_3pct"]
                            if os.path.exists(os.path.join(DATA_DIR, f"trade_proposals_path{l}.csv"))]

    # Asset-class order: by CURRENT total household weight, descending, so the
    # dashboard's class charts read biggest-exposure-first without re-sorting.
    class_of = {t: ASSET_CLASS.get(t, t) for t in usable}
    w0_by_ticker = weight_by_ticker(x0, tickers_row)
    class_weight0 = w0_by_ticker.rename(index=class_of).groupby(level=0).sum()
    asset_classes = list(class_weight0.sort_values(ascending=False).index)

    out = {
        "factors": FF_FACTOR_COLS,
        "benchmark_weights": {k: float(v) for k, v in zip(FF_FACTOR_COLS, beta_bench_vec)},
        "n_scenarios": int(F_mat.shape[1]),
        "cvar_alpha_tail": CVAR_ALPHA_TAIL,
        "zscore_universe_n": int(len(beta_wide)),
        "zscore_mean": {k: float(v) for k, v in z_mean.items()},
        "zscore_std": {k: float(v) for k, v in z_std.items()},
        "benchmark_z": {k: float(v) for k, v in zip(FF_FACTOR_COLS, z_bench)},
        "asset_classes": asset_classes,
        "ticker_class": class_of,
        "paths": {},
    }

    print("\n%-7s %9s %9s %9s   %s" % ("path", "TE(fac)", "TE(idio)", "TE(tot)", "TE contribution by factor"))
    for label in labels:
        x = x0 if label == "before" else load_path_weights(rows, label)
        if x is None:
            continue
        beta_port = beta_matrix.T @ x
        active = beta_port - beta_bench_vec

        te_c, te = te_contributions(active, sigma_mat, FF_FACTOR_COLS)
        w_by_ticker = weight_by_ticker(x, tickers_row)
        te_c_resc, idio_c, te_total, idio_var_total = idio_te_contributions(
            te, te_c, w_by_ticker, idio_var_by_ticker)
        idio_c_by_class = idio_c.rename(index=class_of).groupby(level=0).sum()
        class_weight = w_by_ticker.rename(index=class_of).groupby(level=0).sum()

        cv_c, resid, cvar, k_tail = cvar_contributions(
            x, A_row, bench_annual_vec, F_mat, active, FF_FACTOR_COLS)
        z_port = z_row.T @ x

        out["paths"][label] = {
            "te": float(te),                    # factor-only (matches the optimizer's own objective)
            "te_idio": float(np.sqrt(idio_var_total)),
            "te_total": float(te_total),
            "cvar": float(cvar),
            "tail_scenarios": k_tail,
            "exposure": {k: float(v) for k, v in zip(FF_FACTOR_COLS, beta_port)},
            "active_exposure": {k: float(v) for k, v in zip(FF_FACTOR_COLS, active)},
            "te_contrib": {k: float(v) for k, v in te_c.items()},                # sums to te (factor-only)
            "te_contrib_of_total": {k: float(v) for k, v in te_c_resc.items()},  # sums to te_total, w/ idio below
            "te_idio_contrib_by_class": {k: float(v) for k, v in idio_c_by_class.items()},
            "cvar_contrib": {k: float(v) for k, v in cv_c.items()},
            "cvar_residual": float(resid),
            "z_portfolio": {k: float(v) for k, v in zip(FF_FACTOR_COLS, z_port)},
            "z_active": {k: float(v) for k, v in zip(FF_FACTOR_COLS, z_port - z_bench)},
            "class_weight": {k: float(class_weight.get(k, 0.0)) for k in asset_classes},
        }

        # Identity checks: both Euler splits must reconstruct their totals exactly.
        assert abs(te_c.sum() - te) < 1e-9, f"TE factor-only Euler split broken for {label}"
        assert abs(te_c_resc.sum() + idio_c.sum() - te_total) < 1e-9, f"TE total Euler split broken for {label}"
        print("%-7s %8.2f%% %8.2f%% %8.2f%%   %s" % (
            label, 100 * te, 100 * np.sqrt(idio_var_total), 100 * te_total,
            "  ".join(f"{k}={100*v:+.2f}%" for k, v in te_c.items())))

    print("\n%-7s   %s" % ("path", "idiosyncratic TE contribution by asset class"))
    for label, p in out["paths"].items():
        print("%-7s   %s" % (
            label, "  ".join(f"{k}={100*v:+.2f}%" for k, v in p["te_idio_contrib_by_class"].items())))

    print("\n%-7s %9s   %s" % ("path", "residual", "CVaR contribution by factor"))
    for label, p in out["paths"].items():
        print("%-7s %8.2f%%   %s" % (
            label, 100 * p["cvar_residual"],
            "  ".join(f"{k}={100*v:+.2f}%" for k, v in p["cvar_contrib"].items())))

    print("\n%-7s   %s" % ("path", "active factor exposure, z-score vs. holdings universe"))
    print("%-7s   %s" % ("bench", "  ".join(f"{k}={v:+.2f}" for k, v in out["benchmark_z"].items())))
    for label, p in out["paths"].items():
        print("%-7s   %s" % (label, "  ".join(f"{k}={v:+.2f}" for k, v in p["z_active"].items())))

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_PATH}")
