r"""
generate_showcase_charts.py

Produces the PNG results-showcase for docs/ from data already committed to
this repo -- nothing here is fabricated on the spot, every chart reads the
same CSVs the pipeline scripts wrote. Palette/marks follow the studio's
data-viz method: sequential blue for single-series magnitude, the first
three validated categorical slots (blue/orange/aqua) for multi-series.

Output: docs/assets/*.png (dpi=200, light chart surface)
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_DATA = os.path.join(ROOT, "factor-research", "Data")
PORT_DATA = os.path.join(ROOT, "portfolio-benchmarking", "Data")
OUT_DIR = os.path.join(ROOT, "docs", "assets")
os.makedirs(OUT_DIR, exist_ok=True)

# --- Palette (validated categorical order + chrome, light mode) -----------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
CRITICAL = "#d03b3b"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 1.0,
    "axes.axisbelow": True,
    "font.size": 11,
})


def style_axes(ax, hide_top_right=True):
    if hide_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="x", linewidth=1.0, color=GRID)
    ax.grid(axis="y", visible=False)


def add_titles(fig, title, subtitle, top=0.86):
    """Headline + subtitle as two separate figure-level texts with a real
    gap between them, instead of stacking ax.set_title + a transAxes label
    (which collide at this font size)."""
    fig.suptitle(title, x=0.03, y=0.985, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(0.03, 0.925, subtitle, ha="left", va="top", fontsize=9.5, color=INK_SECONDARY)
    fig.subplots_adjust(top=top)


# ===========================================================================
# 1. VIF by factor -- single-series magnitude, sequential blue, threshold line
# ===========================================================================
def chart_vif():
    df = pd.read_csv(os.path.join(PY_DATA, "asset_full14_tstats_wide.csv"))
    # VIF wasn't saved to CSV by vif_check.py (console-only) -- recompute here
    # from the same 14-factor panel it used, so the chart matches the repo's
    # actual data exactly.
    ff = pd.read_csv(os.path.join(PY_DATA, "FF_plus_macro_workable.csv"), parse_dates=["Date"]).set_index("Date")
    macro = pd.read_csv(os.path.join(PY_DATA, "macro_pca_factors.csv"), parse_dates=["Date"]).set_index("Date")
    ff5 = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    panel = ff[ff5].join(macro, how="inner").dropna()

    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
    X = add_constant(panel)
    vifs = {col: variance_inflation_factor(X.values, i) for i, col in enumerate(X.columns) if col != "const"}
    vif_s = pd.Series(vifs).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=200)
    y = np.arange(len(vif_s))
    ax.barh(y, vif_s.values, height=0.6, color=BLUE, zorder=3)
    ax.axvline(5.0, color=INK_MUTED, linewidth=1, linestyle=(0, (3, 2)), zorder=2)
    ax.text(5.05, len(vif_s) - 0.4, "VIF = 5 (yellow flag)", color=INK_MUTED, fontsize=9, va="top")

    for yi, v in zip(y, vif_s.values):
        ax.text(v + 0.07, yi, f"{v:.2f}", va="center", ha="left", fontsize=9.5, color=INK_SECONDARY)

    ax.set_yticks(y)
    ax.set_yticklabels(vif_s.index, fontsize=9.5)
    ax.set_xlim(0, max(6.0, vif_s.max() * 1.15))
    ax.set_xlabel("Variance Inflation Factor")
    style_axes(ax)
    fig.tight_layout()
    add_titles(fig, "No collinearity problem across the 14-factor set",
               "VIF per factor (5 FF + 9 macro PCA) — all well under the conventional flag threshold")
    fig.savefig(os.path.join(OUT_DIR, "vif_by_factor.png"), facecolor=SURFACE)
    plt.close(fig)
    print("saved vif_by_factor.png")


# ===========================================================================
# 2. Empirical-Bayes tau_hat by factor -- single-series magnitude
# ===========================================================================
def chart_tau_shrinkage():
    df = pd.read_csv(os.path.join(PY_DATA, "asset_full14_betas_tstats_long.csv"))
    df = df[df["Term"] != "Alpha"].copy()
    df["SE"] = (df["Beta"] / df["T_Stat_HAC"]).abs()

    rows = []
    for factor, g in df.groupby("Term"):
        beta_var = g["Beta"].var(ddof=1)
        mean_se2 = (g["SE"] ** 2).mean()
        tau2 = max(0.0, beta_var - mean_se2)
        rows.append({"Factor": factor, "tau_hat": np.sqrt(tau2)})
    tau_s = pd.DataFrame(rows).set_index("Factor")["tau_hat"].sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=200)
    y = np.arange(len(tau_s))
    colors = [CRITICAL if v < 1e-6 else BLUE for v in tau_s.values]
    ax.barh(y, tau_s.values, height=0.6, color=colors, zorder=3)

    for yi, v in zip(y, tau_s.values):
        label = "≈ 0 (pure noise)" if v < 1e-6 else f"{v:.3f}"
        ax.text(v + 0.012, yi, label, va="center", ha="left", fontsize=9.5, color=INK_SECONDARY)

    ax.set_yticks(y)
    ax.set_yticklabels(tau_s.index, fontsize=9.5)
    ax.set_xlim(0, tau_s.max() * 1.35)
    ax.set_xlabel(r"$\hat{\tau}$  (empirical-Bayes prior SD, cross-sectional)")
    style_axes(ax)
    fig.tight_layout()
    add_titles(fig, "Which factors carry real signal vs. pure estimation noise",
               "Normal–normal shrinkage: τ̂² = max(0, cross-sectional Var(β̂) − mean SE²), per factor across 33 assets")
    fig.savefig(os.path.join(OUT_DIR, "tau_shrinkage_by_factor.png"), facecolor=SURFACE)
    plt.close(fig)
    print("saved tau_shrinkage_by_factor.png")


# ===========================================================================
# 3. HAC vs OLS significance count -- 2-series grouped bar, equities only
# ===========================================================================
def chart_hac_significance():
    df = pd.read_csv(os.path.join(PY_DATA, "asset_full14_betas_tstats_long.csv"))
    df = df[df["Term"] != "Alpha"].copy()

    sig_ols = df.assign(sig=df["T_Stat_OLS"].abs() > 2).groupby("Term")["sig"].sum()
    sig_hac = df.assign(sig=df["T_Stat_HAC"].abs() > 2).groupby("Term")["sig"].sum()
    order = sig_hac.sort_values(ascending=False).index
    sig_ols, sig_hac = sig_ols[order], sig_hac[order]

    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=200)
    x = np.arange(len(order))
    w = 0.34
    ax.bar(x - w / 2, sig_ols.values, width=w, color=BLUE, zorder=3, label="OLS (plain SEs)")
    ax.bar(x + w / 2, sig_hac.values, width=w, color=ORANGE, zorder=3, label="HAC (Newey–West SEs)")

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=38, ha="right", fontsize=8.5)
    ax.set_ylabel("Assets with |t| > 2  (of 33)")
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    style_axes(ax)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    add_titles(fig, "HAC correction barely moves the picture",
               "Count of assets with a significant beta, per factor — 16 OLS-sig→HAC-insig vs. 10 the other way, out of 462 pairs",
               top=0.82)
    fig.savefig(os.path.join(OUT_DIR, "hac_vs_ols_significance.png"), facecolor=SURFACE)
    plt.close(fig)
    print("saved hac_vs_ols_significance.png")


# ===========================================================================
# 4. Demo portfolio wealth curves -- 3-series line chart, direct end-labels
# ===========================================================================
def chart_portfolio_wealth():
    df = pd.read_csv(os.path.join(PORT_DATA, "portfolio_active_returns.csv"), parse_dates=["Date"]).set_index("Date")
    wealth = (1 + df[["Portfolio_Full", "Portfolio_ExConcentrated", "Benchmark"]]).cumprod()
    wealth = wealth / wealth.iloc[0]

    fig, ax = plt.subplots(figsize=(10.5, 6), dpi=200)
    series = [
        ("Portfolio_Full", "Portfolio (full, w/ PLTR)", BLUE),
        ("Portfolio_ExConcentrated", "Portfolio (ex-concentrated)", ORANGE),
        ("Benchmark", "Benchmark (60% Mkt-RF / 40% RMW)", AQUA),
    ]
    for col, label, color in series:
        ax.plot(wealth.index, wealth[col], color=color, linewidth=2, zorder=3, label=label)
        ax.plot(wealth.index[-1], wealth[col].iloc[-1], "o", color=color, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)

    # End-labels: when two series finish close together (here, ex-concentrated
    # vs benchmark), stacking them at the same y collides -- stagger by y-rank
    # instead of letting matplotlib overlap the text.
    end_vals = sorted(((wealth[col].iloc[-1], col, color) for col, _, color in series), reverse=True)
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    min_gap = 0.045 * y_range
    placed = []
    for val, col, color in end_vals:
        y = val
        if placed and (placed[-1] - y) < min_gap:
            y = placed[-1] - min_gap
        placed.append(y)
        ax.annotate(f"{val:.2f}x", (wealth.index[-1], y), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9.5,
                    color=INK_SECONDARY, fontweight="bold")

    ax.set_ylabel("Growth of $1 (common overlap window)")
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    style_axes(ax)
    fig.tight_layout()
    add_titles(fig, "Sample portfolio vs. strategic benchmark — growth of $1",
               "Fabricated demo portfolio (generate_sample_holdings.py) — not real holdings")
    fig.savefig(os.path.join(OUT_DIR, "portfolio_wealth_curve.png"), facecolor=SURFACE)
    plt.close(fig)
    print("saved portfolio_wealth_curve.png")


# ===========================================================================
# 5. Tracking error comparison -- ex-ante vs ex-post, full vs ex-concentrated
# ===========================================================================
def chart_tracking_error():
    ret = pd.read_csv(os.path.join(PORT_DATA, "portfolio_active_returns.csv"), parse_dates=["Date"]).set_index("Date")
    ANNUALIZE = np.sqrt(12)
    te_post_full = ret["Active_Full"].std() * ANNUALIZE
    te_post_exconc = ret["Active_ExConcentrated"].std() * ANNUALIZE

    exposures = pd.read_csv(os.path.join(PORT_DATA, "portfolio_factor_exposures.csv"), index_col=0)
    ff = pd.read_csv(os.path.join(PY_DATA, "FF_plus_macro_workable.csv"), parse_dates=["Date"]).set_index("Date")
    ff5 = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    sigma = (ff[ff5] / 100.0).cov().values * 12
    db_full = exposures["Delta_Full"].values
    db_exconc = exposures["Delta_ExConcentrated"].values
    te_ante_full = np.sqrt(db_full @ sigma @ db_full)
    te_ante_exconc = np.sqrt(db_exconc @ sigma @ db_exconc)

    groups = ["Full portfolio\n(w/ PLTR)", "Ex-concentrated\nportfolio"]
    ante = [te_ante_full, te_ante_exconc]
    post = [te_post_full, te_post_exconc]

    fig, ax = plt.subplots(figsize=(7.5, 5.8), dpi=200)
    x = np.arange(len(groups))
    w = 0.32
    b1 = ax.bar(x - w / 2, ante, width=w, color=BLUE, zorder=3, label="Ex-ante (analytic)")
    b2 = ax.bar(x + w / 2, post, width=w, color=ORANGE, zorder=3, label="Ex-post (realized)")
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.004, f"{h:.1%}", ha="center",
                    fontsize=9.5, color=INK_SECONDARY)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel("Annualized tracking error")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    style_axes(ax)
    fig.tight_layout()
    add_titles(fig, "A concentrated position dominates tracking error",
               "Dropping the single concentrated position cuts realized TE from ~36% to ~4%")
    fig.savefig(os.path.join(OUT_DIR, "tracking_error_comparison.png"), facecolor=SURFACE)
    plt.close(fig)
    print("saved tracking_error_comparison.png")


if __name__ == "__main__":
    chart_vif()
    chart_tau_shrinkage()
    chart_hac_significance()
    chart_portfolio_wealth()
    chart_tracking_error()
    print(f"\nAll charts saved to {OUT_DIR}")
