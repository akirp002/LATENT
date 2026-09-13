"""
plot_quant_ff_factors.py

Plots the 5 Fama-French factors (Mkt-RF, SMB, HML, RMW, CMA) as a
small-multiples grid, same layout convention as plot_quant_pca_factors.py,
so the two can be eyeballed side by side.

Input:
  - Data/FF_plus_macro_workable.csv

Output:
  - Data/ff_factors.png
"""

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = r"C:\Users\ak\OneDrive\Desktop\Stuff\Python projects\Data\FF_plus_macro_workable.csv"
OUT_PATH = r"C:\Users\ak\OneDrive\Desktop\Stuff\Python projects\Data\ff_factors.png"

FF_FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
N_COLS = 3

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).set_index("Date")[FF_FACTOR_COLS]
    n_rows = -(-len(FF_FACTOR_COLS) // N_COLS)  # ceil division

    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(5 * N_COLS, 3 * n_rows), sharex=True)
    axes = axes.flatten()

    for ax, factor in zip(axes, FF_FACTOR_COLS):
        series = df[factor].dropna()
        ax.plot(series.index, series.values, color="#a4691f", linewidth=1.1)
        ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
        ax.set_title(factor, fontsize=10)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_ylabel("monthly return (%)", fontsize=8)
        ax.grid(alpha=0.25)

    for ax in axes[len(FF_FACTOR_COLS):]:
        ax.axis("off")

    fig.suptitle("Fama-French 5 Factors (monthly return, %)", fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")
