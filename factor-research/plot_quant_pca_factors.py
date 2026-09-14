"""
plot_quant_pca_factors.py

Plots every macro PCA factor from quant_model_macro_1.py
(Data/macro_pca_factors.csv) as a small-multiples grid, one panel per
factor, so trend/regime behavior is visible at a glance. Each panel gets a
zero line (factors are z-scored PCA scores, so 0 = sample average) and its
own y-scale since factors aren't on a common unit.

Input:
  - Data/macro_pca_factors.csv

Output:
  - Data/macro_pca_factors.png
"""

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = r"C:\Users\ak\OneDrive\Desktop\Stuff\Python projects\Data\macro_pca_factors.csv"
OUT_PATH = r"C:\Users\ak\OneDrive\Desktop\Stuff\Python projects\Data\macro_pca_factors.png"

N_COLS = 3

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).set_index("Date")
    factors = list(df.columns)
    n_rows = -(-len(factors) // N_COLS)  # ceil division

    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(5 * N_COLS, 3 * n_rows), sharex=True)
    axes = axes.flatten()

    for ax, factor in zip(axes, factors):
        series = df[factor].dropna()
        ax.plot(series.index, series.values, color="#1f6f6b", linewidth=1.1)
        ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
        ax.set_title(factor.replace("_", " "), fontsize=10)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(alpha=0.25)

    # hide any unused subplot slots
    for ax in axes[len(factors):]:
        ax.axis("off")

    fig.suptitle("Macro PCA Factors (category-level, z-scored)", fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")
