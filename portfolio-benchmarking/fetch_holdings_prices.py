r"""
fetch_holdings_prices.py

Downloads monthly adjusted-close prices for every unique ticker in the
sample holdings (Data\holdings_tickers.txt) via yfinance, converts to
monthly percent returns (same convention as ..\factor-research\pull_yfinance_data.py
-- percent, not decimal, to match the FF factor file's scale before the
regression step divides everything by 100), and saves one wide CSV.

Output: Data\holdings_monthly_returns.csv
"""

import os

import pandas as pd
import yfinance as yf

OUT_DIR = os.path.join(os.path.dirname(__file__), "Data")
TICKERS_PATH = os.path.join(OUT_DIR, "holdings_tickers.txt")
START = "1990-01-01"


def fetch_monthly_returns(tickers, start=START):
    print(f"Downloading {len(tickers)} tickers from yfinance...")
    raw = yf.download(
        tickers, start=start, interval="1mo",
        auto_adjust=True, progress=False, group_by="ticker", threads=True,
    )

    returns = {}
    missing = []
    for t in tickers:
        try:
            close = raw[t]["Close"].dropna()
        except (KeyError, TypeError):
            missing.append(t)
            continue
        if close.empty:
            missing.append(t)
            continue
        returns[t] = close.pct_change() * 100
        print(f"  fetched {t}: {close.index.min().date()} -> {close.index.max().date()} ({len(close)} obs)")

    if missing:
        print("\nNo data returned for:", missing)

    df = pd.DataFrame(returns)
    df.index = df.index.to_period("M").to_timestamp(how="start")
    df.index.name = "Date"
    return df.reset_index()


if __name__ == "__main__":
    with open(TICKERS_PATH) as f:
        tickers = [line.strip() for line in f if line.strip()]

    returns_df = fetch_monthly_returns(tickers)
    out_path = os.path.join(OUT_DIR, "holdings_monthly_returns.csv")
    returns_df.to_csv(out_path, index=False)

    print(f"\nSaved -> {out_path}")
    print(f"Shape: {returns_df.shape}")
