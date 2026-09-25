r"""
fetch_household_holdings_prices.py

Same as fetch_holdings_prices.py, pointed at the fabricated household
tickers (Data\holdings_household_tickers.txt, from
generate_sample_real_holdings.py) instead of the pipeline-A sample
holdings. The TICKERS are real, liquid, publicly-traded securities (so the
price history and covariance structure are realistic) -- only the
QUANTITIES, COST BASIS, and GAINS attached to them (in
holdings_household_clean.csv) are fabricated.

Output: Data\holdings_household_monthly_returns.csv (wide, Date + one
column per ticker, values in PERCENT -- same convention as the rest of the
pipeline).

If yfinance/network access is unavailable, falls back to a documented,
seeded synthetic random walk per ticker so the rest of the pipeline still
has something internally consistent to run against. This fallback is
clearly labeled in stdout and never silently mistaken for real prices.
"""

import os

import numpy as np
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "Data")
TICKERS_PATH = os.path.join(OUT_DIR, "holdings_household_tickers.txt")
START = "1998-01-01"
SYNTHETIC_SEED = 20240601  # arbitrary, fixed for reproducibility


def fetch_monthly_returns(tickers, start=START):
    import yfinance as yf

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
    if not returns:
        raise RuntimeError("yfinance returned no usable data for any ticker.")

    df = pd.DataFrame(returns)
    df.index = df.index.to_period("M").to_timestamp(how="start")
    df.index.name = "Date"
    return df.reset_index()


def synthetic_monthly_returns(tickers, start=START, end=None, ann_vol=0.20, ann_drift=0.08):
    """*** FABRICATED FALLBACK ***
    A seeded random walk per ticker, monthly percent returns, used only if
    yfinance/network access is unavailable. Not real price history for any
    of these tickers -- purely a stand-in so the rest of the pipeline
    (which needs SOME overlapping monthly return history) still produces
    internally-consistent output.
    """
    print("*** FALLBACK: generating FABRICATED synthetic monthly returns "
          "(no network access to yfinance) ***")
    rng = np.random.default_rng(SYNTHETIC_SEED)
    dates = pd.date_range(start=start, end=end or pd.Timestamp.today().normalize(), freq="MS")
    monthly_vol = ann_vol / np.sqrt(12)
    monthly_drift = ann_drift / 12
    data = {}
    for i, t in enumerate(tickers):
        # de-correlate slightly across tickers via a per-ticker seed offset
        r = np.random.default_rng(SYNTHETIC_SEED + i)
        data[t] = r.normal(loc=monthly_drift, scale=monthly_vol, size=len(dates)) * 100
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df.reset_index()


if __name__ == "__main__":
    with open(TICKERS_PATH) as f:
        tickers = [line.strip() for line in f if line.strip()]

    try:
        returns_df = fetch_monthly_returns(tickers)
    except Exception as exc:
        print(f"\nyfinance fetch failed ({exc!r}); falling back to synthetic data.")
        returns_df = synthetic_monthly_returns(tickers)

    out_path = os.path.join(OUT_DIR, "holdings_household_monthly_returns.csv")
    returns_df.to_csv(out_path, index=False)

    print(f"\nSaved -> {out_path}")
    print(f"Shape: {returns_df.shape}")
