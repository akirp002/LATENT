"""
Pull monthly total-return series for a broad asset-class ETF lineup via
yfinance, convert to monthly percent returns, and save one wide CSV
(Date x Ticker) ready to merge alongside the Fama-French + macro panel.

Where the user's list gave two tickers for one slot (e.g. "IVV / VOO"), the
first is used as primary. Slots marked "no clean pure-play" are skipped and
noted at the bottom of this file.
"""

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Asset class -> ticker. One primary ticker per slot.
# ---------------------------------------------------------------------------
TICKERS = {
    # --- US Equity ---
    "US_LC_Blend": "IVV",       # iShares S&P 500
    "US_LC_Value": "VTV",       # Vanguard Large Value
    "US_LC_Growth": "VUG",      # Vanguard Large Growth
    "US_MC_Blend": "IJH",       # iShares Mid Cap
    "US_MC_Value": "IJJ",       # iShares Mid Value
    "US_MC_Growth": "IJK",      # iShares Mid Growth
    "US_SC_Blend": "IJR",       # iShares Small Cap
    "US_SC_Value": "IJS",       # iShares Small Value
    "US_SC_Growth": "IJT",      # iShares Small Growth

    # --- International Developed Equity ---
    "Intl_LC_Blend": "EFA",     # iShares MSCI EAFE
    "Intl_LC_Value": "EFV",     # iShares MSCI EAFE Value
    "Intl_LC_Growth": "EFG",    # iShares MSCI EAFE Growth
    "Intl_MidSmall_Blend": "SCZ",  # iShares MSCI EAFE Small-Cap
    "Intl_Small_Value": "DLS",  # WisdomTree Intl SmallCap Dividend (value-tilted)
    # Intl_Small_Growth: no clean pure-play ETF exists, skipped

    # --- Emerging Markets Equity ---
    "EM_Broad": "IEMG",         # iShares Core MSCI EM
    "EM_Value": "DEM",          # WisdomTree EM Dividend (value-tilted)
    "EM_Small": "EEMS",         # iShares MSCI EM Small-Cap

    # --- US Fixed Income, Investment Grade ---
    "US_IG_Short": "VCSH",      # Vanguard Short-Term Corp
    "US_IG_Intermediate": "VCIT",  # Vanguard Intermediate-Term Corp
    "US_IG_Long": "VCLT",       # Vanguard Long-Term Corp

    # --- US Fixed Income, High Yield ---
    "US_HY_Short": "SHYG",      # iShares 0-5 Yr High Yield Corp
    "US_HY_Broad": "HYG",       # iShares High Yield Corp (broad, ~4-5yr duration)
    # US_HY_Long: no clean pure-play, HY market is short/intermediate concentrated

    # --- Emerging Market Fixed Income ---
    "EM_Debt_USD_Broad": "EMB",     # iShares JPM USD EM Bond
    "EM_Debt_USD_Short": "EMSH",    # State Street EM Short Duration USD Bond
    "EM_Debt_Local_Broad": "EMLC",  # VanEck JPM EM Local Currency Bond
    "EM_Debt_Local_Alt": "LEMB",    # iShares JPM EM Local Currency Bond

    # --- Treasuries ---
    "UST_Short_1_3Y": "SHY",    # iShares 1-3 Yr Treasury
    "UST_Intermediate_7_10Y": "IEF",  # iShares 7-10 Yr Treasury
    "UST_Long_20Y_Plus": "TLT",  # iShares 20+ Yr Treasury
    "UST_TIPS": "SCHP",         # Schwab TIPS

    # --- Real Assets / Other ---
    "US_REIT": "VNQ",           # Vanguard Real Estate
    "Intl_REIT": "VNQI",        # Vanguard Global ex-US Real Estate
    "Broad_Commodities": "DBC",  # Invesco DB Commodity Index
    "Gold": "GLD",               # SPDR Gold Shares
}


def fetch_monthly_returns(tickers: dict[str, str], start: str = "1990-01-01") -> pd.DataFrame:
    """Download monthly Adjusted Close for each ticker and convert to
    month-over-month percent returns. Returns a wide DataFrame indexed by
    month-start Date, columns = asset class labels."""

    symbols = list(tickers.values())
    print(f"Downloading {len(symbols)} tickers from yfinance...")

    raw = yf.download(
        symbols,
        start=start,
        interval="1mo",
        auto_adjust=True,   # Adjusted Close (dividends/splits) as "Close"
        progress=False,
        group_by="ticker",
        threads=True,
    )

    label_by_symbol = {v: k for k, v in tickers.items()}
    returns = {}
    missing = []

    for symbol in symbols:
        label = label_by_symbol[symbol]
        try:
            close = raw[symbol]["Close"].dropna()
        except (KeyError, TypeError):
            missing.append((label, symbol))
            continue

        if close.empty:
            missing.append((label, symbol))
            continue

        ret = close.pct_change() * 100  # monthly % return, matches FF factor scale
        returns[label] = ret
        print(f"  fetched {label} ({symbol}): {close.index.min().date()} -> {close.index.max().date()}")

    if missing:
        print("\nTickers with no data returned (delisted, wrong symbol, etc.):")
        for label, symbol in missing:
            print(f"  - {label} ({symbol})")

    df = pd.DataFrame(returns)
    # Normalize the index to month-start to match the FF/macro pipeline
    df.index = df.index.to_period("M").to_timestamp(how="start")
    df.index.name = "Date"
    return df.reset_index()


if __name__ == "__main__":
    returns_df = fetch_monthly_returns(TICKERS)

    out_path = r"C:\Users\ak\OneDrive\Desktop\Stuff\Python projects\Data\asset_class_monthly_returns.csv"
    returns_df.to_csv(out_path, index=False)

    print(f"\nSaved -> {out_path}")
    print(f"Shape: {returns_df.shape}")
    print(returns_df.head())
    print(returns_df.tail())

# ---------------------------------------------------------------------------
# Slots skipped (no clean pure-play ETF exists for these):
#   - Intl Small Growth
#   - US High Yield, Long duration
# ---------------------------------------------------------------------------
