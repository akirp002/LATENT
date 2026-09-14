r"""
generate_sample_holdings.py

*** SAMPLE DATA ONLY ***
This repo ships with a fabricated demo portfolio, not a real client's real
holdings. Every ticker, account label, quantity and dollar value below is
made up for illustration. It stands in for the private "load_holdings.py"
step that, in the original pipeline, parses a real brokerage export -- that
step (and the real workbook it reads) is intentionally not included here.

Structure: two accounts, one large single-stock "concentrated position"
(a recent-IPO name, PLTR -- short trading history makes it a good
stress-test for the regression/history-alignment logic downstream),
full-portfolio weights, and ex-concentrated-position weights (remaining
holdings re-normalized to 100%).

Outputs (Data\ folder):
  - holdings_clean.csv       one row per (account, ticker), market value + weight
  - holdings_tickers.txt     unique ticker list, one per line (for the price fetch)
"""

import os

import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "Data")

CONCENTRATED_TICKER = "PLTR"  # stand-in for a large, recent-IPO single-stock position

# Fabricated positions -- two accounts, made-up quantities/costs/values.
SAMPLE_HOLDINGS = [
    # Account, Ticker, Name, Investment_Type, Quantity, Total_Cost, Market_Value
    ("Demo Taxable (Joint)", "PLTR", "Palantir Technologies Inc", "Equity", 9000, 95000, 420000),
    ("Demo Taxable (Joint)", "AAPL", "Apple Inc", "Equity", 300, 42000, 68000),
    ("Demo Taxable (Joint)", "MSFT", "Microsoft Corp", "Equity", 150, 38000, 62000),
    ("Demo Taxable (Joint)", "JNJ", "Johnson & Johnson", "Equity", 250, 34000, 37000),
    ("Demo Taxable (Joint)", "PG", "Procter & Gamble Co", "Equity", 200, 26000, 32000),
    ("Demo Taxable (Joint)", "JPM", "JPMorgan Chase & Co", "Equity", 180, 27000, 35000),
    ("Demo Taxable (Joint)", "XOM", "Exxon Mobil Corp", "Equity", 220, 21000, 25000),
    ("Demo Taxable (Joint)", "VNQ", "Vanguard Real Estate ETF", "ETF", 400, 35000, 36000),
    ("Demo Taxable (Joint)", "GLD", "SPDR Gold Shares", "ETF", 150, 26000, 29000),
    ("Demo IRA", "SPY", "SPDR S&P 500 ETF Trust", "ETF", 250, 90000, 112000),
    ("Demo IRA", "AGG", "iShares Core US Aggregate Bond ETF", "ETF", 500, 51000, 49000),
    ("Demo IRA", "VTV", "Vanguard Value ETF", "ETF", 300, 39000, 43000),
    ("Demo IRA", "VUG", "Vanguard Growth ETF", "ETF", 200, 51000, 58000),
    ("Demo IRA", "KO", "Coca-Cola Co", "Equity", 400, 22000, 26000),
    ("Demo IRA", "PEP", "PepsiCo Inc", "Equity", 150, 21000, 24000),
    ("Demo IRA", "UNH", "UnitedHealth Group Inc", "Equity", 60, 28000, 31000),
    ("Demo IRA", "VFISX", "Vanguard Short-Term Treasury Fund", "Fund", 3000, 32000, 31000),
]


def load_holdings():
    df = pd.DataFrame(SAMPLE_HOLDINGS, columns=[
        "Account", "Ticker", "Name", "Investment_Type", "Quantity", "Total_Cost", "Market_Value",
    ])
    df["Market_Value"] = pd.to_numeric(df["Market_Value"], errors="coerce")
    df["Total_Cost"] = pd.to_numeric(df["Total_Cost"], errors="coerce")
    return df


def add_weights(df):
    total = df["Market_Value"].sum()
    df = df.copy()
    df["Weight_Full"] = df["Market_Value"] / total

    ex_conc = df[df["Ticker"] != CONCENTRATED_TICKER].copy()
    ex_conc_total = ex_conc["Market_Value"].sum()
    df["Weight_ExConcentrated"] = 0.0
    df.loc[df["Ticker"] != CONCENTRATED_TICKER, "Weight_ExConcentrated"] = (
        ex_conc["Market_Value"] / ex_conc_total
    )
    return df, total, ex_conc_total


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_holdings()
    df, total, ex_conc_total = add_weights(df)

    print(f"Accounts: {sorted(df['Account'].unique())}")
    print(f"Total holdings (both accounts): {len(df)} positions, {df['Ticker'].nunique()} unique tickers")
    print(f"Total market value (sample): ${total:,.2f}")
    print(f"Total market value ex-{CONCENTRATED_TICKER} (sample): ${ex_conc_total:,.2f}")

    conc_row = df[df["Ticker"] == CONCENTRATED_TICKER]
    if not conc_row.empty:
        conc_value = conc_row["Market_Value"].iloc[0]
        print(f"\n{CONCENTRATED_TICKER} position: ${conc_value:,.2f}  ({conc_value/total:.2%} of full sample portfolio)")

    print("\nFull-portfolio weights (sample data):")
    print(df[["Account", "Ticker", "Market_Value", "Weight_Full", "Weight_ExConcentrated"]]
          .sort_values("Weight_Full", ascending=False).to_string(index=False))

    df.to_csv(os.path.join(OUT_DIR, "holdings_clean.csv"), index=False)
    with open(os.path.join(OUT_DIR, "holdings_tickers.txt"), "w") as f:
        f.write("\n".join(sorted(df["Ticker"].unique())))

    print(f"\nSaved:")
    print(f"  {OUT_DIR}\\holdings_clean.csv")
    print(f"  {OUT_DIR}\\holdings_tickers.txt")
