r"""
generate_sample_real_holdings.py

*** FABRICATED DATA ONLY ***
This replaces a private "drop a real brokerage export" step. There is no
real household, no real brokerage workbook, and no real dollar figure
anywhere in this file or its output. Every ticker, account label, quantity,
cost basis and gain below is made up for illustration, in the same shape
`ingest_real_holdings.py` (the private, not-included original of this
script) would have produced from a real two-sheet workbook.

Structure this preserves (so the downstream tax-aware CVaR optimizer story
still makes sense), all with fabricated numbers:
  - Two accounts: a taxable joint brokerage account and a tax-advantaged
    IRA, inferred from an account label exactly the way the original
    workbook's sheet name would be.
  - One heavily concentrated single-stock position with a large embedded
    unrealized gain, sitting inside an otherwise diversified, low-cost
    factor/ETF sleeve -- so the optimizer still has to reason about a tax
    cost vs. tail-risk tradeoff on one name.
  - Short-term/long-term unrealized-gain split per position (no per-lot
    dates, exactly like the original), a joint account with almost all of
    household value, and a small IRA sleeve.

Output (Data\ folder), same schema `ingest_real_holdings.py` produced:
  - holdings_household_clean.csv    one row per (account, ticker): quantity,
                                      cost, market value, ST/LT unrealized
                                      gain, weights, Is_Taxable flag
  - holdings_household_tickers.txt  unique ticker list, one per line (for
                                      the price fetch)
"""

import os

import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "Data")

ACCOUNT_TAX_MAP = {
    "Household Joint (Taxable)": True,
    "Account Holder A IRA (Tax-Advantaged)": False,
}

CONCENTRATED_TICKER = "MSTR"  # fabricated stand-in for a large, low-cost-basis single-stock bet

# (Account, Ticker, Name, Investment_Type, Cost_Basis_Method, Quantity,
#  Cost_Per_Share, Market_Value) -- Total_Cost is Quantity * Cost_Per_Share;
# ST/LT gain split is fabricated below (almost all long-term, a small sliver
# short-term on the two most recently added lots, same convention as a real
# multi-year-held book).
FABRICATED_HOLDINGS = [
    # -- Household Joint (Taxable) --
    ("Household Joint (Taxable)", "MSTR", "MicroStrategy Inc", "Equity", "Average Cost",
     1450, 17.24, 650000),
    ("Household Joint (Taxable)", "SCHD", "Schwab US Dividend Equity ETF", "ETF", "Average Cost",
     3200, 42.10, 168000),
    ("Household Joint (Taxable)", "ITOT", "iShares Core S&P Total US Stock Mkt ETF", "ETF", "Average Cost",
     1350, 68.50, 152000),
    ("Household Joint (Taxable)", "IWD", "iShares Russell 1000 Value ETF", "ETF", "Average Cost",
     820, 105.00, 141000),
    ("Household Joint (Taxable)", "IJJ", "iShares S&P Mid-Cap 400 Value ETF", "ETF", "Average Cost",
     1050, 82.40, 138000),
    ("Household Joint (Taxable)", "IEMG", "iShares Core MSCI Emerging Markets ETF", "ETF", "Average Cost",
     2600, 41.00, 135000),
    ("Household Joint (Taxable)", "IJT", "iShares S&P Small-Cap 600 Growth ETF", "ETF", "Average Cost",
     700, 108.50, 128000),
    ("Household Joint (Taxable)", "IJS", "iShares S&P Small-Cap 600 Value ETF", "ETF", "Average Cost",
     650, 95.00, 124000),
    ("Household Joint (Taxable)", "QUAL", "iShares MSCI USA Quality Factor ETF", "ETF", "Average Cost",
     780, 98.00, 118000),
    ("Household Joint (Taxable)", "IEFA", "iShares Core MSCI EAFE ETF", "ETF", "Average Cost",
     1650, 42.30, 113000),
    ("Household Joint (Taxable)", "MTUM", "iShares MSCI USA Momentum Factor ETF", "ETF", "Average Cost",
     540, 105.00, 96000),
    ("Household Joint (Taxable)", "IWF", "iShares Russell 1000 Growth ETF", "ETF", "Average Cost",
     260, 175.00, 90000),
    ("Household Joint (Taxable)", "IJK", "iShares S&P Mid-Cap 400 Growth ETF", "ETF", "Average Cost",
     280, 148.00, 82000),
    ("Household Joint (Taxable)", "SHV", "iShares Short Treasury Bond ETF", "Fund", "Average Cost",
     60, 110.30, 6600),
    ("Household Joint (Taxable)", "XLB", "Materials Select Sector SPDR Fund", "ETF", "Average Cost",
     140, 68.00, 12000),
    ("Household Joint (Taxable)", "SCHH", "Schwab US REIT ETF", "ETF", "Average Cost",
     380, 20.90, 8000),
    # -- Account Holder A IRA (Tax-Advantaged) --
    ("Account Holder A IRA (Tax-Advantaged)", "ITOT", "iShares Core S&P Total US Stock Mkt ETF", "ETF", "Average Cost",
     130, 65.00, 14200),
    ("Account Holder A IRA (Tax-Advantaged)", "IWD", "iShares Russell 1000 Value ETF", "ETF", "Average Cost",
     85, 98.00, 12800),
    ("Account Holder A IRA (Tax-Advantaged)", "QUAL", "iShares MSCI USA Quality Factor ETF", "ETF", "Average Cost",
     95, 90.00, 11400),
    ("Account Holder A IRA (Tax-Advantaged)", "MTUM", "iShares MSCI USA Momentum Factor ETF", "ETF", "Average Cost",
     55, 98.00, 9200),
    ("Account Holder A IRA (Tax-Advantaged)", "IJJ", "iShares S&P Mid-Cap 400 Value ETF", "ETF", "Average Cost",
     70, 75.00, 8800),
    ("Account Holder A IRA (Tax-Advantaged)", "IEMG", "iShares Core MSCI Emerging Markets ETF", "ETF", "Average Cost",
     140, 38.00, 6900),
    ("Account Holder A IRA (Tax-Advantaged)", "SCHD", "Schwab US Dividend Equity ETF", "ETF", "Average Cost",
     150, 38.00, 6200),
    ("Account Holder A IRA (Tax-Advantaged)", "SCHH", "Schwab US REIT ETF", "ETF", "Average Cost",
     220, 18.50, 4500),
]

# Fraction of each position's total gain treated as short-term (recently
# added lot) vs. long-term (everything else). MSTR and the IRA sleeve are
# entirely long-term/tax-free-on-paper by construction; a few of the taxable
# joint account's ETF lines carry a small fabricated short-term sliver.
ST_FRACTION = {
    "IJT": 0.08, "IJS": 0.06, "IWF": 0.05,
}


def build_holdings():
    rows = []
    for account, ticker, name, itype, cb_method, qty, cost_per_share, mv in FABRICATED_HOLDINGS:
        total_cost = round(qty * cost_per_share, 2)
        total_gain = round(mv - total_cost, 2)
        st_frac = ST_FRACTION.get(ticker, 0.0)
        st_gain = round(total_gain * st_frac, 2)
        lt_gain = round(total_gain - st_gain, 2)
        rows.append({
            "Account": account, "Ticker": ticker, "Name": name,
            "Investment_Type": itype, "Cost_Basis_Method": cb_method,
            "Quantity": qty, "Cost_Per_Share": cost_per_share,
            "Total_Cost": total_cost, "Market_Value": mv,
            "ST_Gain": st_gain, "LT_Gain": lt_gain,
        })
    df = pd.DataFrame(rows)
    df["Total_Gain"] = df["ST_Gain"] + df["LT_Gain"]
    df["Pct_Gain"] = (df["Market_Value"] - df["Total_Cost"]) / df["Total_Cost"]
    df["Is_Taxable"] = df["Account"].map(ACCOUNT_TAX_MAP)
    return df


def add_weights(df):
    df = df.copy()
    total = df["Market_Value"].sum()
    df["Weight_Household"] = df["Market_Value"] / total
    df["Weight_Within_Account"] = df["Market_Value"] / df.groupby("Account")["Market_Value"].transform("sum")
    return df, total


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    holdings = build_holdings()
    holdings, total_mv = add_weights(holdings)

    print(f"Loaded {len(holdings)} positions across {holdings['Account'].nunique()} accounts "
          f"({holdings['Ticker'].nunique()} unique tickers) -- ALL FABRICATED")
    print(f"Total household market value (fabricated): ${total_mv:,.2f}\n")

    for account, group in holdings.groupby("Account"):
        is_tax = group["Is_Taxable"].iloc[0]
        mv = group["Market_Value"].sum()
        gain = group["Total_Gain"].sum()
        print(f"  {account} [{'TAXABLE' if is_tax else 'tax-advantaged'}]: "
              f"${mv:,.2f} market value, ${gain:,.2f} total unrealized gain, "
              f"{group['Ticker'].nunique()} tickers")

    conc_row = holdings[holdings["Ticker"] == CONCENTRATED_TICKER]
    if not conc_row.empty:
        conc_value = conc_row["Market_Value"].iloc[0]
        conc_gain_pct = conc_row["Pct_Gain"].iloc[0]
        print(f"\n{CONCENTRATED_TICKER} position: ${conc_value:,.2f} "
              f"({conc_value / total_mv:.2%} of fabricated household value), "
              f"{conc_gain_pct:.2%} unrealized gain")

    print("\nFull holdings (sorted by household weight):")
    print(holdings[["Account", "Ticker", "Quantity", "Total_Cost", "Market_Value",
                     "ST_Gain", "LT_Gain", "Pct_Gain", "Weight_Household"]]
          .sort_values("Weight_Household", ascending=False)
          .to_string(index=False, formatters={
              "Total_Cost": "{:,.2f}".format, "Market_Value": "{:,.2f}".format,
              "ST_Gain": "{:,.2f}".format, "LT_Gain": "{:,.2f}".format,
              "Pct_Gain": "{:.2%}".format, "Weight_Household": "{:.4%}".format,
          }))

    holdings.to_csv(os.path.join(OUT_DIR, "holdings_household_clean.csv"), index=False)
    with open(os.path.join(OUT_DIR, "holdings_household_tickers.txt"), "w") as f:
        f.write("\n".join(sorted(holdings["Ticker"].unique())))

    print(f"\nSaved:")
    print(f"  {os.path.join(OUT_DIR, 'holdings_household_clean.csv')}")
    print(f"  {os.path.join(OUT_DIR, 'holdings_household_tickers.txt')}")
