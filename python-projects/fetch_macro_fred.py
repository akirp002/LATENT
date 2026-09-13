"""
Pull macro series from FRED and merge them onto the monthly Fama-French
factor data produced by transform_data_ff.py.

NOT everything on your wishlist is available from FRED (it's a US-Fed-run
database; several items are proprietary / other-vendor data). What's
skipped and why is listed in UNAVAILABLE_ON_FRED at the bottom of this file.
"""

import pandas as pd
import requests

import os

FRED_API_KEY = os.environ.get("FRED_API_KEY")
if not FRED_API_KEY:
    raise RuntimeError(
        "Set the FRED_API_KEY environment variable (get a free key at "
        "https://fred.stlouisfed.org/docs/api/api_key.html) before running this script."
    )
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# ---------------------------------------------------------------------------
# Series actually available on FRED, grouped to match your channel breakdown.
# Frequencies vary by series; we resample everything to month-start (MS) and
# forward-fill within a month where FRED gives daily/weekly data.
# ---------------------------------------------------------------------------
FRED_SERIES = {
    # Growth / business cycle
    "US_INDPRO": "INDPRO",                       # US Industrial Production
    "OECD_CLI_US": "USALOLITOAASTSAM",            # OECD Composite Leading Indicator, US
    "EA_INDPRO": "EA19PRINTO01GYSAM",             # Euro area Industrial Production (YoY, SA)
    "CHINA_INDPRO": "CHNPRINTO01IXPYM",           # China Industrial Production (index)

    # Inflation
    "US_CPI": "CPIAUCSL",                         # US CPI, all items
    "US_CORE_CPI": "CPILFESL",                    # US CPI ex food & energy
    "EA_HICP": "CP0000EZ19M086NEST",               # Euro area HICP
    "CHINA_CPI": "CHNCPIALLMINMEI",                # China CPI

    # Rates
    "US_10Y": "GS10",                              # US 10Y Treasury yield (monthly)
    "DE_10Y": "IRLTLT01DEM156N",                   # German 10Y (Bund) yield
    "JP_10Y": "IRLTLT01JPM156N",                   # Japan 10Y (JGB) yield
    "US_10Y_REAL": "DFII10",                       # US 10Y TIPS real yield (daily)
    "US_BREAKEVEN_5Y5Y": "T5YIFR",                 # US 5Y5Y forward inflation expectation
    "US_BREAKEVEN_10Y": "T10YIE",                  # US 10Y breakeven inflation

    # Monetary policy
    "FED_FUNDS": "FEDFUNDS",                       # Fed funds rate
    "ECB_DEPO_RATE": "ECBDFR",                     # ECB deposit facility rate
    "JP_DISCOUNT_RATE": "INTDSRJPM193N",           # BOJ discount rate (proxy for policy rate)
    "FED_BALANCE_SHEET": "WALCL",                  # Fed total assets (weekly)
    "US_M2": "M2SL",                                # US M2 money supply
    "EA_M2": "MYAGM2EZM196N",                       # Euro area M2
    "CHINA_M2": "MYAGM2CNM189N",                    # China M2

    # Currency
    "DXY_BROAD": "DTWEXBGS",                       # Trade-weighted USD, broad index (daily)

    # Credit / risk appetite
    # NOTE: FRED's API caps the ICE BofA HY OAS series (BAMLH0A0HYM2,
    # BAMLHE00EHYIOAS) at ~3 years of history for licensing reasons, even
    # though the FRED website shows decades of data. Using Moody's Baa
    # corporate spread instead, which has full history and no such cap.
    "US_CREDIT_SPREAD_BAA10Y": "BAA10Y",            # Moody's Baa yield minus 10Y Treasury
    "VIX": "VIXCLS",                                # CBOE VIX (daily)

    # Commodities
    "BRENT_OIL": "DCOILBRENTEU",                    # Brent crude (daily)
    "COPPER": "PCOPPUSDM",                          # Global copper price (monthly)
}


def fetch_series(series_id: str) -> pd.Series:
    """Fetch one FRED series as a monthly (month-start) pandas Series."""
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
    }
    resp = requests.get(FRED_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    obs = resp.json()["observations"]

    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # FRED uses "." for missing
    df = df.set_index("date")["value"]

    # Resample to month-start: take the last observation available in each
    # month (works whether the source is daily, weekly, or already monthly).
    return df.resample("MS").last()


def build_macro_dataframe() -> pd.DataFrame:
    series_frames = {}
    failed = []
    for label, series_id in FRED_SERIES.items():
        try:
            series_frames[label] = fetch_series(series_id)
            print(f"  fetched {label} ({series_id})")
        except Exception as exc:
            failed.append((label, series_id, str(exc)))
            print(f"  FAILED {label} ({series_id}): {exc}")

    macro_df = pd.DataFrame(series_frames)
    macro_df.index.name = "Date"
    macro_df = macro_df.reset_index()

    if failed:
        print("\nSeries that failed to fetch (check series ID / API key):")
        for label, series_id, err in failed:
            print(f"  - {label} ({series_id}): {err}")

    return macro_df


if __name__ == "__main__":
    # --- Load the monthly Fama-French factors (from transform_data_ff.py) ---
    ff = pd.read_csv(
        r"C:\Users\ak\OneDrive\Desktop\Stuff\Python projects\Data\F_F_Research_Data_5_Factors_2x3.csv",
        skiprows=3,
    )
    ff = ff.rename(columns={ff.columns[0]: "Date"})
    annual_marker = ff["Date"].astype(str).str.contains("Annual Factors", na=False)
    if annual_marker.any():
        ff = ff.loc[: annual_marker.idxmax() - 1]
    ff["Date"] = pd.to_datetime(ff["Date"].astype(str).str.strip(), format="%Y%m")

    # --- Pull macro data from FRED ---
    print("Fetching macro series from FRED...")
    macro = build_macro_dataframe()

    # --- Merge on Date ---
    merged = ff.merge(macro, on="Date", how="left")

    full_path = r"C:\Users\ak\OneDrive\Desktop\Stuff\Python projects\Data\FF_plus_macro_full.csv"
    merged.to_csv(full_path, index=False)
    print(f"\nSaved FULL merged dataset (all history, raw gaps) -> {full_path}")
    print(f"Shape: {merged.shape}")

    # -----------------------------------------------------------------------
    # Build a "workable" dataset: trimmed to a window where the macro panel
    # is actually dense, with slow/lagging-report series forward-filled so
    # you don't lose rows to a handful of stale columns.
    # -----------------------------------------------------------------------
    macro_cols = [c for c in merged.columns if c not in ff.columns]

    # Start where the last "late starter" series (DXY_BROAD, 2006-01) kicks in,
    # so every macro column has at least *some* data across the window.
    WORKABLE_START = "2006-01-01"
    workable = merged[merged["Date"] >= WORKABLE_START].copy()

    # Forward-fill macro columns: legitimate for series that have simply
    # stopped being reported recently (e.g. EA_INDPRO, CHINA_M2 lag or were
    # discontinued) or report less often than monthly. FF factors themselves
    # are never touched.
    workable[macro_cols] = workable[macro_cols].ffill()

    workable_path = r"C:\Users\ak\OneDrive\Desktop\Stuff\Python projects\Data\FF_plus_macro_workable.csv"
    workable.to_csv(workable_path, index=False)
    print(f"Saved WORKABLE dataset ({WORKABLE_START} onward, macro cols ffilled) -> {workable_path}")
    print(f"Shape: {workable.shape}")

    # Coverage report so gaps are visible rather than silent
    print("\nMacro column coverage in the workable window (share non-null after ffill):")
    coverage = workable[macro_cols].notna().mean().sort_values()
    for col, pct in coverage.items():
        flag = "  <-- still has gaps at the start" if pct < 0.999 else ""
        print(f"  {col:<28} {pct:6.1%}{flag}")

    print("\nHead:")
    print(workable.head())
    print("\nTail:")
    print(workable.tail())

# ---------------------------------------------------------------------------
# Items from your wishlist NOT available on FRED (not tracked by the Fed / are
# proprietary to other vendors). You'd need a paid data feed (Bloomberg,
# Refinitiv, S&P Global) or scraping for these:
#
#   - JPMorgan Global Composite PMI          (S&P Global / JPMorgan, licensed)
#   - CPB World Trade Monitor                (Dutch CPB, free but not on FRED
#                                              — could scrape/download separately)
#   - EM broad Industrial Production aggregate (no single clean FRED series)
#   - JPMorgan EM FX index                   (JPMorgan, licensed)
#   - Terms of trade by country               (spotty FRED coverage, not a clean series)
#   - PBOC policy rate                        (not reliably on FRED)
#   - MOVE index                              (ICE/BofA, proprietary)
#   - ICE BofA HY OAS spreads (US & Euro)     (available on FRED's website, but the
#                                              API caps history to ~3 years for
#                                              licensing reasons; substituted with
#                                              Moody's Baa-10Y spread, US only)
#   - EMBI / EMBI+ sovereign spread           (JPMorgan, licensed; FRED's old
#                                              EMBI series was discontinued)
#   - China credit impulse                    (derived series; not published directly,
#                                              you'd compute it from PBOC total social
#                                              financing data)
# ---------------------------------------------------------------------------
