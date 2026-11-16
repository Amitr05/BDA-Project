"""
Market Data Ingestion Module
Retrieves daily historical market benchmark data (S&P 500 and VIX) using yfinance
and persists them as raw CSV files.
"""

import os
import logging
import pandas as pd
import yfinance as yf
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_START_DATE = "2015-01-01"
DEFAULT_END_DATE = "2026-02-01"

RAW_DATA_PATHS = {
    "^GSPC": "data/raw/sp500_raw.csv",
    "^IXIC": "data/raw/nasdaq_raw.csv",
    "^DJI": "data/raw/dow_jones_raw.csv",
    "^RUT": "data/raw/russell2000_raw.csv",
    "^VIX": "data/raw/vix_raw.csv"
}



def is_valid_local_csv(
    file_path: str,
    min_rows: int = 100,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    tolerance_days: int = 7
) -> bool:
    """
    Validate that a local CSV exists, has the required structure,
    sufficient records, and covers the requested date range.
    """
    if not os.path.exists(file_path):
        return False

    try:
        if os.path.getsize(file_path) < 100:
            return False

        # Check required columns
        sample_df = pd.read_csv(file_path, nrows=5)
        required_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}

        if not required_cols.issubset(set(sample_df.columns)):
            return False

        # Read dates and check record count
        date_df = pd.read_csv(file_path, usecols=["Date"])

        if len(date_df) <= min_rows:
            return False

        # Validate date values
        dates = pd.to_datetime(date_df["Date"], errors="coerce")

        if dates.isna().any():
            logger.warning(f"Invalid dates found in {file_path}")
            return False

        actual_start = dates.min()
        actual_end = dates.max()

        requested_start = pd.Timestamp(start_date)
        requested_end = pd.Timestamp(end_date)

        # Allow tolerance for weekends and market holidays
        earliest_allowed = requested_start + pd.Timedelta(days=tolerance_days)
        latest_allowed = requested_end - pd.Timedelta(days=tolerance_days)

        if actual_start > earliest_allowed:
            logger.warning(
                f"Dataset starts too late: {actual_start.date()} "
                f"(expected around {start_date})"
            )
            return False

        if actual_end < latest_allowed:
            logger.warning(
                f"Dataset ends too early: {actual_end.date()} "
                f"(expected around {end_date})"
            )
            return False

        return True

    except Exception as e:
        logger.warning(f"Validation failed for local file {file_path}: {e}")
        return False



def fetch_historical_ticker(
    ticker: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE
) -> pd.DataFrame:
    """
    Download historical market data for a given ticker using yfinance.

    Returns a cleaned DataFrame with columns:
    ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    """
    logger.info(f"Downloading historical data for ticker '{ticker}' from {start_date} to {end_date}...")
    try:
        # auto_adjust=False ensures both 'Close' and 'Adj Close' are returned if available
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    except Exception as err:
        logger.error(f"Failed to query yfinance for ticker '{ticker}': {err}")
        raise RuntimeError(f"Network or API failure while fetching {ticker}: {err}") from err

    if df is None or df.empty:
        raise ValueError(f"No market data returned for ticker '{ticker}'. Please check ticker symbol and date range.")

    # Flatten MultiIndex columns if returned by newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    df = df.reset_index()

    # Standardize Date column
    if "Date" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "Date"})

    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    # Ensure required columns exist
    if "Adj Close" not in df.columns:
        # If Adj Close is not distinct (e.g. indices like VIX), set Adj Close = Close
        df["Adj Close"] = df["Close"]

    required_order = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    available_cols = [c for c in required_order if c in df.columns]
    df = df[available_cols]

    # Drop rows where Close is null
    df = df.dropna(subset=["Close"])
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    logger.info(f"Successfully retrieved {len(df)} records for {ticker} ({df['Date'].min()} to {df['Date'].max()}).")
    return df


def ingest_market_data(
    ticker: str,
    output_path: Optional[str] = None,
    force_download: bool = False,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE
) -> str:
    """
    Ingest data for a given ticker and persist as CSV.
    Skips downloading if valid file exists and force_download is False.
    """
    if output_path is None:
        output_path = RAW_DATA_PATHS.get(ticker, f"data/raw/{ticker.replace('^', '').lower()}_raw.csv")

    if not force_download and is_valid_local_csv(
    output_path,
    start_date=start_date,
    end_date=end_date
):
        logger.info(f"Local file already exists and is valid: {output_path}. Skipping download.")
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = fetch_historical_ticker(ticker=ticker, start_date=start_date, end_date=end_date)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {ticker} dataset ({len(df)} rows) to {output_path}.")
    return output_path



def ingest_all_raw_data(force_download: bool = False) -> dict:
    """
    Ingest all configured market datasets independently.

    If one ticker fails, continue attempting the remaining tickers.
    Returns a dictionary containing successfully ingested datasets.
    """
    results = {}
    failures = {}

    for ticker, path in RAW_DATA_PATHS.items():
        try:
            logger.info(f"Starting ingestion for {ticker}...")

            results[ticker] = ingest_market_data(
                ticker=ticker,
                output_path=path,
                force_download=force_download
            )

        except Exception as e:
            failures[ticker] = str(e)
            logger.exception(
                f"Failed to ingest {ticker}. Continuing with remaining datasets."
            )

    # Final ingestion summary
    logger.info("=" * 60)
    logger.info("MARKET DATA INGESTION SUMMARY")
    logger.info("=" * 60)

    logger.info(f"Successful datasets: {len(results)}")

    for ticker, path in results.items():
        logger.info(f"SUCCESS | {ticker} | {path}")

    logger.info(f"Failed datasets: {len(failures)}")

    for ticker, error in failures.items():
        logger.error(f"FAILED | {ticker} | {error}")

    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    ingest_all_raw_data()
