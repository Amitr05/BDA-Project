"""
Company Stock Data Ingestion Module
Handles bulk downloading, incremental updates, caching, retry logic,
failed download tracking, and historical index membership for individual company stocks.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Set, Any
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_START_DATE = "2015-01-01"
DEFAULT_END_DATE = "2026-02-01"
DEFAULT_STOCKS_DIR = "data/raw/stocks"
COMPANIES_CSV = "data/raw/companies.csv"
CONSTITUENTS_CSV = "data/raw/index_constituents.csv"
INGESTION_LOG_JSON = "data/raw/stock_ingestion_log.json"
FAILED_DOWNLOADS_JSON = "data/raw/failed_downloads.json"


# -----------------------------------------------------------------------------
# 1. Constituent Registry and Historical Membership
# -----------------------------------------------------------------------------

def load_company_registry(
    file_path: str = COMPANIES_CSV,
    max_companies: Optional[int] = None
) -> pd.DataFrame:
    """
    Load company master registry containing ticker, company name, sector, and industry.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Company master registry not found at {file_path}. "
            "Please ensure data/raw/companies.csv is initialized."
        )

    df = pd.read_csv(file_path)
    required_cols = {"company_id", "ticker", "company_name", "sector", "industry"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Company registry missing required columns: {missing}")

    if max_companies is not None and max_companies > 0:
        df = df.head(max_companies).copy()

    return df


def load_index_constituents(file_path: str = CONSTITUENTS_CSV) -> pd.DataFrame:
    """
    Load index constituents junction table with historical validity intervals
    to reduce survivorship bias.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Constituents junction table not found at {file_path}."
        )

    df = pd.read_csv(file_path)
    required_cols = {"constituent_id", "index_id", "ticker", "valid_from", "is_active"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Constituents table missing required columns: {missing}")

    return df


def get_active_constituents(
    as_of_date: str,
    index_id: Optional[str] = None,
    constituents_file: str = CONSTITUENTS_CSV
) -> List[str]:
    """
    Filter constituents active on a specific historical date (YYYY-MM-DD).
    Combats survivorship bias by honoring index join dates (valid_from)
    and removal dates (valid_to).
    """
    df = load_index_constituents(constituents_file)
    as_of = pd.Timestamp(as_of_date)

    valid_from = pd.to_datetime(df["valid_from"], errors="coerce")
    valid_to = pd.to_datetime(df["valid_to"], errors="coerce")

    mask = (valid_from <= as_of) & (valid_to.isna() | (valid_to >= as_of))

    if index_id is not None:
        mask = mask & (df["index_id"] == index_id)

    active_tickers = sorted(df.loc[mask, "ticker"].unique().tolist())
    return active_tickers


# -----------------------------------------------------------------------------
# 2. Validation & Caching
# -----------------------------------------------------------------------------

def is_valid_stock_csv(
    file_path: str,
    min_rows: Optional[int] = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    tolerance_days: int = 14
) -> bool:
    """
    Validate that an individual stock CSV exists, has valid OHLCV columns,
    and spans the required historical period.
    """
    if not os.path.exists(file_path):
        return False

    try:
        if os.path.getsize(file_path) < 100:
            return False

        # Verify header structure
        sample_df = pd.read_csv(file_path, nrows=5)
        required_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
        if not required_cols.issubset(set(sample_df.columns)):
            return False

        date_df = pd.read_csv(file_path, usecols=["Date"])
        
        requested_start = pd.Timestamp(start_date)
        requested_end = pd.Timestamp(end_date)
        span_days = max(1, (requested_end - requested_start).days)

        if min_rows is None:
            effective_min_rows = max(1, min(50, int(span_days * 0.4)))
        else:
            effective_min_rows = min_rows

        if len(date_df) < effective_min_rows:
            return False

        dates = pd.to_datetime(date_df["Date"], errors="coerce")
        if dates.isna().any():
            return False

        actual_start = dates.min()
        actual_end = dates.max()

        eff_tolerance = min(tolerance_days, max(1, int(span_days * 0.2)))
        latest_allowed = requested_end - pd.Timedelta(days=eff_tolerance)

        # Allow newly listed stocks if their data runs through the end of the period
        if actual_end < latest_allowed:
            return False

        return True
    except Exception as e:
        logger.debug(f"Validation check failed for {file_path}: {e}")
        return False


# -----------------------------------------------------------------------------
# 3. Batch Fetching & Incremental Ingestion
# -----------------------------------------------------------------------------

def fetch_single_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
    backoff_factor: float = 1.5
) -> Optional[pd.DataFrame]:
    """
    Download historical daily OHLCV data for a single ticker with retry logic.
    """
    clean_ticker = ticker.replace(".", "-")
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                clean_ticker,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False
            )
            if df is None or df.empty:
                return None

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

            df = df.reset_index()
            if "Date" not in df.columns and len(df.columns) > 0:
                df = df.rename(columns={df.columns[0]: "Date"})

            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            if "Adj Close" not in df.columns:
                df["Adj Close"] = df["Close"]

            df["Ticker"] = ticker

            cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"]
            available = [c for c in cols if c in df.columns]
            df = df[available]

            df = df.dropna(subset=["Date", "Close"])
            df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
            return df
        except Exception as err:
            wait_time = backoff_factor ** attempt
            logger.warning(f"Attempt {attempt}/{max_retries} failed for {ticker}: {err}. Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)

    return None


def fetch_stocks_batch(
    tickers: List[str],
    start_date: str,
    end_date: str,
    batch_size: int = 30,
    pause_between_batches: float = 1.0,
    max_retries: int = 3
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Bulk download tickers in chunks using yfinance with error containment and rate limiting.
    """
    results: Dict[str, Optional[pd.DataFrame]] = {}
    clean_map = {t: t.replace(".", "-") for t in tickers}

    for i in range(0, len(tickers), batch_size):
        chunk_tickers = tickers[i : i + batch_size]
        chunk_clean = [clean_map[t] for t in chunk_tickers]
        logger.info(f"Downloading batch {i // batch_size + 1}/{(len(tickers) + batch_size - 1) // batch_size} ({len(chunk_tickers)} tickers)...")

        batch_success = False
        for attempt in range(1, max_retries + 1):
            try:
                batch_df = yf.download(
                    chunk_clean,
                    start=start_date,
                    end=end_date,
                    group_by="ticker",
                    auto_adjust=False,
                    progress=False
                )
                batch_success = True
                break
            except Exception as e:
                wait_sec = 2.0 * attempt
                logger.warning(f"Batch download attempt {attempt} failed: {e}. Retrying in {wait_sec}s...")
                time.sleep(wait_sec)

        # Parse individual ticker sub-frames
        for orig_ticker in chunk_tickers:
            clean_sym = clean_map[orig_ticker]
            ticker_df = None

            if batch_success and batch_df is not None and not batch_df.empty:
                try:
                    if len(chunk_tickers) == 1:
                        sub = batch_df
                    else:
                        sub = batch_df[clean_sym] if clean_sym in batch_df else None

                    if sub is not None and not sub.empty:
                        sub = sub.dropna(how="all").reset_index()
                        if "Date" not in sub.columns and len(sub.columns) > 0:
                            sub = sub.rename(columns={sub.columns[0]: "Date"})

                        if "Close" in sub.columns and not sub["Close"].dropna().empty:
                            sub["Date"] = pd.to_datetime(sub["Date"]).dt.strftime("%Y-%m-%d")
                            if "Adj Close" not in sub.columns:
                                sub["Adj Close"] = sub["Close"]
                            sub["Ticker"] = orig_ticker

                            cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"]
                            available = [c for c in cols if c in sub.columns]
                            sub = sub[available].dropna(subset=["Date", "Close"])
                            sub = sub.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
                            if len(sub) > 0:
                                ticker_df = sub
                except Exception as parse_err:
                    logger.debug(f"Could not parse batch output for {orig_ticker}: {parse_err}")

            # Fallback to single fetch if batch didn't retrieve valid data
            if ticker_df is None or ticker_df.empty:
                logger.info(f"Falling back to single ticker fetch for {orig_ticker}...")
                ticker_df = fetch_single_ticker(orig_ticker, start_date=start_date, end_date=end_date)

            results[orig_ticker] = ticker_df

        if pause_between_batches > 0:
            time.sleep(pause_between_batches)

    return results


def save_stock_data_incremental(
    ticker: str,
    new_df: pd.DataFrame,
    output_dir: str = DEFAULT_STOCKS_DIR
) -> str:
    """
    Save or merge stock data incrementally into data/raw/stocks/<ticker>_raw.csv,
    avoiding duplicate rows and preserving existing historical data.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{ticker}_raw.csv")

    if os.path.exists(file_path):
        try:
            existing_df = pd.read_csv(file_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df["Date"] = pd.to_datetime(combined_df["Date"]).dt.strftime("%Y-%m-%d")
            combined_df = combined_df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
            combined_df.to_csv(file_path, index=False)
            return file_path
        except Exception as e:
            logger.warning(f"Error merging incremental stock data for {ticker}: {e}. Overwriting with fresh data.")

    new_df["Date"] = pd.to_datetime(new_df["Date"]).dt.strftime("%Y-%m-%d")
    new_df = new_df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    new_df.to_csv(file_path, index=False)
    return file_path


# -----------------------------------------------------------------------------
# 4. Ingestion Orchestrator
# -----------------------------------------------------------------------------

def ingest_company_stocks(
    max_companies: Optional[int] = None,
    output_dir: str = DEFAULT_STOCKS_DIR,
    force_download: bool = False,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    batch_size: int = 30,
    pause_between_batches: float = 0.5,
    companies_file: str = COMPANIES_CSV
) -> Dict[str, Any]:
    """
    Main ingestion pipeline for individual company stocks.
    - Caches valid files.
    - Resumes interrupted downloads.
    - Tracks successes and failures.
    - Supports configurable company counts (e.g. 5 for testing, 300-500 for production).
    """
    os.makedirs(output_dir, exist_ok=True)
    companies_df = load_company_registry(file_path=companies_file, max_companies=max_companies)
    all_tickers = companies_df["ticker"].unique().tolist()

    logger.info("=" * 60)
    logger.info(f"STARTING COMPANY STOCK INGESTION (Target: {len(all_tickers)} companies)")
    logger.info(f"Date Range: {start_date} to {end_date} | Output: {output_dir}")
    logger.info("=" * 60)

    tickers_to_download: List[str] = []
    skipped_cached: List[str] = []

    for t in all_tickers:
        expected_path = os.path.join(output_dir, f"{t}_raw.csv")
        if not force_download and is_valid_stock_csv(expected_path, start_date=start_date, end_date=end_date):
            skipped_cached.append(t)
        else:
            tickers_to_download.append(t)

    logger.info(f"Cached/Skipped: {len(skipped_cached)} companies | Remaining to download: {len(tickers_to_download)}")

    successful: List[str] = list(skipped_cached)
    failed: Dict[str, str] = {}

    if tickers_to_download:
        batch_results = fetch_stocks_batch(
            tickers=tickers_to_download,
            start_date=start_date,
            end_date=end_date,
            batch_size=batch_size,
            pause_between_batches=pause_between_batches
        )

        for ticker, df in batch_results.items():
            if df is not None and not df.empty:
                save_stock_data_incremental(ticker, df, output_dir=output_dir)
                successful.append(ticker)
                logger.info(f"SUCCESS: Saved {ticker} ({len(df)} rows).")
            else:
                failed[ticker] = "No market data returned from Yahoo Finance (possibly delisted or invalid ticker)"
                logger.warning(f"FAILED: No data for {ticker}.")

    # Load existing failure log and merge
    if os.path.exists(FAILED_DOWNLOADS_JSON):
        try:
            with open(FAILED_DOWNLOADS_JSON, "r", encoding="utf-8") as f:
                old_failed = json.load(f)
            # Remove tickers that succeeded on this run
            for s in successful:
                old_failed.pop(s, None)
            old_failed.update(failed)
            failed_to_write = old_failed
        except Exception:
            failed_to_write = failed
    else:
        failed_to_write = failed

    with open(FAILED_DOWNLOADS_JSON, "w", encoding="utf-8") as f:
        json.dump(failed_to_write, f, indent=2)

    # Ingestion summary log
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_target_companies": len(all_tickers),
        "successful_count": len(successful),
        "skipped_from_cache": len(skipped_cached),
        "newly_downloaded": len(successful) - len(skipped_cached),
        "failed_count": len(failed_to_write),
        "output_directory": output_dir,
        "start_date": start_date,
        "end_date": end_date
    }

    with open(INGESTION_LOG_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"COMPANY INGESTION SUMMARY: {summary['successful_count']}/{summary['total_target_companies']} Successful, {summary['failed_count']} Failed")
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    # If run directly as script, ingest first 100 companies by default or accept count from CLI
    limit = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 300
    ingest_company_stocks(max_companies=limit)
