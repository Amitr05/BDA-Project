"""
Sector ETF Data Ingestion Module
Collects historical OHLCV data for US sector benchmark ETFs:
- Technology: XLK
- Financials: XLF
- Energy: XLE
- Healthcare: XLV
- Industrials: XLI
- Consumer Discretionary: XLY
- Consumer Staples: XLP
- Utilities: XLU

Maintains schema consistency with individual stock data:
['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ticker', 'Sector']
Implements local caching, duplicate detection, incremental updates, and ingestion logging.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_START_DATE = "2015-01-01"
DEFAULT_END_DATE = "2026-02-01"
DEFAULT_ETFS_DIR = "data/raw/etfs"
COMBINED_ETFS_RAW_CSV = "data/raw/etfs/sector_etfs_raw.csv"
ETF_INGESTION_LOG_JSON = "data/raw/etf_ingestion_log.json"

# Proposed Sector ETFs mapping
SECTOR_ETFS: Dict[str, str] = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities"
}

ETF_METADATA: Dict[str, Dict[str, str]] = {
    "XLK": {
        "sector": "Technology",
        "fund_name": "Technology Select Sector SPDR Fund",
        "benchmark_index": "Technology Select Sector Index",
        "top_holdings": "MSFT, AAPL, NVDA"
    },
    "XLF": {
        "sector": "Financials",
        "fund_name": "Financial Select Sector SPDR Fund",
        "benchmark_index": "Financial Select Sector Index",
        "top_holdings": "BRK.B, JPM, V"
    },
    "XLE": {
        "sector": "Energy",
        "fund_name": "Energy Select Sector SPDR Fund",
        "benchmark_index": "Energy Select Sector Index",
        "top_holdings": "XOM, CVX, COP"
    },
    "XLV": {
        "sector": "Healthcare",
        "fund_name": "Health Care Select Sector SPDR Fund",
        "benchmark_index": "Health Care Select Sector Index",
        "top_holdings": "LLY, UNH, JNJ"
    },
    "XLI": {
        "sector": "Industrials",
        "fund_name": "Industrial Select Sector SPDR Fund",
        "benchmark_index": "Industrial Select Sector Index",
        "top_holdings": "GE, CAT, UNP"
    },
    "XLY": {
        "sector": "Consumer Discretionary",
        "fund_name": "Consumer Discretionary Select Sector SPDR Fund",
        "benchmark_index": "Consumer Discretionary Select Sector Index",
        "top_holdings": "AMZN, TSLA, HD"
    },
    "XLP": {
        "sector": "Consumer Staples",
        "fund_name": "Consumer Staples Select Sector SPDR Fund",
        "benchmark_index": "Consumer Staples Select Sector Index",
        "top_holdings": "PG, COST, WMT"
    },
    "XLU": {
        "sector": "Utilities",
        "fund_name": "Utilities Select Sector SPDR Fund",
        "benchmark_index": "Utilities Select Sector Index",
        "top_holdings": "NEE, SO, DUK"
    }
}

REQUIRED_STOCK_SCHEMA = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"]
REQUIRED_ETF_SCHEMA = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker", "Sector"]


def get_sector_etfs_metadata() -> pd.DataFrame:
    """Return a DataFrame describing all configured sector ETFs."""
    records = []
    for ticker, meta in ETF_METADATA.items():
        records.append({
            "ticker": ticker,
            "sector": meta["sector"],
            "fund_name": meta["fund_name"],
            "benchmark_index": meta["benchmark_index"],
            "top_holdings": meta["top_holdings"]
        })
    return pd.DataFrame(records)


def is_valid_etf_csv(
    file_path: str,
    min_rows: Optional[int] = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    tolerance_days: int = 14
) -> bool:
    """
    Validate that a local ETF CSV exists, conforms to the stock data schema,
    contains valid dates, and covers the required date range.
    """
    if not os.path.exists(file_path):
        return False

    try:
        if os.path.getsize(file_path) < 100:
            return False

        sample_df = pd.read_csv(file_path, nrows=5)
        # Consistent schema with individual stock data
        required_cols = set(REQUIRED_STOCK_SCHEMA)
        if not required_cols.issubset(set(sample_df.columns)):
            return False

        date_df = pd.read_csv(file_path, usecols=["Date"])
        requested_start = pd.Timestamp(start_date)
        requested_end = pd.Timestamp(end_date)
        span_days = max(1, (requested_end - requested_start).days)

        effective_min_rows = min_rows if min_rows is not None else max(1, min(50, int(span_days * 0.4)))

        if len(date_df) < effective_min_rows:
            return False

        dates = pd.to_datetime(date_df["Date"], errors="coerce")
        if dates.isna().any():
            return False

        actual_start = dates.min()
        actual_end = dates.max()

        eff_tolerance = min(tolerance_days, max(1, int(span_days * 0.2)))
        earliest_allowed = requested_start + pd.Timedelta(days=eff_tolerance)
        latest_allowed = requested_end - pd.Timedelta(days=eff_tolerance)

        if actual_start > earliest_allowed or actual_end < latest_allowed:
            return False

        return True
    except Exception as e:
        logger.debug(f"Validation check failed for {file_path}: {e}")
        return False


def fetch_single_etf(
    ticker: str,
    sector: Optional[str] = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    max_retries: int = 3,
    backoff_factor: float = 1.5
) -> Optional[pd.DataFrame]:
    """
    Download historical daily OHLCV data for a sector ETF with retry logic.
    Maintains a schema strictly consistent with individual stock data.
    """
    sector_name = sector or SECTOR_ETFS.get(ticker, "Unknown Sector")
    clean_ticker = ticker.replace(".", "-")

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Downloading ETF '{ticker}' ({sector_name}) from {start_date} to {end_date} (Attempt {attempt})...")
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
            df["Sector"] = sector_name

            # Enforce schema ordering
            available_cols = [c for c in REQUIRED_ETF_SCHEMA if c in df.columns]
            df = df[available_cols]

            df = df.dropna(subset=["Date", "Close"])
            df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)

            logger.info(f"Successfully retrieved {len(df)} records for ETF {ticker} ({df['Date'].min()} to {df['Date'].max()}).")
            return df
        except Exception as err:
            wait_time = backoff_factor ** attempt
            logger.warning(f"Attempt {attempt}/{max_retries} failed for ETF {ticker}: {err}. Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)

    return None


def fetch_etfs_batch(
    etf_map: Optional[Dict[str, str]] = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    batch_size: int = 8,
    pause_between_batches: float = 0.5,
    max_retries: int = 3
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Bulk download sector ETFs using yfinance with error containment.
    """
    targets = etf_map or SECTOR_ETFS
    tickers = list(targets.keys())
    results: Dict[str, Optional[pd.DataFrame]] = {}

    for i in range(0, len(tickers), batch_size):
        chunk_tickers = tickers[i : i + batch_size]
        logger.info(f"Downloading ETF chunk: {chunk_tickers}...")

        batch_success = False
        batch_df = None
        for attempt in range(1, max_retries + 1):
            try:
                batch_df = yf.download(
                    chunk_tickers,
                    start=start_date,
                    end=end_date,
                    group_by="ticker",
                    auto_adjust=False,
                    progress=False
                )
                batch_success = True
                break
            except Exception as e:
                wait_sec = 1.5 * attempt
                logger.warning(f"Batch ETF attempt {attempt} failed: {e}. Retrying in {wait_sec}s...")
                time.sleep(wait_sec)

        for ticker in chunk_tickers:
            sector_name = targets[ticker]
            ticker_df = None

            if batch_success and batch_df is not None and not batch_df.empty:
                try:
                    if len(chunk_tickers) == 1:
                        sub = batch_df
                    else:
                        sub = batch_df[ticker] if ticker in batch_df else None

                    if sub is not None and not sub.empty:
                        sub = sub.dropna(how="all").reset_index()
                        if "Date" not in sub.columns and len(sub.columns) > 0:
                            sub = sub.rename(columns={sub.columns[0]: "Date"})

                        if "Close" in sub.columns and not sub["Close"].dropna().empty:
                            sub["Date"] = pd.to_datetime(sub["Date"]).dt.strftime("%Y-%m-%d")
                            if "Adj Close" not in sub.columns:
                                sub["Adj Close"] = sub["Close"]
                            sub["Ticker"] = ticker
                            sub["Sector"] = sector_name

                            available = [c for c in REQUIRED_ETF_SCHEMA if c in sub.columns]
                            sub = sub[available].dropna(subset=["Date", "Close"])
                            sub = sub.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
                            if len(sub) > 0:
                                ticker_df = sub
                except Exception as parse_err:
                    logger.debug(f"Could not parse batch output for ETF {ticker}: {parse_err}")

            if ticker_df is None or ticker_df.empty:
                logger.info(f"Falling back to single ticker fetch for ETF {ticker}...")
                ticker_df = fetch_single_etf(ticker, sector=sector_name, start_date=start_date, end_date=end_date)

            results[ticker] = ticker_df

        if pause_between_batches > 0:
            time.sleep(pause_between_batches)

    return results


def save_etf_data_incremental(
    ticker: str,
    sector: str,
    new_df: pd.DataFrame,
    output_dir: str = DEFAULT_ETFS_DIR
) -> str:
    """
    Save or merge ETF data incrementally into data/raw/etfs/<ticker>_raw.csv,
    avoiding duplicate dates and preserving chronological sorting.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{ticker}_raw.csv")

    df_to_save = new_df.copy()
    if "Ticker" not in df_to_save.columns:
        df_to_save["Ticker"] = ticker
    if "Sector" not in df_to_save.columns:
        df_to_save["Sector"] = sector

    if os.path.exists(file_path):
        try:
            existing_df = pd.read_csv(file_path)
            combined_df = pd.concat([existing_df, df_to_save], ignore_index=True)
            combined_df["Date"] = pd.to_datetime(combined_df["Date"]).dt.strftime("%Y-%m-%d")
            combined_df = combined_df.drop_duplicates(subset=["Date"], keep="last").sort_values("Date").reset_index(drop=True)
            combined_df.to_csv(file_path, index=False)
            logger.info(f"Incrementally updated {file_path} (Total rows: {len(combined_df)}).")
            return file_path
        except Exception as e:
            logger.warning(f"Error merging incremental data for ETF {ticker}: {e}. Overwriting with fresh data.")

    df_to_save["Date"] = pd.to_datetime(df_to_save["Date"]).dt.strftime("%Y-%m-%d")
    df_to_save = df_to_save.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df_to_save.to_csv(file_path, index=False)
    logger.info(f"Saved fresh ETF dataset {file_path} ({len(df_to_save)} rows).")
    return file_path


def ingest_sector_etfs(
    etfs: Optional[Dict[str, str]] = None,
    output_dir: str = DEFAULT_ETFS_DIR,
    force_download: bool = False,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    save_combined_csv: bool = True
) -> Dict[str, Any]:
    """
    Main ingestion pipeline for sector ETFs:
    1. Checks local cache for existing valid raw CSVs.
    2. Downloads missing or stale ETFs via yfinance batch/single fetch.
    3. Incrementally merges new data while preventing duplicates.
    4. Generates a consolidated sector ETF CSV if requested.
    5. Writes an ingestion log summary.
    """
    target_etfs = etfs or SECTOR_ETFS
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"STARTING SECTOR ETF INGESTION ({len(target_etfs)} ETFs)")
    logger.info(f"Date Range: {start_date} to {end_date} | Output: {output_dir}")
    logger.info("=" * 60)

    to_download: Dict[str, str] = {}
    skipped_cached: List[str] = []

    for ticker, sector in target_etfs.items():
        file_path = os.path.join(output_dir, f"{ticker}_raw.csv")
        if not force_download and is_valid_etf_csv(file_path, start_date=start_date, end_date=end_date):
            skipped_cached.append(ticker)
        else:
            to_download[ticker] = sector

    logger.info(f"Cached/Skipped: {len(skipped_cached)} ETFs | Remaining to download: {len(to_download)}")

    successful: Dict[str, str] = {}
    failed: Dict[str, str] = {}

    for t in skipped_cached:
        successful[t] = os.path.join(output_dir, f"{t}_raw.csv")

    if to_download:
        batch_results = fetch_etfs_batch(
            etf_map=to_download,
            start_date=start_date,
            end_date=end_date
        )

        for ticker, df in batch_results.items():
            sector = to_download[ticker]
            if df is not None and not df.empty:
                path = save_etf_data_incremental(ticker, sector, df, output_dir=output_dir)
                successful[ticker] = path
            else:
                failed[ticker] = "No market data returned from Yahoo Finance"
                logger.error(f"Failed to ingest ETF {ticker} ({sector})")

    # Generate combined raw CSV if all/some succeeded
    combined_path = None
    if save_combined_csv and successful:
        dfs = []
        for ticker in sorted(successful.keys()):
            p = successful[ticker]
            if os.path.exists(p):
                dfs.append(pd.read_csv(p))
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df["Date"] = pd.to_datetime(combined_df["Date"]).dt.strftime("%Y-%m-%d")
            combined_df = combined_df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
            combined_path = COMBINED_ETFS_RAW_CSV
            os.makedirs(os.path.dirname(combined_path), exist_ok=True)
            combined_df.to_csv(combined_path, index=False)
            logger.info(f"Consolidated all sector ETFs into {combined_path} ({len(combined_df)} rows).")

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "target_count": len(target_etfs),
        "successful_count": len(successful),
        "skipped_from_cache": len(skipped_cached),
        "newly_downloaded": len(successful) - len(skipped_cached),
        "failed_count": len(failed),
        "successful_tickers": sorted(list(successful.keys())),
        "failed_tickers": failed,
        "combined_raw_csv": combined_path,
        "output_directory": output_dir,
        "start_date": start_date,
        "end_date": end_date
    }

    with open(ETF_INGESTION_LOG_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"SECTOR ETF INGESTION SUMMARY: {summary['successful_count']}/{summary['target_count']} Successful, {summary['failed_count']} Failed")
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    force = "--force-download" in sys.argv
    ingest_sector_etfs(force_download=force)
