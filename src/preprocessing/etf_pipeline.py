"""
Sector ETF Preprocessing Pipeline Module
Cleans raw sector ETF data and calculates daily financial metrics:
- Daily returns & percentage returns
- Volume percentage changes
- 21-day rolling volatility and annualized volatility
- Cumulative return trajectory
Saves individual ETF datasets and a consolidated multi-sector dataset to Parquet.
"""

import os
import logging
from typing import Optional, Dict, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_RAW_ETFS_DIR = "data/raw/etfs"
DEFAULT_PROCESSED_ETFS_DIR = "data/processed/etfs"
COMBINED_PROCESSED_PARQUET = "data/processed/etfs/sector_etfs_processed.parquet"
COMBINED_PROCESSED_CSV = "data/processed/etfs/sector_etfs_processed.csv"


def clean_and_sort_etf_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw ETF DataFrame:
    - Standardizes 'Date' to 'YYYY-MM-DD'
    - Eliminates duplicate dates
    - Removes rows where Date or Close is NaN
    - Ensures numeric conversion for price and volume columns
    - Sorts chronologically ascending
    """
    cleaned = df.copy()
    cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    cleaned = cleaned.dropna(subset=["Date", "Close"])
    cleaned = cleaned.drop_duplicates(subset=["Date"], keep="last").sort_values("Date").reset_index(drop=True)

    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # If Adj Close is missing or empty, mirror Close
    if "Adj Close" not in cleaned.columns or cleaned["Adj Close"].isna().all():
        cleaned["Adj Close"] = cleaned["Close"]

    return cleaned


def calculate_etf_financial_metrics(
    df: pd.DataFrame,
    volatility_window: int = 21
) -> pd.DataFrame:
    """
    Compute daily financial metrics over chronological trading days for an ETF:
    - Daily Return: (Close_t - Close_{t-1}) / Close_{t-1}
    - Daily Return Percentage: Daily Return * 100
    - Volume Change Percentage: (Volume_t - Volume_{t-1}) / Volume_{t-1} * 100
    - Rolling Volatility (21 trading days): StdDev(Daily Return) over 21-day window
    - Annualized Volatility: Rolling Volatility * sqrt(252) * 100
    - Cumulative Return Percentage: (Close_t - Close_0) / Close_0 * 100
    """
    enriched = clean_and_sort_etf_dataframe(df)
    if enriched.empty:
        return enriched

    close = enriched["Close"]
    volume = enriched["Volume"]

    # Daily Return
    daily_return = close.pct_change()
    daily_return_pct = daily_return * 100.0

    # Volume change percentage (safeguard division by zero)
    volume_change_pct = volume.pct_change() * 100.0
    volume_change_pct = volume_change_pct.replace([np.inf, -np.inf], np.nan)

    # Rolling Volatility
    rolling_vol = daily_return.rolling(window=volatility_window, min_periods=max(2, volatility_window // 2)).std()
    annualized_vol = rolling_vol * np.sqrt(252.0) * 100.0

    # Cumulative Return
    base_close = close.iloc[0] if len(close) > 0 and close.iloc[0] > 0 else np.nan
    cum_return_pct = ((close / base_close) - 1.0) * 100.0 if pd.notna(base_close) else np.nan

    enriched["daily_return"] = daily_return.round(6)
    enriched["daily_return_pct"] = daily_return_pct.round(4)
    enriched["volume_change_pct"] = volume_change_pct.round(4)
    enriched["rolling_volatility_21d"] = rolling_vol.round(6)
    enriched["annualized_volatility_21d"] = annualized_vol.round(4)
    enriched["cumulative_return_pct"] = cum_return_pct.round(4)

    return enriched


def process_single_etf(
    raw_csv_path: str,
    output_parquet_path: str,
    volatility_window: int = 21
) -> pd.DataFrame:
    """
    Process a single raw ETF CSV and export to Parquet.
    """
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw ETF CSV not found at {raw_csv_path}")

    raw_df = pd.read_csv(raw_csv_path)
    processed_df = calculate_etf_financial_metrics(raw_df, volatility_window=volatility_window)

    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    processed_df.to_parquet(output_parquet_path, engine="pyarrow", index=False)
    logger.info(f"Saved processed ETF dataset to {output_parquet_path} ({len(processed_df)} rows).")

    return processed_df


def run_etf_preprocessing_pipeline(
    raw_etfs_dir: str = DEFAULT_RAW_ETFS_DIR,
    output_dir: str = DEFAULT_PROCESSED_ETFS_DIR,
    save_combined: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Preprocess all raw sector ETF CSVs found in raw_etfs_dir.
    Saves:
    - data/processed/etfs/<ticker>_processed.parquet for each ETF
    - data/processed/etfs/sector_etfs_processed.parquet (combined long-format dataset)
    """
    os.makedirs(output_dir, exist_ok=True)
    processed_dfs: Dict[str, pd.DataFrame] = {}

    if not os.path.exists(raw_etfs_dir):
        logger.warning(f"Raw ETFs directory does not exist: {raw_etfs_dir}")
        return processed_dfs

    csv_files = [f for f in os.listdir(raw_etfs_dir) if f.endswith("_raw.csv") and f != "sector_etfs_raw.csv"]

    for csv_file in sorted(csv_files):
        ticker = csv_file.replace("_raw.csv", "")
        raw_path = os.path.join(raw_etfs_dir, csv_file)
        out_path = os.path.join(output_dir, f"{ticker}_processed.parquet")

        try:
            df = process_single_etf(raw_path, out_path)
            processed_dfs[ticker] = df
        except Exception as e:
            logger.error(f"Error processing ETF {ticker}: {e}")

    # Generate combined long-format processed Parquet & CSV
    if save_combined and processed_dfs:
        combined_df = pd.concat(list(processed_dfs.values()), ignore_index=True)
        combined_df = combined_df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

        os.makedirs(output_dir, exist_ok=True)
        combined_parquet_path = os.path.join(output_dir, "sector_etfs_processed.parquet")
        combined_csv_path = os.path.join(output_dir, "sector_etfs_processed.csv")
        combined_df.to_parquet(combined_parquet_path, engine="pyarrow", index=False)
        combined_df.to_csv(combined_csv_path, index=False)
        logger.info(f"Consolidated all processed ETFs to {combined_parquet_path} ({len(combined_df)} rows).")

    return processed_dfs


if __name__ == "__main__":
    run_etf_preprocessing_pipeline()
