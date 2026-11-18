"""
Unit Tests for Sector ETF Ingestion and Preprocessing Modules (Prompt 5)
Verifies:
- All 8 proposed sector ETFs configured and accessible (XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU).
- Schema consistency with individual stock data.
- Caching and incremental ingestion logic.
- Duplicate record detection and chronological ordering.
- PySpark/pandas preprocessing pipeline metrics (daily returns, rolling vol, cumulative return).
"""

import os
import tempfile
import pandas as pd
import numpy as np
import pytest

from src.ingestion.etf_data import (
    SECTOR_ETFS,
    ETF_METADATA,
    get_sector_etfs_metadata,
    is_valid_etf_csv,
    fetch_single_etf,
    save_etf_data_incremental,
    ingest_sector_etfs,
    REQUIRED_STOCK_SCHEMA,
    REQUIRED_ETF_SCHEMA
)
from src.preprocessing.etf_pipeline import (
    clean_and_sort_etf_dataframe,
    calculate_etf_financial_metrics,
    process_single_etf,
    run_etf_preprocessing_pipeline
)


def test_proposed_sector_etfs_catalog():
    """Verify all 8 specified sector ETFs are defined with complete metadata."""
    expected_etfs = {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLE": "Energy",
        "XLV": "Healthcare",
        "XLI": "Industrials",
        "XLY": "Consumer Discretionary",
        "XLP": "Consumer Staples",
        "XLU": "Utilities"
    }

    assert SECTOR_ETFS == expected_etfs

    meta_df = get_sector_etfs_metadata()
    assert len(meta_df) == 8
    assert set(meta_df["ticker"]) == set(expected_etfs.keys())
    assert set(meta_df["sector"]) == set(expected_etfs.values())

    for ticker in expected_etfs:
        assert ticker in ETF_METADATA
        assert "fund_name" in ETF_METADATA[ticker]
        assert "benchmark_index" in ETF_METADATA[ticker]


def test_etf_schema_consistency_with_stock_data():
    """
    Verify the ETF schema matches individual stock data columns:
    ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ticker']
    and additionally provides 'Sector'.
    """
    for col in REQUIRED_STOCK_SCHEMA:
        assert col in REQUIRED_ETF_SCHEMA, f"Required stock column '{col}' missing from ETF schema"

    assert "Sector" in REQUIRED_ETF_SCHEMA


def test_save_etf_data_incremental_and_deduplication():
    """Verify incremental saving correctly merges overlapping data and avoids duplicate dates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        chunk1 = pd.DataFrame({
            "Date": ["2024-01-02", "2024-01-03"],
            "Open": [190.0, 191.0],
            "High": [192.0, 193.0],
            "Low": [189.0, 190.0],
            "Close": [191.5, 192.5],
            "Adj Close": [191.5, 192.5],
            "Volume": [5000000, 5200000],
            "Ticker": ["XLK", "XLK"],
            "Sector": ["Technology", "Technology"]
        })

        chunk2 = pd.DataFrame({
            "Date": ["2024-01-03", "2024-01-04"],
            "Open": [191.0, 193.0],
            "High": [193.0, 194.0],
            "Low": [190.0, 192.0],
            "Close": [192.5, 193.5],
            "Adj Close": [192.5, 193.5],
            "Volume": [5200000, 5400000],
            "Ticker": ["XLK", "XLK"],
            "Sector": ["Technology", "Technology"]
        })

        # Save initial chunk
        file_path = save_etf_data_incremental("XLK", "Technology", chunk1, output_dir=tmpdir)
        assert os.path.exists(file_path)

        read1 = pd.read_csv(file_path)
        assert len(read1) == 2

        # Save second overlapping chunk
        save_etf_data_incremental("XLK", "Technology", chunk2, output_dir=tmpdir)
        read2 = pd.read_csv(file_path)

        # Total rows must be 3 (no duplicate on 2024-01-03)
        assert len(read2) == 3
        assert list(read2["Date"]) == ["2024-01-02", "2024-01-03", "2024-01-04"]
        assert (read2["Ticker"] == "XLK").all()
        assert (read2["Sector"] == "Technology").all()


def test_is_valid_etf_csv_validation():
    """Verify is_valid_etf_csv correctly verifies schema, dates, and size."""
    with tempfile.TemporaryDirectory() as tmpdir:
        valid_path = os.path.join(tmpdir, "XLK_raw.csv")
        dates = pd.date_range("2015-01-05", "2026-01-28", freq="B").strftime("%Y-%m-%d").tolist()
        df = pd.DataFrame({
            "Date": dates,
            "Open": [100.0] * len(dates),
            "High": [101.0] * len(dates),
            "Low": [99.0] * len(dates),
            "Close": [100.5] * len(dates),
            "Adj Close": [100.5] * len(dates),
            "Volume": [1000000] * len(dates),
            "Ticker": ["XLK"] * len(dates),
            "Sector": ["Technology"] * len(dates)
        })
        df.to_csv(valid_path, index=False)

        assert is_valid_etf_csv(valid_path, start_date="2015-01-01", end_date="2026-02-01")

        # Invalid schema file
        invalid_path = os.path.join(tmpdir, "invalid.csv")
        df[["Date", "Close"]].to_csv(invalid_path, index=False)
        assert not is_valid_etf_csv(invalid_path)


def test_ingest_sector_etfs_caching_and_mock():
    """Verify ingest_sector_etfs caches valid local files and skips downloading them."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Prepopulate fake valid CSVs for 2 ETFs
        dates = pd.date_range("2015-01-05", "2026-01-25", freq="B").strftime("%Y-%m-%d").tolist()
        mock_data = {
            "Date": dates,
            "Open": [100.0] * len(dates),
            "High": [102.0] * len(dates),
            "Low": [98.0] * len(dates),
            "Close": [101.0] * len(dates),
            "Adj Close": [101.0] * len(dates),
            "Volume": [1000000] * len(dates),
            "Ticker": ["XLK"] * len(dates),
            "Sector": ["Technology"] * len(dates)
        }
        pd.DataFrame(mock_data).to_csv(os.path.join(tmpdir, "XLK_raw.csv"), index=False)

        mock_data["Ticker"] = ["XLF"] * len(dates)
        mock_data["Sector"] = ["Financials"] * len(dates)
        pd.DataFrame(mock_data).to_csv(os.path.join(tmpdir, "XLF_raw.csv"), index=False)

        summary = ingest_sector_etfs(
            etfs={"XLK": "Technology", "XLF": "Financials"},
            output_dir=tmpdir,
            force_download=False,
            start_date="2015-01-01",
            end_date="2026-02-01"
        )

        assert summary["target_count"] == 2
        assert summary["skipped_from_cache"] == 2
        assert summary["newly_downloaded"] == 0
        assert summary["successful_count"] == 2


def test_etf_preprocessing_metrics_calculation():
    """Verify calculate_etf_financial_metrics calculates daily return, volume change, rolling volatility, and cumulative return."""
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    closes = [100.0, 102.0, 101.0, 105.0, 106.0]
    volumes = [1000, 1200, 900, 1500, 1400]

    raw_df = pd.DataFrame({
        "Date": dates,
        "Open": closes,
        "High": [c + 1.0 for c in closes],
        "Low": [c - 1.0 for c in closes],
        "Close": closes,
        "Adj Close": closes,
        "Volume": volumes,
        "Ticker": ["XLK"] * len(dates),
        "Sector": ["Technology"] * len(dates)
    })

    metrics_df = calculate_etf_financial_metrics(raw_df, volatility_window=3)

    assert "daily_return" in metrics_df.columns
    assert "daily_return_pct" in metrics_df.columns
    assert "volume_change_pct" in metrics_df.columns
    assert "rolling_volatility_21d" in metrics_df.columns
    assert "annualized_volatility_21d" in metrics_df.columns
    assert "cumulative_return_pct" in metrics_df.columns

    # Verify return on day 2: (102 - 100) / 100 = 0.02, 2.0%
    assert np.isclose(metrics_df.loc[1, "daily_return"], 0.02)
    assert np.isclose(metrics_df.loc[1, "daily_return_pct"], 2.0)

    # Verify cumulative return on last day: (106 - 100) / 100 = 6.0%
    assert np.isclose(metrics_df.loc[4, "cumulative_return_pct"], 6.0)

    # Verify volume change on day 2: (1200 - 1000) / 1000 = +20.0%
    assert np.isclose(metrics_df.loc[1, "volume_change_pct"], 20.0)


def test_etf_preprocessing_pipeline_roundtrip():
    """Verify run_etf_preprocessing_pipeline processes raw files and outputs Parquet."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = os.path.join(tmpdir, "raw")
        proc_dir = os.path.join(tmpdir, "processed")
        os.makedirs(raw_dir, exist_ok=True)

        dates = pd.date_range("2024-01-01", "2024-02-15", freq="B").strftime("%Y-%m-%d").tolist()
        df = pd.DataFrame({
            "Date": dates,
            "Open": [150.0] * len(dates),
            "High": [152.0] * len(dates),
            "Low": [149.0] * len(dates),
            "Close": [151.0] * len(dates),
            "Adj Close": [151.0] * len(dates),
            "Volume": [2000000] * len(dates),
            "Ticker": ["XLI"] * len(dates),
            "Sector": ["Industrials"] * len(dates)
        })
        df.to_csv(os.path.join(raw_dir, "XLI_raw.csv"), index=False)

        processed = run_etf_preprocessing_pipeline(raw_etfs_dir=raw_dir, output_dir=proc_dir, save_combined=True)

        assert "XLI" in processed
        assert os.path.exists(os.path.join(proc_dir, "XLI_processed.parquet"))
        assert os.path.exists(os.path.join(proc_dir, "sector_etfs_processed.parquet"))

        read_parquet = pd.read_parquet(os.path.join(proc_dir, "XLI_processed.parquet"))
        assert len(read_parquet) == len(dates)
        assert "daily_return_pct" in read_parquet.columns
