"""
Tests for Individual Company Stock Data Ingestion Module (Prompt 2)
Verifies:
- Multiple company downloads
- Missing / invalid ticker handling
- Duplicate record detection
- Schema consistency
- Incremental ingestion and caching
- Historical index membership and survivorship bias handling
- Existing index dataset compatibility
"""

import os
import json
import tempfile
import pandas as pd
import pytest

from src.ingestion.company_data import (
    load_company_registry,
    load_index_constituents,
    get_active_constituents,
    is_valid_stock_csv,
    fetch_single_ticker,
    fetch_stocks_batch,
    save_stock_data_incremental,
    ingest_company_stocks,
)
from src.ingestion.market_data import RAW_DATA_PATHS, is_valid_local_csv


def test_company_registry_and_constituents():
    """Verify company registry and constituents junction table load with required schemas."""
    companies = load_company_registry()
    assert isinstance(companies, pd.DataFrame)
    assert len(companies) >= 300  # Target 300-500 companies

    required_company_cols = {"company_id", "ticker", "company_name", "sector", "industry"}
    assert required_company_cols.issubset(set(companies.columns))

    constituents = load_index_constituents()
    assert isinstance(constituents, pd.DataFrame)
    assert len(constituents) >= len(companies)

    required_constituent_cols = {"constituent_id", "index_id", "ticker", "valid_from", "is_active"}
    assert required_constituent_cols.issubset(set(constituents.columns))


def test_historical_index_membership_survivorship_bias():
    """
    Test that active constituents correctly reflect historical join/leave dates,
    preventing survivorship bias.
    """
    # Test 1: S&P 500 in 2018 vs Tesla (Tesla was added in Dec 2020)
    sp500_2018 = get_active_constituents(as_of_date="2018-06-01", index_id="^GSPC")
    assert "AAPL" in sp500_2018
    assert "MSFT" in sp500_2018
    assert "TSLA" not in sp500_2018  # Tesla was NOT in S&P 500 in 2018!

    # Tesla should be in S&P 500 in 2021
    sp500_2021 = get_active_constituents(as_of_date="2021-06-01", index_id="^GSPC")
    assert "TSLA" in sp500_2021

    # Test 2: Dow Jones historical changes (e.g. INTC was in Dow in 2023, but delisted in Nov 2024)
    dow_2023 = get_active_constituents(as_of_date="2023-01-01", index_id="^DJI")
    assert "INTC" in dow_2023
    assert "NVDA" not in dow_2023

    dow_2025 = get_active_constituents(as_of_date="2025-01-01", index_id="^DJI")
    assert "NVDA" in dow_2025
    assert "INTC" not in dow_2025


def test_schema_consistency_and_fetch():
    """Verify downloaded stock data meets strict schema requirements."""
    df = fetch_single_ticker("AAPL", start_date="2024-01-01", end_date="2024-01-15")
    assert df is not None and not df.empty

    required_cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    assert (df["Ticker"] == "AAPL").all()
    assert pd.to_datetime(df["Date"]).is_monotonic_increasing
    assert not df["Close"].isna().any()


def test_missing_and_invalid_ticker_handling():
    """Verify that an invalid or delisted ticker is captured in failed records without crashing."""
    invalid_ticker = "NONEXISTENT_TICKER_XYZ_999"
    res = fetch_single_ticker(invalid_ticker, start_date="2024-01-01", end_date="2024-01-10")
    assert res is None

    batch_res = fetch_stocks_batch([invalid_ticker, "MSFT"], start_date="2024-01-01", end_date="2024-01-05")
    assert batch_res[invalid_ticker] is None
    assert batch_res["MSFT"] is not None and len(batch_res["MSFT"]) > 0


def test_duplicate_detection_and_incremental_saving():
    """Verify save_stock_data_incremental merges new rows and deduplicates dates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initial chunk: Day 1 and Day 2
        df_chunk1 = pd.DataFrame({
            "Date": ["2023-01-03", "2023-01-04"],
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.5, 102.5],
            "Adj Close": [101.5, 102.5],
            "Volume": [1000000, 1100000],
            "Ticker": ["MOCK", "MOCK"]
        })

        # Overlapping and new chunk: Day 2 (duplicate) and Day 3 (new)
        df_chunk2 = pd.DataFrame({
            "Date": ["2023-01-04", "2023-01-05"],
            "Open": [101.0, 103.0],
            "High": [103.0, 104.0],
            "Low": [100.0, 102.0],
            "Close": [102.5, 103.5],
            "Adj Close": [102.5, 103.5],
            "Volume": [1100000, 1200000],
            "Ticker": ["MOCK", "MOCK"]
        })

        path = save_stock_data_incremental("MOCK", df_chunk1, output_dir=tmpdir)
        assert os.path.exists(path)
        read_1 = pd.read_csv(path)
        assert len(read_1) == 2

        # Ingest overlapping chunk
        save_stock_data_incremental("MOCK", df_chunk2, output_dir=tmpdir)
        read_2 = pd.read_csv(path)

        # Total rows should be exactly 3 (no duplicate on 2023-01-04)
        assert len(read_2) == 3
        assert list(read_2["Date"]) == ["2023-01-03", "2023-01-04", "2023-01-05"]


def test_ingest_company_stocks_pipeline_configurable():
    """Test full company ingestion with configurable limit and failure logging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a small mock companies.csv
        mock_companies_csv = os.path.join(tmpdir, "mock_companies.csv")
        pd.DataFrame([
            {"company_id": "CMP_AAPL", "ticker": "AAPL", "company_name": "Apple Inc", "sector": "Technology", "industry": "Consumer Electronics"},
            {"company_id": "CMP_MSFT", "ticker": "MSFT", "company_name": "Microsoft Corp", "sector": "Technology", "industry": "Software"},
            {"company_id": "CMP_BAD", "ticker": "BAD_TICKER_XXX", "company_name": "Bad Co", "sector": "None", "industry": "None"},
        ]).to_csv(mock_companies_csv, index=False)

        stocks_dir = os.path.join(tmpdir, "stocks")
        summary = ingest_company_stocks(
            max_companies=3,
            output_dir=stocks_dir,
            companies_file=mock_companies_csv,
            start_date="2024-01-01",
            end_date="2024-01-10",
            batch_size=5,
            pause_between_batches=0.1
        )

        assert summary["total_target_companies"] == 3
        assert summary["successful_count"] == 2  # AAPL and MSFT
        assert summary["failed_count"] >= 1      # BAD_TICKER_XXX

        # Verify files created
        assert os.path.exists(os.path.join(stocks_dir, "AAPL_raw.csv"))
        assert os.path.exists(os.path.join(stocks_dir, "MSFT_raw.csv"))
        assert not os.path.exists(os.path.join(stocks_dir, "BAD_TICKER_XXX_raw.csv"))

        # Test caching on second run: should skip 2
        summary_2 = ingest_company_stocks(
            max_companies=3,
            output_dir=stocks_dir,
            companies_file=mock_companies_csv,
            start_date="2024-01-01",
            end_date="2024-01-10"
        )
        assert summary_2["skipped_from_cache"] == 2
        assert summary_2["newly_downloaded"] == 0


def test_existing_indices_and_vix_compatibility():
    """Verify all four existing index datasets and VIX remain valid and intact."""
    for ticker, path in RAW_DATA_PATHS.items():
        assert os.path.exists(path), f"Index dataset missing: {path}"
        assert is_valid_local_csv(path), f"Index dataset invalid or corrupted: {path}"
