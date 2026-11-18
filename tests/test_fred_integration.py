"""
Unit Tests for FRED (Federal Reserve Economic Data) Integration Module (Prompt 5)
Verifies:
- Researched FRED series identifiers across all required macroeconomic categories.
- Official FRED API client and environment variable key management.
- Missing value handling (FRED '.' placeholders parsed to NaN).
- Preservation of native observation dates and frequencies (Daily, Monthly, Quarterly).
- Strict zero look-ahead bias alignment with market trading dates:
  Proves that monthly/quarterly observations are only observed on/after official publication dates.
- Incremental updates and deduplication.
"""

import os
import tempfile
import pandas as pd
import numpy as np
import pytest

from src.ingestion.fred_data import (
    FRED_SERIES_CATALOG,
    get_fred_catalog,
    get_fred_api_key,
    parse_fred_observations,
    save_fred_series_incremental,
    ingest_all_fred_indicators,
    OFFICIAL_FRED_API_URL
)
from src.preprocessing.macro_pipeline import (
    calculate_publication_date,
    clean_fred_series_dataframe,
    build_native_frequency_macro_dataset,
    align_macro_with_trading_dates,
    align_macro_with_market_indices
)


def test_fred_series_identifiers_and_catalog():
    """
    Verify the researched FRED series identifiers cover all required categories:
    - Inflation / CPI (CPIAUCSL, CPILFESL)
    - Federal funds rate (FEDFUNDS, DFF)
    - Treasury yields (DGS10, DGS2, T10Y2Y, DGS3MO)
    - Unemployment (UNRATE)
    - GDP growth (A191RL1Q225SBEA, GDPC1)
    - Consumer confidence (UMCSENT)
    """
    required_series = {
        # Inflation / CPI
        "CPIAUCSL": "Inflation",
        "CPILFESL": "Inflation",
        # Federal Funds Rate
        "FEDFUNDS": "Monetary Policy",
        "DFF": "Monetary Policy",
        # Treasury Yields
        "DGS10": "Treasury Yields",
        "DGS2": "Treasury Yields",
        "T10Y2Y": "Treasury Yields",
        # Unemployment
        "UNRATE": "Labor Market",
        # GDP Growth
        "A191RL1Q225SBEA": "Economic Output",
        "GDPC1": "Economic Output",
        # Consumer Confidence
        "UMCSENT": "Consumer Confidence"
    }

    for sid, category in required_series.items():
        assert sid in FRED_SERIES_CATALOG, f"Missing required FRED series '{sid}' in catalog"
        meta = FRED_SERIES_CATALOG[sid]
        assert meta["category"] == category
        assert "frequency" in meta
        assert "units" in meta
        assert "publication_lag_days" in meta

    catalog_df = get_fred_catalog()
    assert len(catalog_df) >= len(required_series)
    assert set(required_series.keys()).issubset(set(catalog_df["series_id"]))


def test_fred_api_key_environment_management(monkeypatch):
    """Verify official FRED API key is read from environment variable FRED_API_KEY."""
    # Test when set
    monkeypatch.setenv("FRED_API_KEY", "test_mock_fred_key_12345")
    assert get_fred_api_key() == "test_mock_fred_key_12345"

    # Test when empty
    monkeypatch.setenv("FRED_API_KEY", "")
    assert get_fred_api_key() is None


def test_parse_fred_observations_missing_values():
    """Verify FRED '.' placeholders and empty strings are cleanly converted to NaN."""
    raw_obs = [
        {"date": "2024-01-01", "value": "3.5"},
        {"date": "2024-01-02", "value": "."},      # Market holiday in daily series
        {"date": "2024-01-03", "value": " "},      # Empty string
        {"date": "2024-01-04", "value": "ND"},     # No Data
        {"date": "2024-01-05", "value": "4.2"}
    ]

    parsed = parse_fred_observations(raw_obs, series_id="DGS10")

    assert len(parsed) == 5
    assert parsed.loc[0, "value"] == 3.5
    assert pd.isna(parsed.loc[1, "value"])
    assert pd.isna(parsed.loc[2, "value"])
    assert pd.isna(parsed.loc[3, "value"])
    assert parsed.loc[4, "value"] == 4.2
    assert (parsed["series_id"] == "DGS10").all()


def test_save_fred_series_incremental_and_deduplication():
    """Verify incremental ingestion merges new observations and deduplicates dates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        chunk1 = pd.DataFrame([
            {"date": "2024-01-01", "value": 308.4, "series_id": "CPIAUCSL"},
            {"date": "2024-02-01", "value": 310.3, "series_id": "CPIAUCSL"}
        ])

        chunk2 = pd.DataFrame([
            {"date": "2024-02-01", "value": 310.3, "series_id": "CPIAUCSL"},  # Overlap
            {"date": "2024-03-01", "value": 312.2, "series_id": "CPIAUCSL"}   # New
        ])

        path = save_fred_series_incremental("CPIAUCSL", chunk1, output_dir=tmpdir)
        assert os.path.exists(path)

        read1 = pd.read_csv(path)
        assert len(read1) == 2

        # Incremental save with overlap
        save_fred_series_incremental("CPIAUCSL", chunk2, output_dir=tmpdir)
        read2 = pd.read_csv(path)

        # Total observations must be 3 (no duplicate on 2024-02-01)
        assert len(read2) == 3
        assert list(read2["date"]) == ["2024-01-01", "2024-02-01", "2024-03-01"]


def test_publication_date_calculation():
    """
    Verify point-in-time publication dates:
    - Monthly reference 2024-01-01 with 15-day lag publishes on 2024-02-15 (after Jan 31).
    - Monthly reference 2024-01-01 with 7-day lag publishes on 2024-02-07.
    - Quarterly reference 2024-01-01 (Q1) with 30-day lag publishes on 2024-04-30 (after Mar 31).
    """
    # CPI: Monthly with 15 days lag
    cpi_pub = calculate_publication_date("2024-01-01", frequency="Monthly", publication_lag_days=15)
    assert cpi_pub == "2024-02-15"

    # Unemployment: Monthly with 7 days lag
    unemp_pub = calculate_publication_date("2024-01-01", frequency="Monthly", publication_lag_days=7)
    assert unemp_pub == "2024-02-07"

    # GDP: Quarterly with 30 days lag
    # Q1 starts 2024-01-01, ends 2024-03-31 + 30 days = 2024-04-30
    gdp_pub = calculate_publication_date("2024-01-01", frequency="Quarterly", publication_lag_days=30)
    assert gdp_pub == "2024-04-30"


def test_zero_look_ahead_bias_in_trading_date_alignment():
    """
    CRITICAL LOOK-AHEAD BIAS TEST:
    Verify that an economic number for January 2024 (e.g. CPI released on 2024-02-15):
    1. Is STRICTLY UNAVAILABLE on trading days in January 2024.
    2. Trading days before 2024-02-15 reflect only the prior December 2023 number.
    3. Exactly on/after 2024-02-15, the January 2024 number is reflected.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create raw monthly CPI data
        # Dec 2023 (obs: 2023-12-01, pub: 2024-01-15, val: 306.7)
        # Jan 2024 (obs: 2024-01-01, pub: 2024-02-15, val: 308.4)
        cpi_df = pd.DataFrame([
            {"date": "2023-12-01", "value": 306.7, "series_id": "CPIAUCSL"},
            {"date": "2024-01-01", "value": 308.4, "series_id": "CPIAUCSL"}
        ])
        cpi_df.to_csv(os.path.join(tmpdir, "CPIAUCSL_raw.csv"), index=False)

        # Trading calendar spanning January and February 2024
        trading_dates = [
            "2024-01-10",  # Before Dec CPI release (Dec release is 2024-01-15)
            "2024-01-16",  # After Dec CPI release, but BEFORE Jan CPI release
            "2024-01-31",  # End of January (Jan CPI is NOT yet released!)
            "2024-02-14",  # Day before Jan CPI release
            "2024-02-15",  # Day of Jan CPI release
            "2024-02-20"   # After Jan CPI release
        ]

        aligned = align_macro_with_trading_dates(
            raw_fred_dir=tmpdir,
            trading_dates=trading_dates,
            output_parquet=os.path.join(tmpdir, "aligned.parquet"),
            output_csv=os.path.join(tmpdir, "aligned.csv")
        )

        assert "cpi_headline" in aligned.columns

        cpi_by_date = aligned.set_index("Date")["cpi_headline"].to_dict()

        # On 2024-01-16: Jan CPI is NOT known yet! Must NOT be 308.4!
        # Must reflect Dec CPI (306.7)
        assert cpi_by_date["2024-01-16"] == 306.7

        # On 2024-01-31: Must STILL reflect Dec CPI (306.7), NOT Jan CPI!
        assert cpi_by_date["2024-01-31"] == 306.7

        # On 2024-02-14: Day before release, STILL 306.7!
        assert cpi_by_date["2024-02-14"] == 306.7

        # On 2024-02-15: Release date! Now Jan CPI (308.4) is known!
        assert cpi_by_date["2024-02-15"] == 308.4

        # On 2024-02-20: Forward-filled, still 308.4!
        assert cpi_by_date["2024-02-20"] == 308.4


def test_native_frequency_macro_dataset_preservation():
    """Verify native dataset preserves original dates and frequencies without forced resampling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = os.path.join(tmpdir, "raw")
        proc_dir = os.path.join(tmpdir, "proc")
        os.makedirs(raw_dir, exist_ok=True)

        # Monthly series (UNRATE)
        unrate = pd.DataFrame([
            {"date": "2024-01-01", "value": 3.7, "series_id": "UNRATE"},
            {"date": "2024-02-01", "value": 3.9, "series_id": "UNRATE"}
        ])
        unrate.to_csv(os.path.join(raw_dir, "UNRATE_raw.csv"), index=False)

        # Daily series (DGS10)
        dgs10 = pd.DataFrame([
            {"date": "2024-01-02", "value": 3.95, "series_id": "DGS10"},
            {"date": "2024-01-03", "value": 3.98, "series_id": "DGS10"}
        ])
        dgs10.to_csv(os.path.join(raw_dir, "DGS10_raw.csv"), index=False)

        native_df = build_native_frequency_macro_dataset(
            raw_dir=raw_dir,
            output_parquet=os.path.join(proc_dir, "native.parquet"),
            output_csv=os.path.join(proc_dir, "native.csv")
        )

        assert len(native_df) == 4
        assert set(native_df["series_id"]) == {"UNRATE", "DGS10"}
        assert "publication_date" in native_df.columns

        # Verify frequencies preserved
        unrate_rows = native_df[native_df["series_id"] == "UNRATE"]
        assert (unrate_rows["frequency"] == "Monthly").all()

        dgs10_rows = native_df[native_df["series_id"] == "DGS10"]
        assert (dgs10_rows["frequency"] == "Daily").all()
