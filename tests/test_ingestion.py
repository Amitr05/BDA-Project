"""
Tests for Data Ingestion Module
"""

import os
import tempfile
import pandas as pd
import pytest

from src.ingestion.event_data import get_curated_events, save_curated_events, load_event_dataset
from src.ingestion.market_data import is_valid_local_csv


def test_curated_events_structure():
    """Verify curated event dataset contains required fields, valid dates, and 5-8 events."""
    df = get_curated_events()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 5 and len(df) <= 10

    required_cols = ["event_name", "event_date", "event_category", "short_description"]
    for col in required_cols:
        assert col in df.columns
        assert df[col].notnull().all()

    # Verify YYYY-MM-DD date format
    for date_str in df["event_date"]:
        assert len(date_str) == 10
        assert date_str[4] == "-" and date_str[7] == "-"


def test_save_and_load_curated_events():
    """Test saving curated events to a temporary file and loading back."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_csv = os.path.join(tmpdir, "events_test.csv")
        saved_path = save_curated_events(output_path=tmp_csv, overwrite=True)
        assert os.path.exists(saved_path)

        loaded_df = load_event_dataset(tmp_csv)
        assert len(loaded_df) >= 5
        assert "event_name" in loaded_df.columns


def test_is_valid_local_csv():
    """Test the validation function for detecting empty/corrupt vs valid market CSV files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Non-existent file
        assert not is_valid_local_csv(os.path.join(tmpdir, "nonexistent.csv"))

        # 2. Empty file
        empty_csv = os.path.join(tmpdir, "empty.csv")
        with open(empty_csv, "w") as f:
            f.write("")
        assert not is_valid_local_csv(empty_csv)

        # 3. Valid dummy CSV with required columns
        valid_csv = os.path.join(tmpdir, "valid.csv")
        rows = ["Date,Open,High,Low,Close,Volume\n"]
        for i in range(150):
            rows.append(f"2020-01-{i%28+1:02d},100,105,95,102,1000000\n")
        with open(valid_csv, "w") as f:
            f.writelines(rows)

        assert is_valid_local_csv(
            valid_csv,
            min_rows=100,
            start_date="2020-01-01",
            end_date="2020-01-28"
        )
