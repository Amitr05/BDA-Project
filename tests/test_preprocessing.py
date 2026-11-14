"""
Tests for PySpark Preprocessing Pipeline
"""

import os
import tempfile
import pytest
import pandas as pd
from pyspark.sql import SparkSession

from src.preprocessing.spark_session import get_spark_session, stop_spark_session
from src.preprocessing.market_pipeline import (
    clean_and_sort_market_data,
    calculate_market_metrics,
    save_dataframe_parquet,
)


@pytest.fixture(scope="module")
def spark():
    """Module-level SparkSession for fast testing."""
    session = get_spark_session("PySparkTests")
    yield session
    stop_spark_session(session)


def test_clean_and_sort_market_data(spark: SparkSession):
    """Verify deduplication, date casting, null dropping, and chronological ordering."""
    raw_data = [
        ("2021-01-05", 100.0, 105.0, 95.0, 102.0, 102.0, 50000.0),
        ("2021-01-04", 98.0, 101.0, 97.0, 99.0, 99.0, 45000.0),
        ("2021-01-04", 98.0, 101.0, 97.0, 99.0, 99.0, 45000.0),  # Duplicate
        ("2021-01-06", 103.0, 104.0, 101.0, None, None, 40000.0),  # Missing Close
        (None, 100.0, 100.0, 100.0, 100.0, 100.0, 1000.0),  # Missing Date
    ]
    cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = spark.createDataFrame(raw_data, cols)

    cleaned = clean_and_sort_market_data(df)
    collected = cleaned.collect()

    # Should have exactly 2 valid distinct dates: 2021-01-04 and 2021-01-05
    assert len(collected) == 2
    assert str(collected[0]["Date"]) == "2021-01-04"
    assert str(collected[1]["Date"]) == "2021-01-05"


def test_calculate_market_metrics(spark: SparkSession):
    """Verify daily returns, volume changes, and rolling volatility calculations."""
    data = [
        ("2021-01-01", 100.0, 1000.0),
        ("2021-01-02", 110.0, 1500.0),  # +10% return, +50% volume
        ("2021-01-03", 99.0, 1200.0),   # -10% return, -20% volume
        ("2021-01-04", 108.9, 1200.0),  # +10% return, 0% volume change
    ]
    cols = ["Date", "Close", "Volume"]
    pdf = pd.DataFrame(data, columns=cols)
    pdf["Open"] = pdf["Close"]
    pdf["High"] = pdf["Close"]
    pdf["Low"] = pdf["Close"]
    pdf["Adj Close"] = pdf["Close"]

    sdf = spark.createDataFrame(pdf)
    cleaned = clean_and_sort_market_data(sdf)
    enriched = calculate_market_metrics(cleaned, volatility_window=3)
    results = enriched.collect()

    # Day 1: First day, return should be None
    assert results[0]["daily_return"] is None
    assert results[0]["volume_change_pct"] is None

    # Day 2: Close went from 100 to 110 (+10%), Volume went from 1000 to 1500 (+50%)
    assert pytest.approx(results[1]["daily_return_pct"], rel=1e-3) == 10.0
    assert pytest.approx(results[1]["volume_change_pct"], rel=1e-3) == 50.0

    # Day 3: Close went from 110 to 99 (-10%), Volume went from 1500 to 1200 (-20%)
    assert pytest.approx(results[2]["daily_return_pct"], rel=1e-3) == -10.0
    assert pytest.approx(results[2]["volume_change_pct"], rel=1e-3) == -20.0

    # Check rolling volatility exists
    assert results[3]["rolling_volatility_21d"] is not None
    assert results[3]["rolling_volatility_21d"] > 0


def test_save_dataframe_parquet(spark: SparkSession):
    """Test saving PySpark DataFrame to Parquet with graceful Windows fallback handling."""
    data = [("2021-01-01", 100.0), ("2021-01-02", 105.0)]
    df = spark.createDataFrame(data, ["Date", "Close"])

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_output.parquet")
        result = save_dataframe_parquet(df, out_path)
        assert os.path.exists(result)

        # Verify reading Parquet via Pandas
        read_df = pd.read_parquet(result)
        assert len(read_df) == 2
        assert "Close" in read_df.columns
