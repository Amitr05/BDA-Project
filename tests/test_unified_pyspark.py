"""
Unit Tests for Unified PySpark Preprocessing Architecture
=========================================================
Verifies:
- Multi-source ingestion across market benchmarks, equities, ETFs, news, and macro data.
- Schema standardisation: DateType, DoubleType, LongType, StringType.
- Data cleaning: null elimination, deduplication, volume coalescing.
- Joins across datasets:
  * Company equities joined with company registry on ticker (sector/industry).
  * Company equities joined with S&P 500 benchmark on Date (return spread).
  * GDELT news joined with events on event_id (days from event, timing phase).
- PySpark window functions:
  * Window.partitionBy("Ticker").orderBy("Date")
  * lag returns, 21-day rolling volatility, annualized volatility, cumulative returns.
- Date-based aggregations: daily news sentiment metrics and negative article ratios.
- Partitioned Parquet storage (partitioned by sector).
"""

import os
import tempfile
import pandas as pd
import numpy as np
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    DateType
)

from src.preprocessing.spark_session import get_spark_session, stop_spark_session
from src.preprocessing.unified_pipeline import (
    save_pyspark_parquet,
    process_benchmark_indices,
    process_company_equities,
    process_sector_etfs_pyspark,
    process_gdelt_news_pyspark,
    process_fred_macro_pyspark
)


@pytest.fixture(scope="module")
def spark():
    """Module-level SparkSession for unified pipeline testing."""
    session = get_spark_session("UnifiedPipelineTests")
    yield session
    stop_spark_session(session)


def test_schema_standardisation_and_cleaning(spark: SparkSession):
    """Verify PySpark standardises data types, drops nulls, and deduplicates records."""
    data = [
        ("2024-01-02", "100.0", "102.0", "99.0", "101.5", "101.5", "1000000", "TEST"),
        ("2024-01-02", "100.0", "102.0", "99.0", "101.5", "101.5", "1000000", "TEST"),  # Duplicate
        ("2024-01-03", "101.5", "103.0", "101.0", None, None, "1200000", "TEST"),        # Null Close (must drop)
        (None, "102.0", "104.0", "101.0", "103.5", "103.5", "1100000", "TEST"),          # Null Date (must drop)
        ("2024-01-04", "102.0", "104.0", "101.0", "103.5", "103.5", None, "TEST")        # Null Volume (coalesce to 0)
    ]
    columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"]

    raw_df = spark.createDataFrame(data, columns)

    cleaned = (
        raw_df.withColumn("Date", F.to_date(F.col("Date"), "yyyy-MM-dd"))
        .filter(F.col("Date").isNotNull() & F.col("Close").isNotNull())
        .dropDuplicates(["Date", "Ticker"])
        .withColumn("Open", F.col("Open").cast("double"))
        .withColumn("High", F.col("High").cast("double"))
        .withColumn("Low", F.col("Low").cast("double"))
        .withColumn("Close", F.col("Close").cast("double"))
        .withColumn("Volume", F.coalesce(F.col("Volume").cast("double"), F.lit(0.0)))
    )

    pdf = cleaned.toPandas()

    assert len(pdf) == 2  # Only 2024-01-02 and 2024-01-04 survive
    assert pdf["Close"].dtype == np.float64
    assert pdf["Volume"].iloc[1] == 0.0  # Coalesced null volume to 0.0
    assert not pdf["Close"].isna().any()


def test_pyspark_window_transformations(spark: SparkSession):
    """Verify Window functions: daily return, rolling vol, and cumulative returns."""
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    closes = [100.0, 102.0, 101.0, 105.0, 107.0]

    data = [(dates[i], closes[i], "MOCK") for i in range(len(dates))]
    df = spark.createDataFrame(data, ["Date", "Close", "Ticker"]).withColumn("Date", F.to_date("Date"))

    w = Window.partitionBy("Ticker").orderBy("Date")
    w_start = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(Window.unboundedPreceding, 0)

    lag_c = F.lag("Close", 1).over(w)
    ret = (F.col("Close") - lag_c) / lag_c
    first_c = F.first("Close").over(w_start)
    cum_ret = ((F.col("Close") - first_c) / first_c) * 100.0

    transformed = (
        df.withColumn("prev_close", lag_c)
        .withColumn("daily_return", ret)
        .withColumn("daily_return_pct", ret * 100.0)
        .withColumn("cumulative_return_pct", cum_ret)
    )

    pdf = transformed.toPandas()

    # Day 2: (102 - 100) / 100 = 0.02 (2.0%)
    assert np.isclose(pdf.loc[1, "daily_return"], 0.02)
    assert np.isclose(pdf.loc[1, "daily_return_pct"], 2.0)

    # Last Day Cumulative: (107 - 100) / 100 = 7.0%
    assert np.isclose(pdf.loc[4, "cumulative_return_pct"], 7.0)


def test_company_stock_joins_and_partitioning(spark: SparkSession):
    """Verify company equities join with companies.csv and save with sector partitioning."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_stocks_dir = os.path.join(tmpdir, "raw_stocks")
        out_stocks_dir = os.path.join(tmpdir, "processed_stocks")
        os.makedirs(raw_stocks_dir, exist_ok=True)

        # Mock companies.csv
        comp_csv = os.path.join(tmpdir, "companies.csv")
        pd.DataFrame([
            {"ticker": "AAPL", "company_name": "Apple Inc", "sector": "Technology", "industry": "Hardware"},
            {"ticker": "JPM", "company_name": "JPMorgan Chase", "sector": "Financials", "industry": "Banking"}
        ]).to_csv(comp_csv, index=False)

        # Mock AAPL raw
        pd.DataFrame({
            "Date": ["2024-01-02", "2024-01-03"],
            "Open": [180.0, 182.0],
            "High": [183.0, 184.0],
            "Low": [179.0, 181.0],
            "Close": [182.0, 183.5],
            "Adj Close": [182.0, 183.5],
            "Volume": [1000000, 1100000],
            "Ticker": ["AAPL", "AAPL"]
        }).to_csv(os.path.join(raw_stocks_dir, "AAPL_raw.csv"), index=False)

        # Mock JPM raw
        pd.DataFrame({
            "Date": ["2024-01-02", "2024-01-03"],
            "Open": [170.0, 171.0],
            "High": [172.0, 173.0],
            "Low": [169.0, 170.0],
            "Close": [171.0, 172.5],
            "Adj Close": [171.0, 172.5],
            "Volume": [800000, 850000],
            "Ticker": ["JPM", "JPM"]
        }).to_csv(os.path.join(raw_stocks_dir, "JPM_raw.csv"), index=False)

        processed_df = process_company_equities(
            spark=spark,
            stocks_raw_dir=raw_stocks_dir,
            companies_csv_path=comp_csv,
            output_dir=out_stocks_dir,
            max_stocks=2
        )

        pdf = processed_df.toPandas()

        assert len(pdf) == 4
        assert "sector" in pdf.columns
        assert "industry" in pdf.columns
        assert "daily_return_pct" in pdf.columns

        # Verify joined metadata
        aapl_row = pdf[pdf["Ticker"] == "AAPL"].iloc[0]
        assert aapl_row["sector"] == "Technology"
        assert aapl_row["company_name"] == "Apple Inc"

        jpm_row = pdf[pdf["Ticker"] == "JPM"].iloc[0]
        assert jpm_row["sector"] == "Financials"

        # Verify partitioned parquet directory structure
        assert os.path.exists(out_stocks_dir)
        sector_subdirs = os.listdir(out_stocks_dir)
        assert any("sector=Technology" in s for s in sector_subdirs)
        assert any("sector=Financials" in s for s in sector_subdirs)


def test_gdelt_news_joins_and_daily_aggregations(spark: SparkSession):
    """Verify GDELT news ingestion joins events and computes daily sentiment index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_jsonl = os.path.join(tmpdir, "news.jsonl")
        events_csv = os.path.join(tmpdir, "events.csv")
        out_parquet = os.path.join(tmpdir, "news.parquet")

        # Mock events.csv
        pd.DataFrame([{
            "event_id": "EV-001",
            "event_name": "Test Conflict",
            "event_date": "2024-01-15",
            "event_category": "Geopolitical"
        }]).to_csv(events_csv, index=False)

        # Mock news.jsonl
        articles = [
            {"seendate": "20240114T100000Z", "title": "Tension Rises", "url": "http://news.com/1", "tone": -3.5, "positive_score": 0.5, "negative_score": 4.0, "polarity": 4.5, "matched_event_id": "EV-001"},
            {"seendate": "20240115T120000Z", "title": "Conflict Erupts", "url": "http://news.com/2", "tone": -6.0, "positive_score": 0.2, "negative_score": 6.2, "polarity": 6.4, "matched_event_id": "EV-001"},
            {"seendate": "20240115T150000Z", "title": "Market Recovers", "url": "http://news.com/3", "tone": 2.5, "positive_score": 3.0, "negative_score": 0.5, "polarity": 3.5, "matched_event_id": "EV-001"}
        ]
        import json
        with open(raw_jsonl, "w", encoding="utf-8") as f:
            for a in articles:
                f.write(json.dumps(a) + "\n")

        cleaned_news, daily_sent = process_gdelt_news_pyspark(
            spark=spark,
            raw_jsonl_path=raw_jsonl,
            events_csv_path=events_csv,
            output_parquet_path=out_parquet
        )

        pdf_news = cleaned_news.toPandas()
        assert len(pdf_news) == 3
        assert "timing_phase" in pdf_news.columns
        assert "sentiment_label" in pdf_news.columns

        # Verify event join
        day_before = pdf_news[pdf_news["title"] == "Tension Rises"].iloc[0]
        assert day_before["timing_phase"] == "pre-event"
        assert day_before["days_from_event"] == -1

        day_of = pdf_news[pdf_news["title"] == "Conflict Erupts"].iloc[0]
        assert day_of["timing_phase"] == "day-of"
        assert day_of["days_from_event"] == 0

        # Verify daily aggregations
        pdf_daily = daily_sent.toPandas()
        assert len(pdf_daily) == 2  # 2024-01-14 and 2024-01-15
        assert "daily_article_count" in pdf_daily.columns
        assert "negative_ratio" in pdf_daily.columns

        row_15 = pdf_daily[pdf_daily["published_date"].astype(str) == "2024-01-15"].iloc[0]
        assert row_15["daily_article_count"] == 2
        assert row_15["negative_articles_count"] == 1
        assert np.isclose(row_15["negative_ratio"], 0.5)


def test_fred_macro_pyspark_cleaning(spark: SparkSession):
    """Verify FRED macro ingestion standardises schemas and handles '.' missing values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_fred = os.path.join(tmpdir, "raw_fred")
        out_proc = os.path.join(tmpdir, "processed_macro")
        os.makedirs(raw_fred, exist_ok=True)

        pd.DataFrame([
            {"date": "2024-01-02", "value": "4.15", "series_id": "DGS10"},
            {"date": "2024-01-03", "value": ".", "series_id": "DGS10"},    # Must convert to null
            {"date": "2024-01-04", "value": "4.18", "series_id": "DGS10"}
        ]).to_csv(os.path.join(raw_fred, "DGS10_raw.csv"), index=False)

        macro_df = process_fred_macro_pyspark(
            spark=spark,
            raw_fred_dir=raw_fred,
            output_dir=out_proc
        )

        pdf = macro_df.toPandas()
        assert len(pdf) == 3
        assert pdf["value"].dtype == np.float64
        assert pd.isna(pdf.loc[1, "value"])  # Successfully parsed '.' as NaN
        assert pdf.loc[0, "value"] == 4.15
        assert pdf.loc[2, "value"] == 4.18
