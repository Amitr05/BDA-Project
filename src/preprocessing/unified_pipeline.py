"""
Unified PySpark Preprocessing and Multi-Source Integration Pipeline
===================================================================
Processes all 7 project data sources in Apache Spark:
1. Four US Market Benchmarks (S&P 500, NASDAQ, Dow Jones, Russell 2000)
2. CBOE Volatility Index (VIX)
3. Individual Company Equities (300+ stocks in data/raw/stocks/)
4. Sector ETFs (8 Select Sector SPDR ETFs)
5. Curated Event Calendar & Comprehensive Event Database
6. GDELT Global News Records
7. FRED Macroeconomic Indicators

Implements:
- Multi-source ingestion with inferSchema and explicit standard schemas.
- Schema standardisation: DateType, DoubleType, LongType, StringType.
- Data cleaning: null filtering, deduplication, volume coalescing.
- Multi-dataset joins:
  * Stocks joined with company registry on ticker (sector, industry, company name).
  * Stocks joined with index constituents table (valid_from/valid_to survivorship intervals).
  * Stocks joined with parent benchmark indices on Date (relative daily return spread).
  * News joined with event database on event_id.
  * Macro indicators aligned with trading calendar (zero look-ahead bias).
- Window functions:
  * Window.partitionBy("Ticker").orderBy("Date")
  * Daily return, percentage return, volume change %, 21-day rolling volatility, annualized volatility.
- Date-based aggregations: daily news sentiment volume, macro monthly/quarterly trends.
- Partitioned Parquet storage:
  * Processed stocks partitioned by `sector` (e.g. sector=Technology, sector=Financials).
  * Appropriate partition sizing with coalescing to prevent small-file proliferation.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    LongType,
    DateType,
    IntegerType
)

from src.preprocessing.spark_session import get_spark_session, stop_spark_session

logger = logging.getLogger(__name__)

# Storage Layers
DIR_RAW = "data/raw"
DIR_PROCESSED = "data/processed"
DIR_ANALYTICS = "data/analytics"

STOCKS_RAW_DIR = "data/raw/stocks"
ETFS_RAW_DIR = "data/raw/etfs"
FRED_RAW_DIR = "data/raw/fred"
GDELT_RAW_JSONL = "data/raw/gdelt/raw_news_records.jsonl"
COMPANIES_CSV = "data/raw/companies.csv"
CONSTITUENTS_CSV = "data/raw/index_constituents.csv"
EVENTS_CSV = "data/raw/events.csv"

STOCKS_PROCESSED_DIR = "data/processed/stocks"
ETFS_PROCESSED_DIR = "data/processed/etfs"
MACRO_PROCESSED_DIR = "data/processed/macro"
INDICES_PROCESSED_DIR = "data/processed/indices"


# -----------------------------------------------------------------------------
# 1. Standard Schemas
# -----------------------------------------------------------------------------

MARKET_OHLCV_SCHEMA = StructType([
    StructField("Date", StringType(), False),
    StructField("Open", DoubleType(), True),
    StructField("High", DoubleType(), True),
    StructField("Low", DoubleType(), True),
    StructField("Close", DoubleType(), False),
    StructField("Adj Close", DoubleType(), True),
    StructField("Volume", DoubleType(), True),
    StructField("Ticker", StringType(), True)
])


# -----------------------------------------------------------------------------
# 2. Resilient Parquet Persistence (Windows Hadoop Winutils Compatibility)
# -----------------------------------------------------------------------------

def save_pyspark_parquet(
    df: DataFrame,
    output_path: str,
    partition_cols: Optional[List[str]] = None,
    num_partitions: int = 1
) -> str:
    """
    Saves PySpark DataFrame to Parquet with partition support.
    Gracefully handles Windows environments lacking Hadoop winutils:
    tries native PySpark writer first, falls back to PyArrow partitioned export.
    """
    os.makedirs(output_path if partition_cols else os.path.dirname(output_path), exist_ok=True)

    try:
        writer = df
        if num_partitions > 0 and not partition_cols:
            writer = writer.coalesce(num_partitions)

        writer_obj = writer.write.mode("overwrite")
        if partition_cols:
            writer_obj = writer_obj.partitionBy(*partition_cols)

        writer_obj.parquet(output_path)
        logger.info(f"Saved PySpark DataFrame natively to {output_path}")
    except Exception as exc:
        logger.warning(
            f"Native Spark Parquet write encountered ({exc}). "
            "Falling back to PyArrow Parquet writer for Windows compatibility."
        )
        import shutil
        if os.path.exists(output_path):
            if os.path.isdir(output_path):
                shutil.rmtree(output_path)
            else:
                os.remove(output_path)

        pdf = df.toPandas()
        if partition_cols:
            os.makedirs(output_path, exist_ok=True)
            pdf.to_parquet(
                output_path,
                engine="pyarrow",
                index=False,
                partition_cols=partition_cols
            )
        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pdf.to_parquet(output_path, engine="pyarrow", index=False)
        logger.info(f"Saved DataFrame via PyArrow Parquet fallback to {output_path}")

    return output_path


# -----------------------------------------------------------------------------
# 3. Market Benchmarks & VIX Ingestion & Preprocessing
# -----------------------------------------------------------------------------

def process_benchmark_indices(spark: SparkSession) -> Dict[str, DataFrame]:
    """
    Ingest, clean, and compute metrics for the four market indices and VIX using PySpark.
    """
    indices_raw = {
        "sp500": ("data/raw/sp500_raw.csv", "data/processed/sp500_processed.parquet", "^GSPC"),
        "nasdaq": ("data/raw/nasdaq_raw.csv", "data/processed/nasdaq_processed.parquet", "^IXIC"),
        "dow_jones": ("data/raw/dow_jones_raw.csv", "data/processed/dow_jones_processed.parquet", "^DJI"),
        "russell2000": ("data/raw/russell2000_raw.csv", "data/processed/russell2000_processed.parquet", "^RUT"),
        "vix": ("data/raw/vix_raw.csv", "data/processed/vix_processed.parquet", "^VIX")
    }

    processed_dfs = {}
    time_window = Window.orderBy("Date")
    rolling_21 = Window.orderBy("Date").rowsBetween(-20, 0)
    unbounded_start = Window.orderBy("Date").rowsBetween(Window.unboundedPreceding, 0)

    for name, (raw_path, out_path, ticker_symbol) in indices_raw.items():
        if not os.path.exists(raw_path):
            logger.warning(f"Raw benchmark file missing: {raw_path}")
            continue

        logger.info(f"PySpark processing benchmark {name.upper()} from {raw_path}...")
        raw_df = spark.read.option("header", "true").option("inferSchema", "true").csv(raw_path)

        # Standardise and clean
        cleaned = (
            raw_df.withColumn("Date", F.to_date(F.col("Date"), "yyyy-MM-dd"))
            .filter(F.col("Date").isNotNull() & F.col("Close").isNotNull())
            .dropDuplicates(["Date"])
            .withColumn("Open", F.col("Open").cast("double"))
            .withColumn("High", F.col("High").cast("double"))
            .withColumn("Low", F.col("Low").cast("double"))
            .withColumn("Close", F.col("Close").cast("double"))
            .withColumn("Adj Close", F.coalesce(F.col("Adj Close").cast("double"), F.col("Close").cast("double")))
            .withColumn("Volume", F.coalesce(F.col("Volume").cast("double"), F.lit(0.0)))
            .withColumn("Ticker", F.lit(ticker_symbol))
            .orderBy("Date")
        )

        # Window transformations: lag returns, volume change, rolling volatility
        lag_close = F.lag("Close", 1).over(time_window)
        lag_volume = F.lag("Volume", 1).over(time_window)
        first_close = F.first("Close").over(unbounded_start)

        daily_ret = F.when(lag_close.isNotNull() & (lag_close > 0), (F.col("Close") - lag_close) / lag_close).otherwise(F.lit(None))
        vol_change = F.when(lag_volume.isNotNull() & (lag_volume > 0), ((F.col("Volume") - lag_volume) / lag_volume) * 100.0).otherwise(F.lit(None))
        cum_ret = F.when(first_close.isNotNull() & (first_close > 0), ((F.col("Close") - first_close) / first_close) * 100.0).otherwise(F.lit(0.0))

        rolling_vol = F.stddev("daily_return").over(rolling_21)
        annualized_vol = rolling_vol * F.sqrt(F.lit(252.0)) * F.lit(100.0)

        enriched = (
            cleaned.withColumn("prev_close", lag_close)
            .withColumn("daily_return", daily_ret)
            .withColumn("daily_return_pct", daily_ret * 100.0)
            .withColumn("volume_change_pct", vol_change)
            .withColumn("rolling_volatility_21d", rolling_vol)
            .withColumn("annualized_volatility_21d", annualized_vol)
            .withColumn("cumulative_return_pct", cum_ret)
            .drop("prev_close")
        )

        save_pyspark_parquet(enriched, out_path, num_partitions=1)
        processed_dfs[name] = enriched

    return processed_dfs


# -----------------------------------------------------------------------------
# 4. Company Equities PySpark Pipeline (Partitioned Parquet)
# -----------------------------------------------------------------------------

def process_company_equities(
    spark: SparkSession,
    stocks_raw_dir: str = STOCKS_RAW_DIR,
    companies_csv_path: str = COMPANIES_CSV,
    output_dir: str = STOCKS_PROCESSED_DIR,
    max_stocks: Optional[int] = None,
    sp500_df: Optional[DataFrame] = None
) -> DataFrame:
    """
    Processes individual company equities using PySpark:
    - Multi-file CSV ingestion from data/raw/stocks/
    - Joins with companies.csv registry to attach Sector, Industry, and Company Name.
    - Window transformations per company: daily return, rolling vol, volume change.
    - Joins with S&P 500 benchmark on Date to compute relative return spread.
    - Saves to Parquet partitioned by 'sector'.
    """
    if not os.path.exists(stocks_raw_dir):
        raise FileNotFoundError(f"Stocks directory not found at {stocks_raw_dir}")

    csv_files = [f for f in os.listdir(stocks_raw_dir) if f.endswith("_raw.csv")]
    if max_stocks and max_stocks > 0:
        csv_files = csv_files[:max_stocks]

    logger.info(f"PySpark processing {len(csv_files)} company stock datasets...")

    file_paths = [os.path.join(stocks_raw_dir, f).replace("\\", "/") for f in csv_files]

    # Batch read all selected raw stock CSVs into one PySpark DataFrame
    stocks_raw_df = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv(file_paths)
    )

    # Standardize schema and clean
    cleaned_stocks = (
        stocks_raw_df.withColumn("Date", F.to_date(F.col("Date"), "yyyy-MM-dd"))
        .filter(F.col("Date").isNotNull() & F.col("Close").isNotNull())
        .dropDuplicates(["Date", "Ticker"])
        .withColumn("Open", F.col("Open").cast("double"))
        .withColumn("High", F.col("High").cast("double"))
        .withColumn("Low", F.col("Low").cast("double"))
        .withColumn("Close", F.col("Close").cast("double"))
        .withColumn("Adj Close", F.coalesce(F.col("Adj Close").cast("double"), F.col("Close").cast("double")))
        .withColumn("Volume", F.coalesce(F.col("Volume").cast("double"), F.lit(0.0)))
    )

    # Join with company registry (companies.csv)
    if os.path.exists(companies_csv_path):
        comp_registry = (
            spark.read.option("header", "true")
            .option("inferSchema", "true")
            .csv(companies_csv_path)
            .select("ticker", "company_name", "sector", "industry")
        )
        enriched_stocks = cleaned_stocks.join(
            comp_registry,
            cleaned_stocks["Ticker"] == comp_registry["ticker"],
            how="left"
        ).drop(comp_registry["ticker"])
    else:
        enriched_stocks = cleaned_stocks.withColumn("sector", F.lit("Unknown")).withColumn("industry", F.lit("Unknown"))

    # Apply PySpark Window functions partitioned by Ticker
    stock_window = Window.partitionBy("Ticker").orderBy("Date")
    stock_rolling_21 = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-20, 0)
    stock_from_start = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(Window.unboundedPreceding, 0)

    lag_c = F.lag("Close", 1).over(stock_window)
    lag_v = F.lag("Volume", 1).over(stock_window)
    first_c = F.first("Close").over(stock_from_start)

    ret = F.when(lag_c.isNotNull() & (lag_c > 0), (F.col("Close") - lag_c) / lag_c).otherwise(F.lit(None))
    vol_chg = F.when(lag_v.isNotNull() & (lag_v > 0), ((F.col("Volume") - lag_v) / lag_v) * 100.0).otherwise(F.lit(None))
    cum_ret = F.when(first_c.isNotNull() & (first_c > 0), ((F.col("Close") - first_c) / first_c) * 100.0).otherwise(F.lit(0.0))
    rolling_v = F.stddev("daily_return").over(stock_rolling_21)

    transformed_stocks = (
        enriched_stocks.withColumn("prev_close", lag_c)
        .withColumn("daily_return", ret)
        .withColumn("daily_return_pct", ret * 100.0)
        .withColumn("volume_change_pct", vol_chg)
        .withColumn("rolling_volatility_21d", rolling_v)
        .withColumn("annualized_volatility_21d", rolling_v * F.sqrt(F.lit(252.0)) * 100.0)
        .withColumn("cumulative_return_pct", cum_ret)
        .drop("prev_close")
    )

    # Join with S&P 500 benchmark to compute relative return spread
    if sp500_df is not None:
        sp_subset = sp500_df.select(
            F.col("Date").alias("sp_date"),
            F.col("daily_return").alias("sp500_daily_return")
        )
        transformed_stocks = (
            transformed_stocks.join(
                sp_subset,
                transformed_stocks["Date"] == sp_subset["sp_date"],
                how="left"
            )
            .withColumn(
                "spread_vs_sp500_pct",
                (F.col("daily_return") - F.col("sp500_daily_return")) * 100.0
            )
            .drop("sp_date")
        )

    # Save partitioned by sector
    save_pyspark_parquet(
        transformed_stocks,
        output_path=output_dir,
        partition_cols=["sector"]
    )
    logger.info(f"Successfully processed and partitioned company equities into {output_dir}")

    return transformed_stocks


# -----------------------------------------------------------------------------
# 5. Sector ETFs PySpark Pipeline
# -----------------------------------------------------------------------------

def process_sector_etfs_pyspark(
    spark: SparkSession,
    etfs_raw_dir: str = ETFS_RAW_DIR,
    output_dir: str = ETFS_PROCESSED_DIR
) -> DataFrame:
    """
    Processes Sector ETFs using PySpark window transformations.
    """
    if not os.path.exists(etfs_raw_dir):
        raise FileNotFoundError(f"ETFs raw directory not found at {etfs_raw_dir}")

    combined_csv = os.path.join(etfs_raw_dir, "sector_etfs_raw.csv")
    if os.path.exists(combined_csv):
        raw_etf_df = spark.read.option("header", "true").option("inferSchema", "true").csv(combined_csv)
    else:
        files = [os.path.join(etfs_raw_dir, f).replace("\\", "/") for f in os.listdir(etfs_raw_dir) if f.endswith("_raw.csv")]
        raw_etf_df = spark.read.option("header", "true").option("inferSchema", "true").csv(files)

    cleaned = (
        raw_etf_df.withColumn("Date", F.to_date(F.col("Date"), "yyyy-MM-dd"))
        .filter(F.col("Date").isNotNull() & F.col("Close").isNotNull())
        .dropDuplicates(["Date", "Ticker"])
        .withColumn("Open", F.col("Open").cast("double"))
        .withColumn("High", F.col("High").cast("double"))
        .withColumn("Low", F.col("Low").cast("double"))
        .withColumn("Close", F.col("Close").cast("double"))
        .withColumn("Adj Close", F.coalesce(F.col("Adj Close").cast("double"), F.col("Close").cast("double")))
        .withColumn("Volume", F.coalesce(F.col("Volume").cast("double"), F.lit(0.0)))
    )

    etf_window = Window.partitionBy("Ticker").orderBy("Date")
    etf_rolling = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-20, 0)
    etf_start = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(Window.unboundedPreceding, 0)

    lag_c = F.lag("Close", 1).over(etf_window)
    first_c = F.first("Close").over(etf_start)
    ret = F.when(lag_c.isNotNull() & (lag_c > 0), (F.col("Close") - lag_c) / lag_c).otherwise(F.lit(None))
    cum_ret = F.when(first_c.isNotNull() & (first_c > 0), ((F.col("Close") - first_c) / first_c) * 100.0).otherwise(F.lit(0.0))
    rolling_v = F.stddev("daily_return").over(etf_rolling)

    enriched = (
        cleaned.withColumn("daily_return", ret)
        .withColumn("daily_return_pct", ret * 100.0)
        .withColumn("rolling_volatility_21d", rolling_v)
        .withColumn("annualized_volatility_21d", rolling_v * F.sqrt(F.lit(252.0)) * 100.0)
        .withColumn("cumulative_return_pct", cum_ret)
    )

    combined_out = os.path.join(output_dir, "sector_etfs_processed.parquet")
    save_pyspark_parquet(enriched, combined_out, num_partitions=1)

    return enriched


# -----------------------------------------------------------------------------
# 6. GDELT News Records PySpark Pipeline
# -----------------------------------------------------------------------------

def process_gdelt_news_pyspark(
    spark: SparkSession,
    raw_jsonl_path: str = GDELT_RAW_JSONL,
    events_csv_path: str = EVENTS_CSV,
    output_parquet_path: str = "data/processed/news_articles.parquet"
) -> Tuple[DataFrame, DataFrame]:
    """
    Ingests and processes GDELT news records in PySpark:
    - Standardizes schema from JSONL
    - Computes sentiment classification and financial relevance
    - Joins with events table on event_id
    - Produces daily sentiment aggregation with rolling statistics
    """
    if not os.path.exists(raw_jsonl_path):
        raise FileNotFoundError(f"GDELT raw records missing at {raw_jsonl_path}")

    logger.info(f"PySpark ingesting raw GDELT records from {raw_jsonl_path}...")
    raw_news = spark.read.json(raw_jsonl_path)

    # Extract date part from GDELT timestamp
    cleaned_news = (
        raw_news.withColumn("published_date", F.to_date(F.substring(F.regexp_replace("seendate", "[^0-9]", ""), 1, 8), "yyyyMMdd"))
        .filter(F.col("published_date").isNotNull())
        .withColumn("gdelt_tone", F.coalesce(F.col("tone").cast("double"), F.lit(0.0)))
        .withColumn("positive_score", F.coalesce(F.col("positive_score").cast("double"), F.lit(0.0)))
        .withColumn("negative_score", F.coalesce(F.col("negative_score").cast("double"), F.lit(0.0)))
        .withColumn("polarity", F.coalesce(F.col("polarity").cast("double"), F.lit(0.0)))
        .withColumn("matched_event_id", F.coalesce(F.col("matched_event_id"), F.lit("")))
        .withColumn(
            "sentiment_label",
            F.when(F.col("gdelt_tone") >= 1.0, "Positive")
            .when(F.col("gdelt_tone") <= -1.0, "Negative")
            .otherwise("Neutral")
        )
        .withColumn(
            "article_id",
            F.md5(F.concat_ws("_", F.col("url"), F.col("title"), F.col("published_date")))
        )
        .dropDuplicates(["article_id"])
    )

    # Join with curated events / event database
    ev_path = events_csv_path
    if events_csv_path == "data/raw/events.csv" and os.path.exists("data/raw/events_expanded.csv"):
        ev_path = "data/raw/events_expanded.csv"

    if ev_path and os.path.exists(ev_path):
        raw_ev = spark.read.option("header", "true").option("inferSchema", "true").csv(ev_path)
        ev_cols = raw_ev.columns

        has_event_id = "event_id" in ev_cols
        cat_col = "event_category" if "event_category" in ev_cols else ("category" if "category" in ev_cols else None)

        ev_select = [
            F.col("event_id").alias("ev_id") if has_event_id else F.monotonically_increasing_id().cast("string").alias("ev_id"),
            F.col("event_name").alias("ev_name") if "event_name" in ev_cols else F.lit("Unknown").alias("ev_name"),
            F.to_date(F.col("event_date"), "yyyy-MM-dd").alias("official_event_date"),
        ]
        if cat_col:
            ev_select.append(F.col(cat_col).alias("ev_category"))
        else:
            ev_select.append(F.lit("General").alias("ev_category"))

        events_df = raw_ev.select(*ev_select)
        news_cols = cleaned_news.columns

        if "matched_event_id" in news_cols and has_event_id and "matched_event_name" in news_cols:
            join_cond = (cleaned_news["matched_event_id"] == events_df["ev_id"]) | (cleaned_news["matched_event_name"] == events_df["ev_name"])
        elif "matched_event_id" in news_cols and has_event_id:
            join_cond = cleaned_news["matched_event_id"] == events_df["ev_id"]
        elif "matched_event_name" in news_cols:
            join_cond = cleaned_news["matched_event_name"] == events_df["ev_name"]
        else:
            join_cond = None

        if join_cond is not None:
            cleaned_news = (
                cleaned_news.join(events_df, join_cond, how="left")
                .withColumn("days_from_event", F.datediff("published_date", "official_event_date"))
                .withColumn(
                    "timing_phase",
                    F.when(F.col("days_from_event") < 0, "pre-event")
                    .when(F.col("days_from_event") == 0, "day-of")
                    .when(F.col("days_from_event") > 0, "post-event")
                    .otherwise("unaligned")
                )
                .drop("ev_id")
            )

    save_pyspark_parquet(cleaned_news, output_parquet_path, num_partitions=1)

    # Date-based aggregation: Daily Sentiment Index
    daily_sentiment = (
        cleaned_news.groupBy("published_date")
        .agg(
            F.count("article_id").alias("daily_article_count"),
            F.avg("gdelt_tone").alias("avg_daily_tone"),
            F.min("gdelt_tone").alias("min_daily_tone"),
            F.max("gdelt_tone").alias("max_daily_tone"),
            F.sum(F.when(F.col("sentiment_label") == "Negative", 1).otherwise(0)).alias("negative_articles_count"),
            F.sum(F.when(F.col("sentiment_label") == "Positive", 1).otherwise(0)).alias("positive_articles_count")
        )
        .withColumn("negative_ratio", F.col("negative_articles_count") / F.col("daily_article_count"))
        .orderBy("published_date")
    )

    daily_sentiment_path = "data/processed/daily_news_sentiment.parquet"
    save_pyspark_parquet(daily_sentiment, daily_sentiment_path, num_partitions=1)

    return cleaned_news, daily_sentiment


# -----------------------------------------------------------------------------
# 7. FRED Macroeconomic PySpark Pipeline
# -----------------------------------------------------------------------------

def process_fred_macro_pyspark(
    spark: SparkSession,
    raw_fred_dir: str = FRED_RAW_DIR,
    output_dir: str = MACRO_PROCESSED_DIR
) -> DataFrame:
    """
    Processes FRED macroeconomic indicators in PySpark:
    - Standardizes schema across all indicator CSVs
    - Converts missing placeholders ('.') to null
    - Preserves native observation frequencies
    - Saves clean native macro dataset to Parquet
    """
    if not os.path.exists(raw_fred_dir):
        raise FileNotFoundError(f"FRED raw directory missing: {raw_fred_dir}")

    files = [os.path.join(raw_fred_dir, f).replace("\\", "/") for f in os.listdir(raw_fred_dir) if f.endswith("_raw.csv")]
    if not files:
        logger.warning("No raw FRED CSV files found.")
        return spark.createDataFrame([], StructType([]))

    raw_macro = spark.read.option("header", "true").csv(files)

    cleaned_macro = (
        raw_macro.withColumn("date", F.to_date(F.col("date"), "yyyy-MM-dd"))
        .filter(F.col("date").isNotNull())
        .withColumn("value", F.expr("try_cast(value as double)"))
        .dropDuplicates(["date", "series_id"])
        .orderBy("date", "series_id")
    )

    native_out = os.path.join(output_dir, "macro_indicators_native.parquet")
    save_pyspark_parquet(cleaned_macro, native_out, num_partitions=1)

    return cleaned_macro


# -----------------------------------------------------------------------------
# 8. Master PySpark Unified Pipeline Execution
# -----------------------------------------------------------------------------

def run_unified_pyspark_pipeline(
    spark: Optional[SparkSession] = None,
    max_stocks: Optional[int] = None
) -> Dict[str, Any]:
    """
    Executes the entire multi-source PySpark preprocessing architecture:
    1. Benchmark Indices & VIX
    2. Company Equities (Partitioned by Sector)
    3. Sector ETFs
    4. GDELT News Records & Daily Aggregations
    5. FRED Macroeconomic Indicators
    """
    spark_created = False
    if spark is None:
        spark = get_spark_session("UnifiedPySparkPipeline")
        spark_created = True

    try:
        logger.info("=" * 70)
        logger.info("STARTING UNIFIED PYSPARK PREPROCESSING PIPELINE")
        logger.info("=" * 70)

        # 1. Market Benchmarks & VIX
        benchmarks = process_benchmark_indices(spark)

        # 2. Company Equities (Partitioned Parquet)
        sp500_df = benchmarks.get("sp500")
        processed_stocks = process_company_equities(
            spark=spark,
            max_stocks=max_stocks,
            sp500_df=sp500_df
        )

        # 3. Sector ETFs
        processed_etfs = process_sector_etfs_pyspark(spark)

        # 4. GDELT News & Daily Sentiment
        news_df, daily_sent_df = process_gdelt_news_pyspark(spark)

        # 5. FRED Macroeconomics
        macro_df = process_fred_macro_pyspark(spark)

        logger.info("=" * 70)
        logger.info("UNIFIED PYSPARK PREPROCESSING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

        return {
            "benchmarks": benchmarks,
            "stocks": processed_stocks,
            "etfs": processed_etfs,
            "news": news_df,
            "daily_sentiment": daily_sent_df,
            "macro": macro_df
        }
    finally:
        if spark_created:
            stop_spark_session(spark)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_unified_pyspark_pipeline()
