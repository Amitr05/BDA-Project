"""
PySpark Preprocessing Pipeline Module
Performs data cleaning, date standardization, daily return computation,
volume change calculation, and rolling volatility estimation using PySpark SQL.
"""

import os
import logging
from typing import Optional
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.preprocessing.spark_session import get_spark_session

logger = logging.getLogger(__name__)


def load_raw_market_csv(spark: SparkSession, csv_path: str) -> DataFrame:
    """
    Load raw stock market CSV into PySpark DataFrame with header and inferred schema.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw CSV file does not exist at {csv_path}")

    logger.info(f"Loading raw market CSV from {csv_path}...")
    df = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv(csv_path)
    )
    return df


def clean_and_sort_market_data(df: DataFrame) -> DataFrame:
    """
    Clean raw market DataFrame:
    - Casts 'Date' to DateType
    - Drops duplicates by Date
    - Drops records where Date or Close is null
    - Casts numeric price and volume columns to Double / Long
    - Orders chronologically by Date
    """
    cleaned_df = (
        df.withColumn("Date", F.to_date(F.col("Date"), "yyyy-MM-dd"))
        .filter(F.col("Date").isNotNull() & F.col("Close").isNotNull())
        .dropDuplicates(["Date"])
        .withColumn("Open", F.col("Open").cast("double"))
        .withColumn("High", F.col("High").cast("double"))
        .withColumn("Low", F.col("Low").cast("double"))
        .withColumn("Close", F.col("Close").cast("double"))
        .withColumn("Adj Close", F.when(F.col("Adj Close").isNotNull(), F.col("Adj Close").cast("double")).otherwise(F.col("Close").cast("double")))
        .withColumn("Volume", F.coalesce(F.col("Volume").cast("double"), F.lit(0.0)))
        .orderBy("Date")
    )
    return cleaned_df


def calculate_market_metrics(df: DataFrame, volatility_window: int = 21) -> DataFrame:
    """
    Compute daily financial metrics over chronological trading days:
    - Daily Return: (Close_t - Close_{t-1}) / Close_{t-1}
    - Daily Return Percentage: Daily Return * 100
    - Daily Volume Change Percentage: (Volume_t - Volume_{t-1}) / Volume_{t-1} * 100
    - Rolling Volatility (21 trading days): StdDev(Daily Return) over 21-day window
    - Annualized Volatility: Rolling Volatility * sqrt(252) * 100
    """
    time_window = Window.orderBy("Date")
    rolling_window = Window.orderBy("Date").rowsBetween(-(volatility_window - 1), 0)

    lag_close = F.lag("Close", 1).over(time_window)
    lag_volume = F.lag("Volume", 1).over(time_window)

    # Daily Return
    daily_return = F.when(lag_close.isNotNull() & (lag_close > 0), (F.col("Close") - lag_close) / lag_close).otherwise(F.lit(None))
    daily_return_pct = daily_return * F.lit(100.0)

    # Volume change percentage (safeguard against 0 volume)
    volume_change_pct = F.when(
        lag_volume.isNotNull() & (lag_volume > 0),
        ((F.col("Volume") - lag_volume) / lag_volume) * F.lit(100.0)
    ).otherwise(F.lit(None))

    metrics_df = (
        df.withColumn("prev_close", lag_close)
        .withColumn("daily_return", daily_return)
        .withColumn("daily_return_pct", daily_return_pct)
        .withColumn("volume_change_pct", volume_change_pct)
    )

    # Rolling Volatility of daily returns
    rolling_vol = F.stddev("daily_return").over(rolling_window)
    annualized_vol = rolling_vol * F.sqrt(F.lit(252.0)) * F.lit(100.0)

    enriched_df = (
        metrics_df.withColumn("rolling_volatility_21d", rolling_vol)
        .withColumn("annualized_volatility_21d", annualized_vol)
        .drop("prev_close")
    )

    return enriched_df


def save_dataframe_parquet(df: DataFrame, output_path: str) -> str:
    """
    Save PySpark DataFrame to Parquet.
    Handles Windows Hadoop winutils compatibility gracefully:
    Tries native Spark parquet write first; falls back to PyArrow parquet write if
    Hadoop native winutils is absent in the Windows environment.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        # Coalesce to 1 partition for compact, clean storage of daily series
        df.coalesce(1).write.mode("overwrite").parquet(output_path)
        logger.info(f"Successfully saved DataFrame via PySpark Parquet to {output_path}")
    except Exception as exc:
        logger.warning(
            f"Native Spark parquet write failed ({exc}). Falling back to PyArrow parquet engine for Windows compatibility."
        )
        import shutil
        if os.path.exists(output_path):
            if os.path.isdir(output_path):
                shutil.rmtree(output_path)
            else:
                os.remove(output_path)

        pdf = df.toPandas()
        pdf.to_parquet(output_path, engine="pyarrow", index=False)
        logger.info(f"Saved DataFrame via PyArrow Parquet fallback to {output_path}")

    return output_path


def process_market_dataset(
    spark: SparkSession,
    raw_csv_path: str,
    output_parquet_path: str,
    volatility_window: int = 21
) -> DataFrame:
    """
    Full preprocessing execution pipeline for a market dataset.
    """
    raw_df = load_raw_market_csv(spark, raw_csv_path)
    cleaned_df = clean_and_sort_market_data(raw_df)
    processed_df = calculate_market_metrics(cleaned_df, volatility_window=volatility_window)
    save_dataframe_parquet(processed_df, output_parquet_path)
    return processed_df


def run_preprocessing_pipeline(
    spark: Optional[SparkSession] = None,
    sp500_raw_path: str = "data/raw/sp500_raw.csv",
    vix_raw_path: str = "data/raw/vix_raw.csv",
    sp500_out_path: str = "data/processed/sp500_processed.parquet",
    vix_out_path: str = "data/processed/vix_processed.parquet"
) -> dict:
    """
    Execute preprocessing pipeline for S&P 500 and VIX.
    """
    spark_created = False
    if spark is None:
        spark = get_spark_session("MarketPreprocessing")
        spark_created = True

    try:
        logger.info("Executing S&P 500 preprocessing pipeline...")
        sp500_df = process_market_dataset(spark, sp500_raw_path, sp500_out_path)

        logger.info("Executing VIX preprocessing pipeline...")
        vix_df = process_market_dataset(spark, vix_raw_path, vix_out_path)

        return {
            "sp500": sp500_df,
            "vix": vix_df
        }
    finally:
        if spark_created:
            # We keep session or let caller manage; in standalone run, can keep active
            pass
