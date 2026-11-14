"""
Main Execution Script - Global Event & US Stock Market Analysis (Phase 1)

Orchestrates the entire Phase 1 workflow:
1. Market Data Ingestion (^GSPC, ^VIX)
2. Curated Event Dataset Creation
3. PySpark Preprocessing Pipeline (Returns, Volume Change, Rolling Volatility, Parquet Storage)
4. Event-Study Quantitative Calculations
"""

import sys
import os
import logging
from src.ingestion.market_data import ingest_all_raw_data
from src.ingestion.event_data import save_curated_events
from src.preprocessing.spark_session import get_spark_session, stop_spark_session
from src.preprocessing.market_pipeline import run_preprocessing_pipeline
from src.analytics.event_study import run_event_study_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("Phase1-Pipeline")


def run_pipeline(force_download: bool = False):
    logger.info("=================================================================")
    logger.info("Starting Phase 1: Global Event & US Stock Market Analysis")
    logger.info("=================================================================")

    # Step 1: Ingestion of Market Data
    logger.info("[Step 1/4] Ingesting Market Data via yfinance (^GSPC, ^VIX)...")
    market_files = ingest_all_raw_data(force_download=force_download)
    for ticker, path in market_files.items():
        logger.info(f" - {ticker}: {path}")

    # Step 2: Ingestion / Verification of Curated Events
    logger.info("[Step 2/4] Initializing Curated Global Events Calendar...")
    events_path = save_curated_events("data/raw/events.csv", overwrite=False)
    logger.info(f" - Events stored at: {events_path}")

    # Step 3: PySpark Preprocessing
    logger.info("[Step 3/4] Launching PySpark Preprocessing Pipeline...")
    spark = get_spark_session("Phase1_Data_Processing")
    try:
        processed_dfs = run_preprocessing_pipeline(
            spark=spark,
            sp500_raw_path=market_files.get("^GSPC", "data/raw/sp500_raw.csv"),
            vix_raw_path=market_files.get("^VIX", "data/raw/vix_raw.csv"),
            sp500_out_path="data/processed/sp500_processed.parquet",
            vix_out_path="data/processed/vix_processed.parquet"
        )
        logger.info(f" - S&P 500 Processed Records: {processed_dfs['sp500'].count()}")
        logger.info(f" - VIX Processed Records: {processed_dfs['vix'].count()}")
    finally:
        stop_spark_session(spark)

    # Step 4: Event-Study Analytics
    logger.info("[Step 4/4] Executing Event-Study Analytics...")
    study_results = run_event_study_analysis(
        events_csv_path="data/raw/events.csv",
        sp500_parquet_path="data/processed/sp500_processed.parquet",
        vix_parquet_path="data/processed/vix_processed.parquet",
        output_csv_path="data/processed/event_study_results.csv",
        output_parquet_path="data/processed/event_study_results.parquet"
    )
    logger.info(f" - Processed event study for {len(study_results)} events.")

    logger.info("=================================================================")
    logger.info("Phase 1 Base Prototype Execution Completed Successfully!")
    logger.info("To launch the Streamlit dashboard, run:")
    logger.info("    python -m streamlit run dashboard/app.py")
    logger.info("=================================================================")


if __name__ == "__main__":
    force_dl = "--force-download" in sys.argv
    run_pipeline(force_download=force_dl)
