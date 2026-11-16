
"""
Main Execution Script - Global Event & US Stock Market Analysis (Phase 1)

Orchestrates the entire Phase 1 workflow:
1. Market Data Ingestion (four equity indices + VIX)
2. Curated Event Dataset Creation
3. PySpark Preprocessing Pipeline
4. Multi-Index Event-Study Quantitative Calculations
"""

import sys
import logging

from src.ingestion.market_data import ingest_all_raw_data
from src.ingestion.event_data import save_curated_events
from src.preprocessing.spark_session import (
    get_spark_session,
    stop_spark_session
)
from src.preprocessing.market_pipeline import run_preprocessing_pipeline
from src.analytics.event_study import run_event_study_analysis


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

logger = logging.getLogger("Phase1-Pipeline")


# All datasets required for the pipeline
REQUIRED_TICKERS = ["^GSPC", "^IXIC", "^DJI", "^RUT", "^VIX"]

# Raw CSV paths
RAW_PATHS = {
    "^GSPC": "data/raw/sp500_raw.csv",
    "^IXIC": "data/raw/nasdaq_raw.csv",
    "^DJI": "data/raw/dow_jones_raw.csv",
    "^RUT": "data/raw/russell2000_raw.csv",
    "^VIX": "data/raw/vix_raw.csv"
}

# Processed Parquet paths
PROCESSED_PATHS = {
    "sp500": "data/processed/sp500_processed.parquet",
    "nasdaq": "data/processed/nasdaq_processed.parquet",
    "dow_jones": "data/processed/dow_jones_processed.parquet",
    "russell2000": "data/processed/russell2000_processed.parquet",
    "vix": "data/processed/vix_processed.parquet"
}


def run_pipeline(force_download: bool = False):

    logger.info("=" * 70)
    logger.info("Starting Phase 1: Global Event & US Stock Market Analysis")
    logger.info("=" * 70)

    # ---------------------------------------------------------
    # Step 1: Market Data Ingestion
    # ---------------------------------------------------------

    logger.info("[Step 1/4] Ingesting market datasets...")

    market_files = ingest_all_raw_data(
        force_download=force_download
    )

    for ticker, path in market_files.items():
        logger.info(" - %s: %s", ticker, path)

    # Validate ingestion before proceeding
    missing_tickers = [
        ticker for ticker in REQUIRED_TICKERS
        if ticker not in market_files
    ]

    if missing_tickers:
        raise RuntimeError(
            "Market data ingestion incomplete. "
            f"Missing datasets: {missing_tickers}. "
            "Check the ingestion logs and retry."
        )

    logger.info(
        "Successfully ingested all %d required datasets.",
        len(REQUIRED_TICKERS)
    )

    # ---------------------------------------------------------
    # Step 2: Curated Events
    # ---------------------------------------------------------

    logger.info("[Step 2/4] Initializing curated global events calendar...")

    events_path = save_curated_events(
        "data/raw/events.csv",
        overwrite=False
    )

    logger.info(" - Events stored at: %s", events_path)

    # ---------------------------------------------------------
    # Step 3: PySpark Preprocessing
    # ---------------------------------------------------------

    logger.info("[Step 3/4] Launching PySpark preprocessing pipeline...")

    spark = get_spark_session("Phase1_Data_Processing")

    try:
        processed_dfs = run_preprocessing_pipeline(
            spark=spark,

            # S&P 500
            sp500_raw_path=market_files["^GSPC"],
            sp500_out_path=PROCESSED_PATHS["sp500"],

            # NASDAQ
            nasdaq_raw_path=market_files["^IXIC"],
            nasdaq_out_path=PROCESSED_PATHS["nasdaq"],

            # Dow Jones
            dow_jones_raw_path=market_files["^DJI"],
            dow_jones_out_path=PROCESSED_PATHS["dow_jones"],

            # Russell 2000
            russell2000_raw_path=market_files["^RUT"],
            russell2000_out_path=PROCESSED_PATHS["russell2000"],

            # VIX
            vix_raw_path=market_files["^VIX"],
            vix_out_path=PROCESSED_PATHS["vix"]
        )

        for dataset_name, dataframe in processed_dfs.items():
            logger.info(
                " - %s Processed Records: %d",
                dataset_name.upper(),
                dataframe.count()
            )

    finally:
        stop_spark_session(spark)

    # ---------------------------------------------------------
    # Step 4: Multi-Index Event Study
    # ---------------------------------------------------------

    logger.info("[Step 4/4] Executing multi-index event-study analytics...")

    index_parquet_paths = {
        "sp500": PROCESSED_PATHS["sp500"],
        "nasdaq": PROCESSED_PATHS["nasdaq"],
        "dow_jones": PROCESSED_PATHS["dow_jones"],
        "russell2000": PROCESSED_PATHS["russell2000"]
    }

    study_results = run_event_study_analysis(
        events_csv_path=events_path,
        sp500_parquet_path=PROCESSED_PATHS["sp500"],
        vix_parquet_path=PROCESSED_PATHS["vix"],
        output_csv_path="data/processed/event_study_results.csv",
        output_parquet_path="data/processed/event_study_results.parquet",
        index_parquet_paths=index_parquet_paths
    )

    logger.info(
        " - Generated %d event-index records.",
        len(study_results)
    )

    if not study_results.empty:
        logger.info(
            " - Indices represented: %s",
            ", ".join(sorted(study_results["index"].unique()))
        )

        logger.info(
            " - Events represented: %d",
            study_results["event_name"].nunique()
        )

    logger.info("=" * 70)
    logger.info("Phase 1 Pipeline Execution Completed Successfully!")
    logger.info("To launch the Streamlit dashboard, run:")
    logger.info("    python -m streamlit run dashboard/app.py")
    logger.info("=" * 70)


if __name__ == "__main__":
    force_dl = "--force-download" in sys.argv
    run_pipeline(force_download=force_dl)
