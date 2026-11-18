"""
Main Execution Script - Global Event, Multi-Asset & Macroeconomic Analytics Platform
===================================================================================
Orchestrates the end-to-end data processing and financial analytics architecture
across three logical storage layers (Raw -> Processed -> Analytics):

Layer 1: Raw Data Collection & Ingestion
  1. US Benchmark Equity Indices (S&P 500, NASDAQ, Dow Jones, Russell 2000) & VIX
  2. Curated Global Events Calendar & Comprehensive Historical Event Database
  3. 8 Select Sector SPDR ETFs (XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU)
  4. FRED Point-in-Time Macroeconomic Indicators (CPI, Fed Funds, Yield Curve, etc.)
  5. 300+ Company Equities and GDELT Global News Records

Layer 2: PySpark Preprocessing & Multi-Source Integration Engine
  1. Schema standardisation and data cleaning across all 7 sources
  2. Complex multi-dataset joins (equities-to-sectors, news-to-events, macro-to-trading calendar)
  3. Distributed window functions (lagged returns, rolling 21D volatility, cumulative returns)
  4. Date-based aggregations (daily news sentiment ratios, macro publication dates)
  5. Sector-partitioned Parquet persistence (data/processed/stocks/sector=.../)

Layer 3: Advanced Financial Analytics & Econometric Engine
  1. Multi-Index & Sector Event Studies
  2. Company-Level Event Studies with Market Model Abnormal Returns (AR & CAR)
  3. Parametric Cross-Sectional Statistical Significance Testing (t-stats, p-values, DoF)
  4. Explicit econometric assumptions and limitations reporting (no correlation-as-causation)
  5. Cross-Company intra-sector dispersion and resilience rankings
  6. Macroeconomic regime conditioning (yield curve inversions and inflation regimes)
  7. Benchmark & Mathematical Verification (PySpark vs independent NumPy precision < 1e-5)
"""

import sys
import os
import logging
from typing import Optional

from src.ingestion.market_data import ingest_all_raw_data
from src.ingestion.event_data import save_curated_events
from src.ingestion.etf_data import ingest_sector_etfs
from src.ingestion.fred_data import ingest_all_fred_indicators
from src.preprocessing.spark_session import (
    get_spark_session,
    stop_spark_session
)
from src.preprocessing.market_pipeline import run_preprocessing_pipeline
from src.preprocessing.etf_pipeline import run_etf_preprocessing_pipeline
from src.preprocessing.macro_pipeline import run_macro_pipeline
from src.preprocessing.unified_pipeline import run_unified_pyspark_pipeline
from src.analytics.event_study import run_event_study_analysis
from src.analytics.sector_analysis import (
    load_processed_etf_frames,
    align_sector_etfs_with_market_and_vix,
    run_sector_event_study,
    calculate_sector_performance_summary
)
from src.analytics.advanced_event_study import run_master_analytics_pipeline
from src.analytics.benchmark_validation import run_benchmark_and_verification


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

logger = logging.getLogger("BDA-Pipeline")

REQUIRED_TICKERS = ["^GSPC", "^IXIC", "^DJI", "^RUT", "^VIX"]

PROCESSED_PATHS = {
    "sp500": "data/processed/sp500_processed.parquet",
    "nasdaq": "data/processed/nasdaq_processed.parquet",
    "dow_jones": "data/processed/dow_jones_processed.parquet",
    "russell2000": "data/processed/russell2000_processed.parquet",
    "vix": "data/processed/vix_processed.parquet"
}


def run_pipeline(
    force_download: bool = False,
    include_etfs: bool = True,
    include_macro: bool = True,
    include_pyspark_master: bool = True,
    include_analytics_master: bool = True
):
    logger.info("=" * 80)
    logger.info("STARTING BDA PLATFORM: MULTI-ASSET, PYSPARK & ECONOMETRIC PIPELINE")
    logger.info("=" * 80)

    # =========================================================================
    # LAYER 1: RAW INGESTION
    # =========================================================================
    logger.info("\n>>> LAYER 1: RAW DATA INGESTION <<<")

    logger.info("[1.1] Ingesting US benchmark indices and VIX...")
    market_files = ingest_all_raw_data(force_download=force_download)
    for ticker, path in market_files.items():
        logger.info(" - %s -> %s", ticker, path)

    missing = [t for t in REQUIRED_TICKERS if t not in market_files]
    if missing:
        raise RuntimeError(f"Missing required benchmark market datasets: {missing}")

    logger.info("[1.2] Initializing curated event calendar and database...")
    events_path = save_curated_events("data/raw/events.csv", overwrite=False)
    logger.info(" - Events registered at %s", events_path)

    if include_etfs:
        logger.info("[1.3] Ingesting 8 Sector ETFs (XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU)...")
        etf_summary = ingest_sector_etfs(force_download=force_download)
        logger.info(" - Sector ETF ingestion: %d/%d successful.", etf_summary["successful_count"], etf_summary["target_count"])

    if include_macro:
        logger.info("[1.4] Ingesting FRED Macroeconomic Indicators (CPI, Rates, Yields, GDP, Sentiment)...")
        fred_summary = ingest_all_fred_indicators(force_download=force_download)
        logger.info(" - FRED indicators: %d/%d series active.", fred_summary["successful_series"], fred_summary["total_series"])

    # =========================================================================
    # LAYER 2: PYSPARK PREPROCESSING & DATA STORAGE
    # =========================================================================
    logger.info("\n>>> LAYER 2: PYSPARK PREPROCESSING & STORAGE <<<")

    if include_pyspark_master:
        logger.info("[2.1] Executing Unified PySpark Preprocessing Pipeline...")
        spark_results = run_unified_pyspark_pipeline()
        logger.info(" - PySpark unified pipeline completed across benchmarks, equities, ETFs, news, and macro.")

    if include_macro:
        logger.info("[2.2] Aligning macroeconomic series with trading calendar (Zero Look-Ahead Bias)...")
        macro_dfs = run_macro_pipeline()
        logger.info(" - Aligned %d trading days of point-in-time macro indicators.", len(macro_dfs.get("aligned", [])))

    if include_etfs:
        logger.info("[2.3] Processing Sector ETF pipeline and market alignment...")
        etf_frames = load_processed_etf_frames()
        if etf_frames:
            align_sector_etfs_with_market_and_vix(
                etf_frames=etf_frames,
                market_parquet_paths=PROCESSED_PATHS,
                vix_parquet_path=PROCESSED_PATHS["vix"]
            )
            run_sector_event_study(
                events_csv_path=events_path,
                etf_frames=etf_frames,
                vix_parquet_path=PROCESSED_PATHS["vix"]
            )
            perf_summary = calculate_sector_performance_summary(etf_frames)
            logger.info(" - Sector ETF summary generated (%d sectors).", len(perf_summary))

    # Benchmark equity indices event study
    logger.info("[2.4] Running equity indices event study baseline...")
    run_event_study_analysis(
        events_csv_path=events_path,
        sp500_parquet_path=PROCESSED_PATHS["sp500"],
        vix_parquet_path=PROCESSED_PATHS["vix"],
        output_csv_path="data/processed/event_study_results.csv",
        output_parquet_path="data/processed/event_study_results.parquet",
        index_parquet_paths=PROCESSED_PATHS
    )

    # =========================================================================
    # LAYER 3: ADVANCED FINANCIAL ANALYTICS & ECONOMETRICS
    # =========================================================================
    logger.info("\n>>> LAYER 3: ADVANCED FINANCIAL ANALYTICS (data/analytics/) <<<")

    if include_analytics_master:
        logger.info("[3.1] Executing Master Financial Analytics & Econometric Engine...")
        analytics_outputs = run_master_analytics_pipeline(output_dir="data/analytics")
        logger.info(" - Successfully persisted 8 analytical products to Layer 3 (data/analytics/).")

        logger.info("[3.2] Executing Benchmark & Mathematical Precision Verification...")
        bench_res = run_benchmark_and_verification()
        logger.info(
            " - Benchmark Precision Verified: %s (Spark time: %ss, Max return diff: %.2e, Max vol diff: %.2e)",
            bench_res["numerical_precision_verified"],
            bench_res["spark_execution_time_seconds"],
            bench_res["max_return_difference"],
            bench_res["max_volatility_difference"]
        )

    logger.info("=" * 80)
    logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY ACROSS ALL THREE LAYERS!")
    logger.info("To explore the interactive dashboard:")
    logger.info("    python -m streamlit run dashboard/app.py")
    logger.info("=" * 80)


if __name__ == "__main__":
    force_dl = "--force-download" in sys.argv
    skip_etfs = "--skip-etfs" in sys.argv
    skip_macro = "--skip-macro" in sys.argv
    skip_spark = "--skip-pyspark" in sys.argv
    skip_analytics = "--skip-analytics" in sys.argv

    run_pipeline(
        force_download=force_dl,
        include_etfs=not skip_etfs,
        include_macro=not skip_macro,
        include_pyspark_master=not skip_spark,
        include_analytics_master=not skip_analytics
    )
