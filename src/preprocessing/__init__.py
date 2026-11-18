"""
PySpark Preprocessing Package
Provides Spark session initialization and transformation pipelines for stock market benchmark datasets,
company equities, sector ETFs, GDELT news sentiment, and FRED macroeconomic indicator pipelines.
"""

from src.preprocessing.spark_session import get_spark_session, stop_spark_session
from src.preprocessing.market_pipeline import clean_and_sort_market_data, calculate_market_metrics, save_dataframe_parquet
from src.preprocessing.news_processor import process_gdelt_news
from src.preprocessing.etf_pipeline import (
    clean_and_sort_etf_dataframe,
    calculate_etf_financial_metrics,
    process_single_etf,
    run_etf_preprocessing_pipeline
)
from src.preprocessing.macro_pipeline import (
    calculate_publication_date,
    clean_fred_series_dataframe,
    build_native_frequency_macro_dataset,
    align_macro_with_trading_dates,
    align_macro_with_market_indices,
    run_macro_pipeline
)
from src.preprocessing.unified_pipeline import (
    save_pyspark_parquet,
    process_benchmark_indices,
    process_company_equities,
    process_sector_etfs_pyspark,
    process_gdelt_news_pyspark,
    process_fred_macro_pyspark,
    run_unified_pyspark_pipeline
)
