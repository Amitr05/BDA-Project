"""
PySpark Preprocessing Package
Provides Spark session initialization and transformation pipelines for stock market benchmark datasets,
as well as news sentiment and event processing pipelines.
"""

from src.preprocessing.spark_session import get_spark_session
from src.preprocessing.market_pipeline import clean_and_sort_market_data, calculate_market_metrics, save_dataframe_parquet
from src.preprocessing.news_processor import process_gdelt_news
