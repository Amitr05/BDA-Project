"""
Data Ingestion Package
Provides modules to ingest market benchmarks, company-level equities, historical event datasets,
sector ETFs, and FRED macroeconomic indicators.
"""

from src.ingestion.market_data import ingest_all_raw_data, ingest_market_data
from src.ingestion.event_data import get_curated_events, save_curated_events, load_event_dataset
from src.ingestion.company_data import (
    load_company_registry,
    load_index_constituents,
    get_active_constituents,
    ingest_company_stocks,
)
from src.ingestion.event_database import (
    build_event_dataframe,
    validate_event_database,
    save_event_database,
    load_event_database,
    filter_events,
    get_related_companies,
    build_legacy_events_df,
    VALID_CATEGORIES,
    REQUIRED_COLUMNS,
)
from src.ingestion.gdelt_client import (
    GDELTClient,
    ingest_event_aligned_news,
    HISTORICAL_BENCHMARK_NEWS
)
from src.ingestion.etf_data import (
    SECTOR_ETFS,
    ETF_METADATA,
    get_sector_etfs_metadata,
    is_valid_etf_csv,
    fetch_single_etf,
    fetch_etfs_batch,
    save_etf_data_incremental,
    ingest_sector_etfs
)
from src.ingestion.fred_data import (
    FRED_SERIES_CATALOG,
    get_fred_api_key,
    get_fred_catalog,
    fetch_fred_series,
    save_fred_series_incremental,
    ingest_all_fred_indicators
)
