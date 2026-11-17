"""
Data Ingestion Package
Provides modules to ingest market benchmarks, company-level equities, and historical event datasets.
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
