"""
Tests for Dashboard Components and Data Contract
"""

import os
import pandas as pd
import pytest
from dashboard.app import load_market_data, load_event_study


def test_dashboard_data_loading_missing():
    """Verify dashboard gracefully handles missing data files."""
    # When files don't exist in a hypothetical empty state
    # Functions should return None, None rather than raising unhandled exceptions
    sp500, vix = load_market_data()
    # If the pipeline has not run yet, both are None
    if not os.path.exists("data/processed/sp500_processed.parquet"):
        assert sp500 is None
        assert vix is None


def test_dashboard_data_loading_when_present():
    """Verify that if processed files exist, dashboard loads them with datetime index."""
    if os.path.exists("data/processed/sp500_processed.parquet") and os.path.exists("data/processed/vix_processed.parquet"):
        sp500, vix = load_market_data()
        assert sp500 is not None
        assert vix is not None
        assert pd.api.types.is_datetime64_any_dtype(sp500["Date"])
        assert pd.api.types.is_datetime64_any_dtype(vix["Date"])
