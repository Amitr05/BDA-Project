"""
Integration & Rendering Tests for All 8 Multipage Dashboard Pages
==================================================================
Verifies that each dedicated page module executes cleanly under AppTest
without unhandled exceptions, renders required UI components, and binds data.
"""

import os
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

DASHBOARD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dashboard"))
PAGES_DIR = os.path.join(DASHBOARD_DIR, "pages")


def test_home_page_renders():
    """Verify Home / Overview page executes cleanly."""
    page_path = os.path.join(PAGES_DIR, "Home.py")
    at = AppTest.from_file(page_path, default_timeout=30)
    at.run()
    assert not at.exception, f"Home page error: {at.exception}"
    assert len(at.title) >= 1
    assert len(at.subheader) >= 3
    assert len(at.metric) >= 4
    assert len(at.dataframe) >= 1


def test_market_explorer_page_renders():
    """Verify Market Explorer page executes cleanly."""
    page_path = os.path.join(PAGES_DIR, "Market_Explorer.py")
    at = AppTest.from_file(page_path, default_timeout=30)
    at.run()
    assert not at.exception, f"Market Explorer error: {at.exception}"
    assert len(at.subheader) >= 2
    assert len(at.selectbox) >= 2
    assert len(at.metric) >= 4


def test_global_event_explorer_page_renders():
    """Verify Global Event Explorer page executes cleanly."""
    page_path = os.path.join(PAGES_DIR, "Global_Event_Explorer.py")
    at = AppTest.from_file(page_path, default_timeout=30)
    at.run()
    assert not at.exception, f"Global Event Explorer error: {at.exception}"
    assert len(at.subheader) >= 2
    assert len(at.metric) >= 4
    assert len(at.dataframe) >= 1
    assert len(at.selectbox) >= 2


def test_event_impact_analyzer_page_renders():
    """Verify Event Impact Analyzer page executes cleanly."""
    page_path = os.path.join(PAGES_DIR, "Event_Impact_Analyzer.py")
    at = AppTest.from_file(page_path, default_timeout=30)
    at.run()
    assert not at.exception, f"Event Impact Analyzer error: {at.exception}"
    assert len(at.subheader) >= 2
    assert len(at.metric) >= 4
    assert len(at.selectbox) >= 2


def test_sector_analysis_page_renders():
    """Verify Sector Analysis page executes cleanly."""
    page_path = os.path.join(PAGES_DIR, "Sector_Analysis.py")
    at = AppTest.from_file(page_path, default_timeout=30)
    at.run()
    assert not at.exception, f"Sector Analysis error: {at.exception}"
    assert len(at.subheader) >= 3
    assert len(at.dataframe) >= 1


def test_ml_predictions_page_renders():
    """Verify ML Predictions page executes and invokes prediction pipeline."""
    page_path = os.path.join(PAGES_DIR, "ML_Predictions.py")
    at = AppTest.from_file(page_path, default_timeout=30)
    at.run()
    assert not at.exception, f"ML Predictions error: {at.exception}"
    assert len(at.subheader) >= 3
    assert len(at.selectbox) >= 2
    assert len(at.dataframe) >= 1
    assert len(at.button) >= 1


def test_advanced_analytics_page_renders():
    """Verify Advanced Analytics page executes cleanly."""
    page_path = os.path.join(PAGES_DIR, "Advanced_Analytics.py")
    at = AppTest.from_file(page_path, default_timeout=30)
    at.run()
    assert not at.exception, f"Advanced Analytics error: {at.exception}"
    assert len(at.subheader) >= 4
    assert len(at.metric) >= 4
    assert len(at.dataframe) >= 2


def test_macro_and_news_page_renders():
    """Verify Macro & News page executes cleanly."""
    page_path = os.path.join(PAGES_DIR, "Macro_and_News.py")
    at = AppTest.from_file(page_path, default_timeout=30)
    at.run()
    assert not at.exception, f"Macro & News error: {at.exception}"
    assert len(at.subheader) >= 4
    assert len(at.metric) >= 4
    assert len(at.dataframe) >= 1
