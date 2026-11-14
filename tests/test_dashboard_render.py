"""
Tests for Streamlit Dashboard Chart Rendering using AppTest
"""

import os
import pytest
from streamlit.testing.v1 import AppTest


def test_dashboard_renders_successfully():
    """Verify Streamlit dashboard executes all logic and generates charts without exceptions."""
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dashboard", "app.py"))
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()

    # Verify no exceptions or errors occurred during app execution
    assert not at.exception, f"Dashboard raised unexpected exception: {at.exception}"

    # Verify title and subheaders are rendered
    assert len(at.title) >= 1
    assert "Global Events" in at.title[0].value
    assert len(at.subheader) >= 5

    # Verify metrics and widgets
    assert len(at.metric) >= 4
    assert len(at.dataframe) >= 1
    assert len(at.selectbox) >= 1
