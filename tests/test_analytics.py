"""
Tests for Event Study Analytics Module
"""

import os
import tempfile
import pandas as pd
import numpy as np
import pytest

from src.analytics.event_study import (
    find_effective_trading_day,
    calculate_event_metrics,
    run_event_study_analysis,
)


def test_find_effective_trading_day():
    """Verify mapping of calendar dates (including weekends and holidays) to trading days."""
    trading_dates = ["2023-10-05", "2023-10-06", "2023-10-09", "2023-10-10"]  # Oct 7-8 was weekend

    # Exact weekday match (Friday Oct 6)
    idx_friday = find_effective_trading_day("2023-10-06", trading_dates)
    assert idx_friday == 1
    assert trading_dates[idx_friday] == "2023-10-06"

    # Weekend match (Saturday Oct 7) should map to next trading day (Monday Oct 9)
    idx_saturday = find_effective_trading_day("2023-10-07", trading_dates)
    assert idx_saturday == 2
    assert trading_dates[idx_saturday] == "2023-10-09"

    # Sunday Oct 8 should also map to Monday Oct 9
    idx_sunday = find_effective_trading_day("2023-10-08", trading_dates)
    assert idx_sunday == 2

    # Beyond dataset range
    idx_future = find_effective_trading_day("2025-01-01", trading_dates)
    assert idx_future is None


def test_calculate_event_metrics_windows():
    """Test calculation of Pre-5D, Post-5D, Post-30D returns and VIX metrics."""
    # Create 40 consecutive mock trading days
    date_range = pd.date_range("2023-01-01", periods=45, freq="B").strftime("%Y-%m-%d").tolist()
    # T0 will be at index 10: T-5 is index 5, T+5 is index 15, T+30 is index 40
    # Prices:
    # index 5 (T-5): 100.0
    # index 10 (T0): 110.0  -> Pre-5D return = (110 - 100) / 100 = +10.0%
    # index 15 (T+5): 104.5 -> Post-5D return = (104.5 - 110) / 110 = -5.0%
    # index 40 (T+30): 121.0 -> Post-30D return = (121 - 110) / 110 = +10.0%
    prices = [100.0] * len(date_range)
    prices[5] = 100.0
    prices[10] = 110.0
    prices[15] = 104.5
    prices[40] = 121.0

    sp500_df = pd.DataFrame({"Date": date_range, "Close": prices})

    # VIX mock data
    vix_values = [15.0] * len(date_range)
    vix_values[10] = 20.0  # VIX at event
    vix_values[12] = 35.0  # Max VIX in 30d window
    vix_values[15] = 22.0  # VIX at T+5
    vix_df = pd.DataFrame({"Date": date_range, "Close": vix_values})

    event = {
        "event_name": "Mock Geopolitical Shock",
        "event_date": date_range[10],
        "event_category": "Geopolitical",
        "short_description": "Mock event for testing."
    }

    metrics = calculate_event_metrics(event, sp500_df, vix_df)

    assert metrics["effective_trading_date"] == date_range[10]
    assert pytest.approx(metrics["sp500_pre_5d_return_pct"], rel=1e-2) == 10.0
    assert pytest.approx(metrics["sp500_post_5d_return_pct"], rel=1e-2) == -5.0
    assert pytest.approx(metrics["sp500_post_30d_return_pct"], rel=1e-2) == 10.0
    assert metrics["vix_at_event"] == 20.0
    assert metrics["vix_post_5d"] == 22.0
    assert metrics["vix_max_30d"] == 35.0
    assert metrics["vix_5d_change"] == 2.0


def test_full_event_study_pipeline():
    """Verify end-to-end execution of event study exporter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dates = pd.date_range("2020-01-01", periods=60, freq="B").strftime("%Y-%m-%d").tolist()
        sp500_df = pd.DataFrame({"Date": dates, "Close": np.linspace(3000, 3500, len(dates))})
        vix_df = pd.DataFrame({"Date": dates, "Close": np.linspace(15, 25, len(dates))})

        sp_path = os.path.join(tmpdir, "sp.parquet")
        vix_path = os.path.join(tmpdir, "vix.parquet")
        sp500_df.to_parquet(sp_path, index=False)
        vix_df.to_parquet(vix_path, index=False)

        events_csv = os.path.join(tmpdir, "events.csv")
        ev_df = pd.DataFrame([{
            "event_name": "Test Event",
            "event_date": dates[10],
            "event_category": "Test",
            "short_description": "Pipeline test"
        }])
        ev_df.to_csv(events_csv, index=False)

        out_csv = os.path.join(tmpdir, "out.csv")
        out_parquet = os.path.join(tmpdir, "out.parquet")

        results = run_event_study_analysis(
            events_csv_path=events_csv,
            sp500_parquet_path=sp_path,
            vix_parquet_path=vix_path,
            output_csv_path=out_csv,
            output_parquet_path=out_parquet
        )

        assert len(results) == 1
        assert os.path.exists(out_csv)
        assert os.path.exists(out_parquet)
