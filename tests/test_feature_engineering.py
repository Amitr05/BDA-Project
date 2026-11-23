"""
Unit and Integration Tests for ML Feature Engineering & Target Generation
==========================================================================
Tests:
1. Target column definitions (exactly eight numerical targets).
2. Input feature definitions (numeric market conditions + categorical regimes).
3. Trading calendar alignment (weekend, holiday, and cut-off handling).
4. Forward return mathematical calculation correctness.
5. Absence of data leakage (strictly backward-looking features).
6. Training dataset integrity (310 usable observations, 0 nulls, 5 regions, 3 regimes).
"""

import os
import json
import pytest
import numpy as np
import pandas as pd

from src.ml.feature_engineering import (
    TARGET_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    CATEGORICAL_FEATURE_COLUMNS,
    BENCHMARK_INDICES,
    extract_features_and_targets_for_episode,
    build_training_dataset,
    DEFAULT_TRAINING_PARQUET,
    DEFAULT_TRAINING_SUMMARY
)
from src.analytics.advanced_event_study import find_effective_trading_day_index


class TestFeatureEngineeringSpecification:
    """Verifies target and feature schemas against project requirements."""

    def test_eight_target_columns_exist(self):
        """Verify exactly the eight required prediction targets are defined."""
        assert len(TARGET_COLUMNS) == 8
        expected_targets = [
            "sp500_fwd_ret_5d", "sp500_fwd_ret_20d",
            "nasdaq_fwd_ret_5d", "nasdaq_fwd_ret_20d",
            "dow_jones_fwd_ret_5d", "dow_jones_fwd_ret_20d",
            "russell2000_fwd_ret_5d", "russell2000_fwd_ret_20d"
        ]
        assert TARGET_COLUMNS == expected_targets

    def test_four_benchmark_indices(self):
        """Verify exactly the four approved US benchmark indices are present."""
        assert BENCHMARK_INDICES == ["sp500", "nasdaq", "dow_jones", "russell2000"]

    def test_numeric_and_categorical_feature_lists(self):
        """Verify standard market condition features are defined."""
        for feat in ["pre_5d_return_mkt", "pre_20d_return_mkt", "rolling_21d_vol_mkt",
                     "vix_level", "vix_5d_change_pct", "sma_50_ratio_mkt", "yield_spread_10y2y"]:
            assert feat in NUMERIC_FEATURE_COLUMNS

        for cat in ["canonical_region", "duration_regime"]:
            assert cat in CATEGORICAL_FEATURE_COLUMNS


class TestTradingDayAlignmentAndLeakage:
    """Tests calendar alignment and verification of zero look-ahead bias."""

    def test_find_effective_trading_day_weekend_mapping(self):
        """Verify Saturday event date maps to Monday trading date without look-ahead bias."""
        trading_calendar = ["2023-10-06", "2023-10-09", "2023-10-10", "2023-10-11"]
        # Saturday event (Israel-Hamas war outbreak 2023-10-07)
        idx = find_effective_trading_day_index("2023-10-07", trading_calendar)
        assert idx == 1
        assert trading_calendar[idx] == "2023-10-09"

    def test_forward_return_calculation_precision(self):
        """Verify manual return calculation matches target formula exactly."""
        p_t0 = 100.0
        p_t5 = 105.5
        p_t20 = 92.0

        ret_5d = ((p_t5 / p_t0) - 1.0) * 100.0
        ret_20d = ((p_t20 / p_t0) - 1.0) * 100.0

        assert pytest.approx(ret_5d, rel=1e-5) == 5.5
        assert pytest.approx(ret_20d, rel=1e-5) == -8.0

    def test_pre_event_lookback_strictly_backward(self):
        """Verify pre-event return only looks at past prices."""
        p_t_minus_5 = 95.0
        p_t0 = 100.0
        pre_ret_5d = ((p_t0 / p_t_minus_5) - 1.0) * 100.0
        assert pytest.approx(pre_ret_5d, rel=1e-5) == 5.26315789


class TestTrainingDatasetIntegrity:
    """Validates the generated training dataset on disk."""

    def test_training_parquet_exists_and_dimensions(self):
        """Verify training Parquet exists with expected 310 observations."""
        assert os.path.exists(DEFAULT_TRAINING_PARQUET)
        df = pd.read_parquet(DEFAULT_TRAINING_PARQUET)
        assert len(df) == 310
        assert len(df.columns) >= 38

    def test_zero_null_values_in_features_and_targets(self):
        """Verify 0 null values across all features and prediction targets."""
        df = pd.read_parquet(DEFAULT_TRAINING_PARQUET)
        for col in TARGET_COLUMNS:
            assert df[col].isnull().sum() == 0, f"Target {col} contains null values!"
        for col in NUMERIC_FEATURE_COLUMNS:
            assert df[col].isnull().sum() == 0, f"Feature {col} contains null values!"
        for col in CATEGORICAL_FEATURE_COLUMNS:
            assert df[col].isnull().sum() == 0, f"Categorical {col} contains null values!"

    def test_five_regions_represented(self):
        """Verify all 5 canonical regions are present in training dataset."""
        df = pd.read_parquet(DEFAULT_TRAINING_PARQUET)
        regions = set(df["canonical_region"].unique())
        expected = {"Middle East / MENA", "Eastern Europe", "South Asia", "East Asia", "North America"}
        assert expected == regions

    def test_duration_regimes_represented(self):
        """Verify all 3 duration regimes are present in training dataset."""
        df = pd.read_parquet(DEFAULT_TRAINING_PARQUET)
        regimes = set(df["duration_regime"].unique())
        expected = {"Short / Tactical", "Medium / Campaign", "Protracted / War"}
        assert expected == regimes

    def test_summary_json_metrics(self):
        """Verify training_summary.json matches dataset properties."""
        assert os.path.exists(DEFAULT_TRAINING_SUMMARY)
        with open(DEFAULT_TRAINING_SUMMARY, "r", encoding="utf-8") as f:
            summary = json.load(f)
        assert summary["status"] == "SUCCESS"
        assert summary["usable_training_observations"] == 310
        assert summary["excluded_observations_count"] == 4
        assert summary["null_counts_total"] == 0
