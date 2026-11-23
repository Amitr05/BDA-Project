"""
Unit and Integration Tests for Inference Engine and Predictor (Step 5)
======================================================================
Tests:
1. Loading the trained ConflictPredictor from disk.
2. Exact duration regime mapping (<30d, 30-180d, >180d).
3. Geographical region validation across all five canonical regions.
4. Generation of all eight continuous return predictions across the four benchmark indices.
5. Exact output DataFrame formatting for Streamlit presentation.
6. Regional sparsity warnings for East Asia and North America.
7. Dynamic econometric explanation synthesis based on feature importances.
8. Retrieval of feature importances and evaluation scorecards for dashboard diagnostics.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.ml.predictor import (
    load_conflict_predictor,
    map_duration_to_regime,
    CANONICAL_REGIONS,
    REGIONAL_SPARSITY_WARNINGS,
    REGIONAL_OBSERVATION_COUNTS,
    ConflictPredictor,
    ANALYTICS_MODEL_DIR
)


@pytest.fixture(scope="module")
def predictor():
    """Provides an instantiated ConflictPredictor loaded from serialized model artifacts."""
    return load_conflict_predictor(ANALYTICS_MODEL_DIR)


class TestDurationMapping:
    """Verifies duration regime mapping boundaries."""

    def test_short_tactical_mapping(self):
        assert map_duration_to_regime(1) == "Short / Tactical"
        assert map_duration_to_regime(15) == "Short / Tactical"
        assert map_duration_to_regime(29.9) == "Short / Tactical"

    def test_medium_campaign_mapping(self):
        assert map_duration_to_regime(30) == "Medium / Campaign"
        assert map_duration_to_regime(90) == "Medium / Campaign"
        assert map_duration_to_regime(180) == "Medium / Campaign"

    def test_protracted_war_mapping(self):
        assert map_duration_to_regime(181) == "Protracted / War"
        assert map_duration_to_regime(365) == "Protracted / War"
        assert map_duration_to_regime(720) == "Protracted / War"


class TestRegionValidationAndSparsity:
    """Verifies canonical region enforcement and data sparsity warnings."""

    def test_all_five_canonical_regions_accepted(self, predictor):
        for region in CANONICAL_REGIONS:
            result = predictor.predict_scenario(
                region=region,
                expected_duration_days=30,
                fatalities=50
            )
            assert result["scenario"]["canonical_region"] == region

    def test_invalid_region_rejected(self, predictor):
        with pytest.raises(ValueError, match="Invalid region"):
            predictor.predict_scenario(
                region="Western Europe",  # Excluded region
                expected_duration_days=30
            )

    def test_east_asia_sparsity_warning(self, predictor):
        result = predictor.predict_scenario(
            region="East Asia",
            expected_duration_days=15
        )
        assert result["sparsity_warning"] is not None
        assert "Severe Historical Data Sparsity" in result["sparsity_warning"]
        assert result["scenario"]["regional_sample_size"] == 3

    def test_north_america_sparsity_warning(self, predictor):
        result = predictor.predict_scenario(
            region="North America",
            expected_duration_days=10
        )
        assert result["sparsity_warning"] is not None
        assert "Moderate Historical Data Sparsity" in result["sparsity_warning"]
        assert result["scenario"]["regional_sample_size"] == 16

    def test_middle_east_no_sparsity_warning(self, predictor):
        result = predictor.predict_scenario(
            region="Middle East / MENA",
            expected_duration_days=60
        )
        assert result["sparsity_warning"] is None
        assert result["scenario"]["regional_sample_size"] == 121


class TestEightTargetPredictionOutput:
    """Verifies that all eight continuous returns are produced in the exact requested format."""

    def test_table_structure_and_index_order(self, predictor):
        result = predictor.predict_scenario(
            region="Middle East / MENA",
            expected_duration_days=45,
            fatalities=120,
            vix_level=22.0
        )
        table = result["prediction_table"]

        assert isinstance(table, pd.DataFrame)
        assert list(table.columns) == ["Index", "5-Day Predicted Change", "20-Day Predicted Change"]
        assert len(table) == 4
        expected_indices = ["S&P 500", "NASDAQ Composite", "Dow Jones", "Russell 2000"]
        assert list(table["Index"]) == expected_indices

    def test_prediction_values_are_finite_and_sensible(self, predictor):
        result = predictor.predict_scenario(
            region="South Asia",
            expected_duration_days=90,
            fatalities=80,
            vix_level=18.5,
            yield_spread_10y2y=0.45
        )
        raw = result["raw_predictions"]

        assert len(raw) == 8
        for k, v in raw.items():
            assert isinstance(v, float)
            assert np.isfinite(v)
            # Returns should be realistic percentages (between -40% and +40%)
            assert -40.0 <= v <= 40.0

    def test_explanation_contains_key_drivers(self, predictor):
        result = predictor.predict_scenario(
            region="Eastern Europe",
            expected_duration_days=120,
            fatalities=200,
            vix_level=28.0,
            market_trend_sma_ratio=-3.5,
            yield_spread_10y2y=-0.50
        )
        explanation = result["explanation"]
        assert "Key Drivers Identified by Random Forest" in explanation
        assert "Implied Volatility" in explanation
        assert "Market Trend" in explanation
        assert "Monetary Conditioning" in explanation

    def test_feature_importances_and_metrics_retrieval(self, predictor):
        df_imp = predictor.get_feature_importances_df()
        assert not df_imp.empty
        assert "feature" in df_imp.columns
        assert "importance" in df_imp.columns
        assert df_imp["importance"].sum() > 0.99

        metrics = predictor.get_evaluation_metrics_dict()
        assert "models" in metrics
        assert "random_forest" in metrics["models"]
        assert "ridge_regression" in metrics["models"]
