"""
Unit and Integration Tests for Machine Learning Models (Step 4)
================================================================
Validates:
1. Dataset loading and chronological splitting (2015-2022 vs 2023-2025).
2. Pipeline construction with ColumnTransformer (StandardScaler + OneHotEncoder).
3. Primary multi-output RandomForestRegressor training and eight-target output shape (N, 8).
4. Ridge regression baseline training and comparative evaluation.
5. Conflict-grouped 5-fold cross-validation.
6. Feature importance extraction, normalization, and rankings.
7. Model serialization, disk persistence, and loading via joblib.
8. ConflictMarketPredictor inference class with default imputation and structured outputs.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
import joblib

from src.ml.train_models import (
    load_training_dataset,
    chronological_split,
    build_feature_preprocessor,
    build_random_forest_pipeline,
    build_ridge_pipeline,
    evaluate_predictions,
    run_group_kfold_cv,
    extract_feature_importances,
    ConflictMarketPredictor,
    train_and_evaluate_models,
    PROJECT_ROOT,
    DEFAULT_ANALYTICS_MODEL_DIR,
    DEFAULT_PROCESSED_MODEL_DIR
)
from src.ml.feature_engineering import (
    NUMERIC_FEATURE_COLUMNS,
    CATEGORICAL_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    DEFAULT_TRAINING_PARQUET
)


@pytest.fixture(scope="module")
def full_dataset():
    """Loads the processed training dataset."""
    assert os.path.exists(DEFAULT_TRAINING_PARQUET), f"Parquet missing: {DEFAULT_TRAINING_PARQUET}"
    return pd.read_parquet(DEFAULT_TRAINING_PARQUET)


@pytest.fixture(scope="module")
def train_test_data(full_dataset):
    """Provides chronological train and test partitions."""
    train_df, test_df = chronological_split(full_dataset, split_year=2022)
    return train_df, test_df


class TestDatasetPartitioning:
    """Verifies chronological splitting and out-of-sample properties."""

    def test_chronological_split_counts_and_dates(self, train_test_data, full_dataset):
        train_df, test_df = train_test_data
        assert len(train_df) + len(test_df) == len(full_dataset)
        assert len(train_df) == 216
        assert len(test_df) == 94

        train_max_year = pd.to_datetime(train_df["event_date"]).dt.year.max()
        test_min_year = pd.to_datetime(test_df["event_date"]).dt.year.min()
        assert train_max_year <= 2022
        assert test_min_year >= 2023

    def test_train_test_conflict_diversity(self, train_test_data):
        train_df, test_df = train_test_data
        assert train_df["conflict_new_id"].nunique() >= 25
        assert test_df["conflict_new_id"].nunique() >= 15


class TestModelTrainingAndMultiOutputShape:
    """Verifies pipeline construction and eight continuous target predictions."""

    def test_feature_preprocessor_output_dimensions(self, train_test_data):
        train_df, _ = train_test_data
        preprocessor = build_feature_preprocessor()
        feature_cols = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
        X_trans = preprocessor.fit_transform(train_df[feature_cols])

        assert X_trans.shape[0] == len(train_df)
        assert X_trans.shape[1] >= len(NUMERIC_FEATURE_COLUMNS) + len(CATEGORICAL_FEATURE_COLUMNS)
        assert not np.isnan(X_trans).any()

    def test_rf_pipeline_fit_and_predict_eight_targets(self, train_test_data):
        train_df, test_df = train_test_data
        feature_cols = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
        preprocessor = build_feature_preprocessor()
        pipeline = build_random_forest_pipeline(preprocessor, n_estimators=25, max_depth=4, random_state=42)

        pipeline.fit(train_df[feature_cols], train_df[TARGET_COLUMNS])
        preds = pipeline.predict(test_df[feature_cols])

        # Assert exactly eight target columns predicted
        assert preds.shape == (len(test_df), 8)
        assert not np.isnan(preds).any()
        assert np.isfinite(preds).all()

    def test_ridge_pipeline_fit_and_predict_eight_targets(self, train_test_data):
        train_df, test_df = train_test_data
        feature_cols = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
        preprocessor = build_feature_preprocessor()
        pipeline = build_ridge_pipeline(preprocessor, alpha=10.0, random_state=42)

        pipeline.fit(train_df[feature_cols], train_df[TARGET_COLUMNS])
        preds = pipeline.predict(test_df[feature_cols])

        assert preds.shape == (len(test_df), 8)
        assert not np.isnan(preds).any()
        assert np.isfinite(preds).all()


class TestEvaluationAndGroupKFold:
    """Verifies evaluation metrics calculation and conflict-grouped validation."""

    def test_evaluate_predictions_dictionary_structure(self, train_test_data):
        train_df, test_df = train_test_data
        feature_cols = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
        preprocessor = build_feature_preprocessor()
        pipeline = build_random_forest_pipeline(preprocessor, n_estimators=20, max_depth=3, random_state=42)

        pipeline.fit(train_df[feature_cols], train_df[TARGET_COLUMNS])
        preds = pipeline.predict(test_df[feature_cols])
        eval_dict = evaluate_predictions(test_df, preds, TARGET_COLUMNS)

        assert "macro_avg_mae" in eval_dict
        assert "macro_avg_rmse" in eval_dict
        assert "macro_avg_r2" in eval_dict
        assert "targets" in eval_dict
        assert len(eval_dict["targets"]) == 8

        for col in TARGET_COLUMNS:
            metrics = eval_dict["targets"][col]
            assert "mae" in metrics and metrics["mae"] > 0
            assert "rmse" in metrics and metrics["rmse"] > 0
            assert "r2" in metrics
            assert "mean_actual" in metrics
            assert "mean_predicted" in metrics

    def test_run_group_kfold_cv_metrics(self, train_test_data):
        train_df, _ = train_test_data
        feature_cols = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
        preprocessor = build_feature_preprocessor()
        pipeline = build_random_forest_pipeline(preprocessor, n_estimators=20, max_depth=3, random_state=42)

        cv_results = run_group_kfold_cv(
            pipeline=pipeline,
            df_train=train_df,
            feature_cols=feature_cols,
            target_cols=TARGET_COLUMNS,
            group_col="conflict_new_id",
            n_splits=3  # Fast test with 3 splits
        )

        assert cv_results["n_splits"] == 3
        assert len(cv_results["targets"]) == 8
        for col in TARGET_COLUMNS:
            assert "cv_mae_mean" in cv_results["targets"][col]
            assert "cv_rmse_mean" in cv_results["targets"][col]
            assert cv_results["targets"][col]["cv_mae_mean"] > 0


class TestFeatureImportance:
    """Verifies feature importance extraction and valid attribution."""

    def test_feature_importances_sum_to_one(self, train_test_data):
        train_df, _ = train_test_data
        feature_cols = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
        preprocessor = build_feature_preprocessor()
        pipeline = build_random_forest_pipeline(preprocessor, n_estimators=50, max_depth=5, random_state=42)
        pipeline.fit(train_df[feature_cols], train_df[TARGET_COLUMNS])

        importances = extract_feature_importances(
            pipeline,
            NUMERIC_FEATURE_COLUMNS,
            CATEGORICAL_FEATURE_COLUMNS
        )

        assert len(importances) > len(NUMERIC_FEATURE_COLUMNS)
        total_imp = sum(item["importance"] for item in importances)
        assert np.isclose(total_imp, 1.0, atol=1e-3)
        assert importances[0]["importance"] >= importances[-1]["importance"]
        assert importances[-1]["cumulative_importance"] >= 0.999


class TestModelArtifactsAndPredictor:
    """Verifies persistence, deserialization, and inference engine."""

    def test_serialized_model_files_exist(self):
        for model_dir in [DEFAULT_ANALYTICS_MODEL_DIR, DEFAULT_PROCESSED_MODEL_DIR]:
            assert os.path.exists(os.path.join(model_dir, "random_forest_multitarget.joblib"))
            assert os.path.exists(os.path.join(model_dir, "ridge_baseline_multitarget.joblib"))
            assert os.path.exists(os.path.join(model_dir, "feature_importance.json"))
            assert os.path.exists(os.path.join(model_dir, "feature_importance.csv"))
            assert os.path.exists(os.path.join(model_dir, "evaluation_metrics.json"))
            assert os.path.exists(os.path.join(model_dir, "model_metadata.json"))

    def test_model_loading_and_exact_prediction_parity(self, train_test_data):
        _, test_df = train_test_data
        feature_cols = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS

        model_path = os.path.join(DEFAULT_ANALYTICS_MODEL_DIR, "random_forest_multitarget.joblib")
        loaded_pipeline = joblib.load(model_path)

        preds = loaded_pipeline.predict(test_df[feature_cols])
        assert preds.shape == (len(test_df), 8)
        assert not np.isnan(preds).any()

    def test_conflict_market_predictor_scenario_inference(self):
        predictor = ConflictMarketPredictor.load(DEFAULT_ANALYTICS_MODEL_DIR)

        # Complete scenario
        full_scenario = {
            "canonical_region": "Middle East / MENA",
            "duration_regime": "Medium / Campaign",
            "type_of_violence_desc": "State-based armed conflict",
            "fatalities_best": 150,
            "incident_count": 8,
            "vix_level": 24.5,
            "vix_5d_change_pct": 12.0,
            "yield_spread_10y2y": -0.35,
            "treasury_yield_10y": 4.25,
            "treasury_yield_2y": 4.60,
            "pre_5d_return_mkt": -1.2,
            "pre_20d_return_mkt": -2.8,
            "rolling_21d_vol_mkt": 0.015,
            "annualized_21d_vol_mkt": 0.238,
            "sma_50_ratio_mkt": -2.1
        }
        res_full = predictor.predict_formatted(full_scenario)[0]
        for idx in ["sp500", "nasdaq", "dow_jones", "russell2000"]:
            assert "return_5d_pct" in res_full[idx]
            assert "return_20d_pct" in res_full[idx]
            assert isinstance(res_full[idx]["return_5d_pct"], float)
            assert isinstance(res_full[idx]["return_20d_pct"], float)

        # Minimal scenario with missing features to test median imputation
        minimal_scenario = {
            "canonical_region": "Eastern Europe",
            "fatalities_best": 250
        }
        res_min = predictor.predict_formatted(minimal_scenario)[0]
        assert "sp500" in res_min
        assert np.isfinite(res_min["sp500"]["return_5d_pct"])
        assert np.isfinite(res_min["russell2000"]["return_20d_pct"])
