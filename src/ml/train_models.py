"""
Machine Learning Model Development and Cross-Validation Pipeline
================================================================
Trains, validates, and serializes multi-output regression models predicting
financial market impacts of War and Armed Conflicts on four US benchmark indices:
- S&P 500 (^GSPC): 5-day and 20-day returns
- NASDAQ Composite (^IXIC): 5-day and 20-day returns
- Dow Jones Industrial Average (^DJI): 5-day and 20-day returns
- Russell 2000 (^RUT): 5-day and 20-day returns

Architectural Choices:
1. Primary Model: Multi-output RandomForestRegressor (n_estimators=100, max_depth=5, min_samples_split=5)
   - Natively predicts all eight continuous returns simultaneously.
   - Captures non-linear volatility regimes, trend thresholds, and geopolitical conditioning.
   - Constrained depth prevents overfitting on financial time-series noise.
2. Baseline Model: Multi-output Ridge Regression (alpha=10.0) with StandardScaler
   - Linear econometric benchmark with L2 regularization.
3. Dual Validation:
   - Chronological Out-of-Sample Split: Train on 2015-2022 (216 episodes), Test on 2023-2025 (94 episodes).
   - GroupKFold Cross-Validation: 5 folds grouped by conflict_new_id within the training set.
4. Feature Importance Extraction:
   - Extracts MDI (Mean Decrease in Impurity) mapped to one-hot feature names.
5. Serialization:
   - Persists fitted pipelines, feature schemas, evaluation metrics, and metadata to data/analytics/ml_models/.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from src.ml.feature_engineering import (
    NUMERIC_FEATURE_COLUMNS,
    CATEGORICAL_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    DEFAULT_TRAINING_PARQUET
)

logger = logging.getLogger(__name__)

# Standard Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_ANALYTICS_MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "analytics", "ml_models")
DEFAULT_PROCESSED_MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "ml", "models")


# -----------------------------------------------------------------------------
# 1. Dataset Loading and Chronological Split
# -----------------------------------------------------------------------------

def load_training_dataset(parquet_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads the supervised conflict-market training dataset.
    """
    path = parquet_path or DEFAULT_TRAINING_PARQUET
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training dataset not found at: {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded training dataset from {path} with shape {df.shape}")
    return df


def chronological_split(
    df: pd.DataFrame,
    split_year: int = 2022
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset chronologically based on event_date:
    - Train: event_date <= split_year-12-31 (2015–2022)
    - Test:  event_date >= (split_year + 1)-01-01 (2023–2025)

    Guarantees strict out-of-sample forward evaluation with zero temporal leakage.
    """
    df_copy = df.copy()
    df_copy["_event_dt"] = pd.to_datetime(df_copy["event_date"])
    
    train_mask = df_copy["_event_dt"].dt.year <= split_year
    test_mask = df_copy["_event_dt"].dt.year > split_year
    
    train_df = df[train_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)
    
    logger.info(
        f"Chronological split at year {split_year}: "
        f"Train={len(train_df)} ({len(train_df)/len(df):.1%}), "
        f"Test={len(test_df)} ({len(test_df)/len(df):.1%})"
    )
    return train_df, test_df


# -----------------------------------------------------------------------------
# 2. Pipeline Builders
# -----------------------------------------------------------------------------

def build_feature_preprocessor(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer for feature preprocessing:
    - StandardScaler for continuous market and conflict numeric features.
    - OneHotEncoder(handle_unknown='ignore', sparse_output=False) for categorical features.
    """
    num_cols = numeric_features or NUMERIC_FEATURE_COLUMNS
    cat_cols = categorical_features or CATEGORICAL_FEATURE_COLUMNS

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ],
        remainder="drop"
    )
    return preprocessor


def build_random_forest_pipeline(
    preprocessor: ColumnTransformer,
    n_estimators: int = 100,
    max_depth: int = 5,
    min_samples_split: int = 5,
    random_state: int = 42
) -> Pipeline:
    """
    Constructs the primary multi-output Random Forest regression pipeline.
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", rf)])


def build_ridge_pipeline(
    preprocessor: ColumnTransformer,
    alpha: float = 10.0,
    random_state: int = 42
) -> Pipeline:
    """
    Constructs the baseline multi-output Ridge regression pipeline.
    """
    ridge = Ridge(alpha=alpha, random_state=random_state)
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", ridge)])


# -----------------------------------------------------------------------------
# 3. Model Evaluation Engine
# -----------------------------------------------------------------------------

def evaluate_predictions(
    y_true: Union[pd.DataFrame, np.ndarray],
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluates multi-target continuous predictions individually for each target:
    - Mean Absolute Error (MAE) in percentage points
    - Root Mean Squared Error (RMSE) in percentage points
    - R-squared (R²) score
    - Actual vs. predicted empirical means and standard deviations
    """
    targets = target_names or TARGET_COLUMNS
    if isinstance(y_true, pd.DataFrame):
        y_true_arr = y_true[targets].values
    else:
        y_true_arr = np.asarray(y_true)

    metrics_per_target = {}
    mae_list = []
    rmse_list = []
    r2_list = []

    for i, col in enumerate(targets):
        y_t = y_true_arr[:, i]
        y_p = y_pred[:, i]
        
        mae = float(mean_absolute_error(y_t, y_p))
        rmse = float(root_mean_squared_error(y_t, y_p))
        r2 = float(r2_score(y_t, y_p))
        
        metrics_per_target[col] = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "mean_actual": round(float(np.mean(y_t)), 4),
            "std_actual": round(float(np.std(y_t)), 4),
            "mean_predicted": round(float(np.mean(y_p)), 4),
            "std_predicted": round(float(np.std(y_p)), 4)
        }
        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)

    overall_metrics = {
        "macro_avg_mae": round(float(np.mean(mae_list)), 4),
        "macro_avg_rmse": round(float(np.mean(rmse_list)), 4),
        "macro_avg_r2": round(float(np.mean(r2_list)), 4),
        "targets": metrics_per_target
    }
    return overall_metrics


def run_group_kfold_cv(
    pipeline: Pipeline,
    df_train: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    group_col: str = "conflict_new_id",
    n_splits: int = 5
) -> Dict[str, Any]:
    """
    Performs conflict-grouped K-Fold cross validation on training data to assess
    cross-episode generalization and prevent within-conflict leakage across folds.
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = df_train[group_col].values
    
    fold_results = {t: {"mae": [], "rmse": [], "r2": []} for t in target_cols}

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_train, groups=groups)):
        X_tr = df_train.iloc[train_idx][feature_cols]
        Y_tr = df_train.iloc[train_idx][target_cols]
        X_va = df_train.iloc[val_idx][feature_cols]
        Y_va = df_train.iloc[val_idx][target_cols]

        pipeline.fit(X_tr, Y_tr)
        preds = pipeline.predict(X_va)

        for i, col in enumerate(target_cols):
            y_t = Y_va[col].values
            y_p = preds[:, i]
            fold_results[col]["mae"].append(mean_absolute_error(y_t, y_p))
            fold_results[col]["rmse"].append(root_mean_squared_error(y_t, y_p))
            fold_results[col]["r2"].append(r2_score(y_t, y_p))

    cv_summary = {}
    for col in target_cols:
        cv_summary[col] = {
            "cv_mae_mean": round(float(np.mean(fold_results[col]["mae"])), 4),
            "cv_mae_std": round(float(np.std(fold_results[col]["mae"])), 4),
            "cv_rmse_mean": round(float(np.mean(fold_results[col]["rmse"])), 4),
            "cv_rmse_std": round(float(np.std(fold_results[col]["rmse"])), 4),
            "cv_r2_mean": round(float(np.mean(fold_results[col]["r2"])), 4),
            "cv_r2_std": round(float(np.std(fold_results[col]["r2"])), 4)
        }

    return {
        "n_splits": n_splits,
        "group_column": group_col,
        "targets": cv_summary
    }


# -----------------------------------------------------------------------------
# 4. Feature Importance Extraction
# -----------------------------------------------------------------------------

def extract_feature_importances(
    pipeline: Pipeline,
    numeric_cols: List[str],
    cat_cols: List[str]
) -> List[Dict[str, Any]]:
    """
    Extracts feature importances from a fitted Random Forest pipeline.
    Resolves clean, human-readable feature names from the ColumnTransformer.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    rf = pipeline.named_steps["regressor"]

    raw_feature_names = preprocessor.get_feature_names_out()
    importances = rf.feature_importances_

    feature_info = []
    for raw_name, imp in zip(raw_feature_names, importances):
        # Format clean display name
        if raw_name.startswith("num__"):
            clean_name = raw_name.replace("num__", "")
            feat_type = "numeric"
        elif raw_name.startswith("cat__"):
            clean_name = raw_name.replace("cat__", "")
            feat_type = "categorical"
        else:
            clean_name = raw_name
            feat_type = "other"

        feature_info.append({
            "feature": clean_name,
            "raw_name": raw_name,
            "feature_type": feat_type,
            "importance": round(float(imp), 6)
        })

    # Sort descending by importance
    feature_info.sort(key=lambda x: x["importance"], reverse=True)
    
    # Calculate cumulative importance
    cum_imp = 0.0
    for item in feature_info:
        cum_imp += item["importance"]
        item["cumulative_importance"] = round(cum_imp, 6)

    return feature_info


# -----------------------------------------------------------------------------
# 5. Model Inference Class (For Dashboard and Step 5 Integration)
# -----------------------------------------------------------------------------

class ConflictMarketPredictor:
    """
    Unified Inference Engine for Conflict Financial Impact Prediction.
    Loads trained pipeline and metadata, validates feature inputs,
    imputes missing variables using training medians, and predicts all 8 targets.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        metadata: Dict[str, Any]
    ):
        self.pipeline = pipeline
        self.metadata = metadata
        self.numeric_features = metadata.get("numeric_features", NUMERIC_FEATURE_COLUMNS)
        self.categorical_features = metadata.get("categorical_features", CATEGORICAL_FEATURE_COLUMNS)
        self.target_columns = metadata.get("target_columns", TARGET_COLUMNS)
        self.feature_medians = metadata.get("feature_medians", {})

    @classmethod
    def load(cls, model_dir: Optional[str] = None) -> "ConflictMarketPredictor":
        """
        Loads the predictor from the serialized model directory.
        """
        search_dirs = [
            model_dir,
            DEFAULT_ANALYTICS_MODEL_DIR,
            DEFAULT_PROCESSED_MODEL_DIR
        ]
        chosen_dir = None
        for d in search_dirs:
            if d and os.path.exists(os.path.join(d, "random_forest_multitarget.joblib")):
                chosen_dir = d
                break

        if not chosen_dir:
            raise FileNotFoundError(
                f"Trained model 'random_forest_multitarget.joblib' not found in search paths: {search_dirs}"
            )

        model_path = os.path.join(chosen_dir, "random_forest_multitarget.joblib")
        meta_path = os.path.join(chosen_dir, "model_metadata.json")

        pipeline = joblib.load(model_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        logger.info(f"Loaded ConflictMarketPredictor from {chosen_dir}")
        return cls(pipeline=pipeline, metadata=metadata)

    def prepare_input_dataframe(self, input_data: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        """
        Formats input features into a validated DataFrame, imputing missing features.
        """
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()

        # Impute missing numeric features with medians
        for col in self.numeric_features:
            if col not in df.columns:
                df[col] = self.feature_medians.get(col, 0.0)
            else:
                df[col] = df[col].fillna(self.feature_medians.get(col, 0.0))

        # Impute missing categorical features
        for col in self.categorical_features:
            if col not in df.columns:
                if col == "canonical_region":
                    df[col] = "Middle East / MENA"
                elif col == "duration_regime":
                    df[col] = "Short / Tactical"
                else:
                    df[col] = "State-based armed conflict"
            else:
                df[col] = df[col].fillna("Unknown")

        return df[self.numeric_features + self.categorical_features]

    def predict(self, input_data: Union[pd.DataFrame, Dict[str, Any]]) -> np.ndarray:
        """
        Generates raw predictions of shape (N, 8).
        """
        X = self.prepare_input_dataframe(input_data)
        return self.pipeline.predict(X)

    def predict_formatted(self, input_data: Union[pd.DataFrame, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generates structured, human-readable predictions for each input scenario.
        Returns percentage returns partitioned by benchmark index and horizon.
        """
        raw_preds = self.predict(input_data)
        results = []
        for row in raw_preds:
            formatted = {
                "sp500": {
                    "return_5d_pct": round(float(row[0]), 3),
                    "return_20d_pct": round(float(row[1]), 3)
                },
                "nasdaq": {
                    "return_5d_pct": round(float(row[2]), 3),
                    "return_20d_pct": round(float(row[3]), 3)
                },
                "dow_jones": {
                    "return_5d_pct": round(float(row[4]), 3),
                    "return_20d_pct": round(float(row[5]), 3)
                },
                "russell2000": {
                    "return_5d_pct": round(float(row[6]), 3),
                    "return_20d_pct": round(float(row[7]), 3)
                },
                "summary": {
                    "avg_5d_return_pct": round(float(np.mean([row[0], row[2], row[4], row[6]])), 3),
                    "avg_20d_return_pct": round(float(np.mean([row[1], row[3], row[5], row[7]])), 3)
                }
            }
            results.append(formatted)
        return results


# -----------------------------------------------------------------------------
# 6. Artifact Serialization
# -----------------------------------------------------------------------------

def save_artifacts(
    pipeline_rf: Pipeline,
    pipeline_ridge: Pipeline,
    feature_importances: List[Dict[str, Any]],
    evaluation_results: Dict[str, Any],
    metadata: Dict[str, Any],
    output_dirs: Optional[List[str]] = None
) -> None:
    """
    Persists trained pipelines, metrics, importance lists, and metadata across target paths.
    """
    dirs = output_dirs or [DEFAULT_ANALYTICS_MODEL_DIR, DEFAULT_PROCESSED_MODEL_DIR]

    for out_dir in dirs:
        os.makedirs(out_dir, exist_ok=True)

        # 1. Pipelines
        joblib.dump(pipeline_rf, os.path.join(out_dir, "random_forest_multitarget.joblib"))
        joblib.dump(pipeline_ridge, os.path.join(out_dir, "ridge_baseline_multitarget.joblib"))

        # 2. Feature Importances
        imp_json_path = os.path.join(out_dir, "feature_importance.json")
        with open(imp_json_path, "w", encoding="utf-8") as f:
            json.dump(feature_importances, f, indent=2)

        imp_df = pd.DataFrame(feature_importances)
        imp_df.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)

        # 3. Evaluation Metrics
        eval_json_path = os.path.join(out_dir, "evaluation_metrics.json")
        with open(eval_json_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, indent=2)

        # 4. Metadata
        meta_json_path = os.path.join(out_dir, "model_metadata.json")
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Successfully saved all model artifacts to {out_dir}")


# -----------------------------------------------------------------------------
# 7. Full Training Orchestrator
# -----------------------------------------------------------------------------

def train_and_evaluate_models(
    dataset_path: Optional[str] = None,
    output_dirs: Optional[List[str]] = None
) -> Tuple[Pipeline, Pipeline, Dict[str, Any]]:
    """
    Executes full Step 4 Machine Learning training pipeline:
    1. Loads processed dataset.
    2. Performs chronological split (Train: 2015-2022, Test: 2023-2025).
    3. Builds ColumnTransformer and models.
    4. Runs GroupKFold cross-validation on training data.
    5. Fits full training pipelines for Random Forest and Ridge.
    6. Evaluates out-of-sample test metrics across all eight targets.
    7. Evaluates Mean Baseline benchmark.
    8. Extracts feature importances.
    9. Serializes all artifacts.
    """
    df = load_training_dataset(dataset_path)

    feature_cols = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
    target_cols = TARGET_COLUMNS

    # Chronological Split
    train_df, test_df = chronological_split(df, split_year=2022)

    X_train = train_df[feature_cols]
    Y_train = train_df[target_cols]
    X_test = test_df[feature_cols]
    Y_test = test_df[target_cols]

    # Preprocessors and Pipelines
    preprocessor = build_feature_preprocessor()
    rf_pipeline = build_random_forest_pipeline(preprocessor)
    ridge_pipeline = build_ridge_pipeline(preprocessor)

    # 1. Conflict-Grouped Cross-Validation on Training Set
    logger.info("Executing 5-Fold GroupKFold Cross-Validation on 2015-2022 training set...")
    rf_cv_results = run_group_kfold_cv(
        pipeline=rf_pipeline,
        df_train=train_df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        group_col="conflict_new_id",
        n_splits=5
    )

    # 2. Fit Primary Random Forest Model on Full Training Set
    logger.info("Fitting Random Forest Regressor on full 2015-2022 training set...")
    rf_pipeline.fit(X_train, Y_train)

    # 3. Fit Ridge Regression Baseline on Full Training Set
    logger.info("Fitting Ridge Regressor on full 2015-2022 training set...")
    ridge_pipeline.fit(X_train, Y_train)

    # 4. Out-of-Sample Predictions (2023-2025 Test Set)
    rf_test_preds = rf_pipeline.predict(X_test)
    ridge_test_preds = ridge_pipeline.predict(X_test)

    # 5. Mean Baseline Predictor (Econometric benchmark predicting training mean)
    train_target_means = Y_train.mean().values
    mean_baseline_preds = np.tile(train_target_means, (len(test_df), 1))

    # 6. Compute Evaluation Metrics
    rf_test_metrics = evaluate_predictions(Y_test, rf_test_preds, target_cols)
    ridge_test_metrics = evaluate_predictions(Y_test, ridge_test_preds, target_cols)
    mean_baseline_metrics = evaluate_predictions(Y_test, mean_baseline_preds, target_cols)

    rf_train_preds = rf_pipeline.predict(X_train)
    rf_train_metrics = evaluate_predictions(Y_train, rf_train_preds, target_cols)

    # 7. Extract Feature Importances
    feature_importances = extract_feature_importances(
        rf_pipeline,
        NUMERIC_FEATURE_COLUMNS,
        CATEGORICAL_FEATURE_COLUMNS
    )

    # 8. Feature Medians for Robust Missing-Value Imputation during Inference
    feature_medians = {
        col: float(train_df[col].median())
        for col in NUMERIC_FEATURE_COLUMNS
    }

    # Consolidated Evaluation Dictionary
    evaluation_summary = {
        "dataset_summary": {
            "total_samples": len(df),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "train_date_range": [str(train_df["event_date"].min()), str(train_df["event_date"].max())],
            "test_date_range": [str(test_df["event_date"].min()), str(test_df["event_date"].max())],
            "unique_conflicts_train": int(train_df["conflict_new_id"].nunique()),
            "unique_conflicts_test": int(test_df["conflict_new_id"].nunique())
        },
        "models": {
            "random_forest": {
                "algorithm": "RandomForestRegressor",
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "min_samples_split": 5,
                    "random_state": 42
                },
                "group_kfold_cv_train": rf_cv_results,
                "in_sample_train_metrics": rf_train_metrics,
                "out_of_sample_test_metrics": rf_test_metrics
            },
            "ridge_regression": {
                "algorithm": "Ridge",
                "hyperparameters": {
                    "alpha": 10.0,
                    "random_state": 42
                },
                "out_of_sample_test_metrics": ridge_test_metrics
            },
            "mean_baseline": {
                "algorithm": "HistoricalMeanPredictor",
                "out_of_sample_test_metrics": mean_baseline_metrics
            }
        }
    }

    # Model Metadata
    metadata = {
        "model_version": "1.0.0",
        "primary_model": "RandomForestRegressor",
        "target_columns": target_cols,
        "numeric_features": NUMERIC_FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURE_COLUMNS,
        "feature_medians": feature_medians,
        "canonical_regions": [
            "Middle East / MENA",
            "Eastern Europe",
            "South Asia",
            "East Asia",
            "North America"
        ],
        "duration_regimes": [
            "Short / Tactical",
            "Medium / Campaign",
            "Protracted / War"
        ],
        "train_samples": len(train_df),
        "test_samples": len(test_df)
    }

    # 9. Persist Artifacts
    save_artifacts(
        pipeline_rf=rf_pipeline,
        pipeline_ridge=ridge_pipeline,
        feature_importances=feature_importances,
        evaluation_results=evaluation_summary,
        metadata=metadata,
        output_dirs=output_dirs
    )

    return rf_pipeline, ridge_pipeline, evaluation_summary


# -----------------------------------------------------------------------------
# 8. Main CLI Execution
# -----------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=" * 80)
    print("STEP 4: MACHINE LEARNING MODEL DEVELOPMENT & CROSS-VALIDATION")
    print("=" * 80)

    rf_pipe, ridge_pipe, eval_results = train_and_evaluate_models()

    ds = eval_results["dataset_summary"]
    print(f"\n[+] Dataset Partitioning:")
    print(f"    - Total Observations : {ds['total_samples']}")
    print(f"    - Training (2015-2022): {ds['train_samples']} samples ({ds['unique_conflicts_train']} conflicts)")
    print(f"    - Testing  (2023-2025): {ds['test_samples']} samples ({ds['unique_conflicts_test']} conflicts)")

    print("\n[+] Out-of-Sample Test Evaluation Across All Eight Targets (2023-2025):")
    print("-" * 92)
    print(f"{'Target Variable':<24} | {'RF MAE':<9} {'Ridge MAE':<9} | {'RF RMSE':<9} {'Ridge RMSE':<10} | {'RF R²':<8} {'Ridge R²':<8}")
    print("-" * 92)

    rf_targets = eval_results["models"]["random_forest"]["out_of_sample_test_metrics"]["targets"]
    ridge_targets = eval_results["models"]["ridge_regression"]["out_of_sample_test_metrics"]["targets"]

    for col in TARGET_COLUMNS:
        rf_m = rf_targets[col]
        rd_m = ridge_targets[col]
        print(
            f"{col:<24} | {rf_m['mae']:>6.3f}%   {rd_m['mae']:>6.3f}%   | "
            f"{rf_m['rmse']:>6.3f}%   {rd_m['rmse']:>6.3f}%   | "
            f"{rf_m['r2']:>7.3f}  {rd_m['r2']:>7.3f}"
        )
    print("-" * 92)

    rf_overall = eval_results["models"]["random_forest"]["out_of_sample_test_metrics"]
    ridge_overall = eval_results["models"]["ridge_regression"]["out_of_sample_test_metrics"]
    print(
        f"{'Macro Average':<24} | {rf_overall['macro_avg_mae']:>6.3f}%   {ridge_overall['macro_avg_mae']:>6.3f}%   | "
        f"{rf_overall['macro_avg_rmse']:>6.3f}%   {ridge_overall['macro_avg_rmse']:>6.3f}%   | "
        f"{rf_overall['macro_avg_r2']:>7.3f}  {ridge_overall['macro_avg_r2']:>7.3f}"
    )

    print("\n[+] Top 10 Random Forest Features by Importance:")
    imp_path = os.path.join(DEFAULT_ANALYTICS_MODEL_DIR, "feature_importance.json")
    with open(imp_path, "r", encoding="utf-8") as f:
        imps = json.load(f)
    for rank, item in enumerate(imps[:10], 1):
        print(f"    {rank:2d}. {item['feature']:<30} : {item['importance']*100:>5.2f}% (Cumulative: {item['cumulative_importance']*100:>5.2f}%)")

    print("\n[+] Testing ConflictMarketPredictor Quick Inference:")
    predictor = ConflictMarketPredictor.load(DEFAULT_ANALYTICS_MODEL_DIR)
    test_scenario = {
        "canonical_region": "Middle East / MENA",
        "duration_regime": "Medium / Campaign",
        "fatalities_best": 120,
        "vix_level": 22.5,
        "yield_spread_10y2y": 0.20
    }
    sample_res = predictor.predict_formatted(test_scenario)[0]
    print(f"    Sample Scenario: Middle East / MENA, Medium Campaign, Fatalities=120, VIX=22.5")
    for idx_name in ["sp500", "nasdaq", "dow_jones", "russell2000"]:
        ret5 = sample_res[idx_name]["return_5d_pct"]
        ret20 = sample_res[idx_name]["return_20d_pct"]
        print(f"    - {idx_name.upper():<12}: 5D Return = {ret5:>+6.2f}%, 20D Return = {ret20:>+6.2f}%")

    print("\n[✓] Step 4 Model Training, Evaluation, and Serialization Complete.")


if __name__ == "__main__":
    main()
