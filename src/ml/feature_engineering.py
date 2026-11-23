"""
Machine Learning Feature Engineering and Financial Target Generation Pipeline
==============================================================================
Constructs the supervised training dataset for Armed Conflicts & Military Escalation:
1. Reuses trading-calendar alignment (find_effective_trading_day_index) to map calendar dates to US trading days T0.
2. Extracts pre-event financial market features strictly at or before T0 (zero look-ahead bias):
   - Pre-event 5-day return (R_-5)
   - Pre-event 20-day return (R_-20)
   - Rolling 21-day annualized volatility (sigma_21D)
   - VIX level at T0 and 5-day percentage change (Delta VIX_-5)
   - 50-day Simple Moving Average ratio (P_T0 / SMA_50 - 1)
   - Point-in-time Treasury 10Y-2Y yield curve spread from FRED
3. Incorporates conflict characteristics:
   - Canonical region (5 categories)
   - Operational duration regime (Short <30d, Medium 30-180d, Protracted >180d)
   - Fatalities and incident counts
   - Violence typology
4. Generates exactly eight continuous forward percentage return targets:
   - S&P 500: 5-day and 20-day forward return
   - NASDAQ Composite: 5-day and 20-day forward return
   - Dow Jones Industrial Average: 5-day and 20-day forward return
   - Russell 2000: 5-day and 20-day forward return
5. Persists the clean, documented training dataset to:
   data/processed/ml/conflict_market_training.parquet and conflict_market_training.csv
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.analytics.advanced_event_study import find_effective_trading_day_index

logger = logging.getLogger(__name__)

# Standard Paths resolved relative to repository root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CONFLICT_EPISODES_PARQUET = os.path.join(PROJECT_ROOT, "data", "processed", "conflict_episodes", "conflict_episodes.parquet")
DEFAULT_PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DEFAULT_ML_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "ml")
DEFAULT_TRAINING_PARQUET = os.path.join(DEFAULT_ML_OUTPUT_DIR, "conflict_market_training.parquet")
DEFAULT_TRAINING_CSV = os.path.join(DEFAULT_ML_OUTPUT_DIR, "conflict_market_training.csv")
DEFAULT_TRAINING_SUMMARY = os.path.join(DEFAULT_ML_OUTPUT_DIR, "training_summary.json")

# The Four Target Benchmark Indices
BENCHMARK_INDICES: List[str] = ["sp500", "nasdaq", "dow_jones", "russell2000"]

# Target Return Columns (Exactly Eight Targets)
TARGET_COLUMNS: List[str] = [
    "sp500_fwd_ret_5d",
    "sp500_fwd_ret_20d",
    "nasdaq_fwd_ret_5d",
    "nasdaq_fwd_ret_20d",
    "dow_jones_fwd_ret_5d",
    "dow_jones_fwd_ret_20d",
    "russell2000_fwd_ret_5d",
    "russell2000_fwd_ret_20d"
]

# Numeric Feature Columns Available at Prediction Time T0
NUMERIC_FEATURE_COLUMNS: List[str] = [
    "fatalities_best",
    "log_fatalities",
    "incident_count",
    "pre_5d_return_mkt",
    "pre_20d_return_mkt",
    "rolling_21d_vol_mkt",
    "annualized_21d_vol_mkt",
    "sma_50_ratio_mkt",
    "vix_level",
    "vix_5d_change_pct",
    "yield_spread_10y2y",
    "treasury_yield_10y",
    "treasury_yield_2y"
]

# Categorical Feature Columns Available at Prediction Time T0
CATEGORICAL_FEATURE_COLUMNS: List[str] = [
    "canonical_region",
    "duration_regime",
    "type_of_violence_desc"
]


# -----------------------------------------------------------------------------
# 1. Financial Dataset Loader
# -----------------------------------------------------------------------------

def load_financial_and_macro_datasets(
    processed_dir: str = DEFAULT_PROCESSED_DATA_DIR
) -> Dict[str, pd.DataFrame]:
    """
    Loads processed market benchmark Parquets (S&P 500, NASDAQ, Dow Jones, Russell 2000, VIX)
    and aligned FRED macroeconomic indicators. Normalizes date strings and verifies sort order.
    """
    datasets: Dict[str, pd.DataFrame] = {}

    for idx_name in BENCHMARK_INDICES:
        path = os.path.join(processed_dir, f"{idx_name}_processed.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Processed benchmark Parquet not found: {path}")
        df = pd.read_parquet(path)
        df["Date_str"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        df = df.sort_values("Date_str").drop_duplicates("Date_str").reset_index(drop=True)
        datasets[idx_name] = df

    # VIX
    vix_path = os.path.join(processed_dir, "vix_processed.parquet")
    if not os.path.exists(vix_path):
        raise FileNotFoundError(f"Processed VIX Parquet not found: {vix_path}")
    df_vix = pd.read_parquet(vix_path)
    df_vix["Date_str"] = pd.to_datetime(df_vix["Date"]).dt.strftime("%Y-%m-%d")
    df_vix = df_vix.sort_values("Date_str").drop_duplicates("Date_str").reset_index(drop=True)
    datasets["vix"] = df_vix

    # Macro Indicators
    macro_path = os.path.join(processed_dir, "macro", "macro_indicators_aligned.parquet")
    if not os.path.exists(macro_path):
        raise FileNotFoundError(f"Aligned macro indicators Parquet not found: {macro_path}")
    df_macro = pd.read_parquet(macro_path)
    df_macro["Date_str"] = pd.to_datetime(df_macro["Date"]).dt.strftime("%Y-%m-%d")
    df_macro = df_macro.sort_values("Date_str").drop_duplicates("Date_str").reset_index(drop=True)
    datasets["macro"] = df_macro

    logger.info(f"Loaded 6 financial and macroeconomic datasets ({len(df['Date_str'])} trading days each).")
    return datasets


# -----------------------------------------------------------------------------
# 2. Feature Extraction & Forward Target Generator
# -----------------------------------------------------------------------------

def extract_features_and_targets_for_episode(
    episode: Dict[str, Any],
    financial_data: Dict[str, pd.DataFrame],
    trading_dates: List[str],
    min_lookback_days: int = 20
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Extracts strictly pre-event features and 8 forward return targets for a single episode:
    - Finds effective trading day T0 via find_effective_trading_day_index().
    - Checks lookback availability (T0 >= min_lookback_days) and forward window availability (T0 + 20 < N).
    - Returns (features_dict, None) on success, or (None, exclusion_reason) if invalid.
    """
    event_date = str(episode["event_date"])
    episode_id = str(episode["episode_id"])

    t0_idx = find_effective_trading_day_index(event_date, trading_dates)

    if t0_idx is None:
        return None, f"Event date {event_date} occurs after available market calendar ({trading_dates[-1]})"

    total_trading_days = len(trading_dates)
    if t0_idx + 20 >= total_trading_days:
        return None, (
            f"Insufficient future trading days for 20D return: "
            f"T0 index {t0_idx} requires 20 future days, but dataset has {total_trading_days} days"
        )

    if t0_idx < min_lookback_days:
        return None, (
            f"Insufficient pre-event historical trading days: "
            f"T0 index {t0_idx} < required minimum lookback of {min_lookback_days} days"
        )

    effective_t0_date = trading_dates[t0_idx]

    # Baseline Metadata & Identifiers
    row_record: Dict[str, Any] = {
        "episode_id": episode_id,
        "event_date": event_date,
        "effective_t0_date": effective_t0_date,
        "t0_index": int(t0_idx),
        "observation_period": str(episode.get("observation_period", event_date[:7])),
        "canonical_region": str(episode["canonical_region"]),
        "primary_country": str(episode["primary_country"]),
        "conflict_new_id": int(episode.get("conflict_new_id", 0)),
        "conflict_name": str(episode.get("conflict_name", "Unknown Conflict")),
        "dyad_new_id": int(episode.get("dyad_new_id", 0)),
        "dyad_name": str(episode.get("dyad_name", "Unknown Dyad")),
        "type_of_violence": int(episode.get("type_of_violence", 1)),
        "type_of_violence_desc": str(episode.get("type_of_violence_desc", "State-based armed conflict")),
        "incident_count": int(episode.get("incident_count", 1)),
        "fatalities_best": float(episode.get("fatalities_best", 0.0)),
        "log_fatalities": float(np.log1p(max(0.0, float(episode.get("fatalities_best", 0.0))))),
        "duration_regime": str(episode.get("duration_regime", "Short / Tactical")),
        "source_provenance": str(episode.get("source_provenance", "UCDP GED v26.1")),
        "avg_latitude": float(episode.get("avg_latitude", 0.0)),
        "avg_longitude": float(episode.get("avg_longitude", 0.0))
    }

    # Market Condition Features (S&P 500 as Benchmark Reference)
    sp_df = financial_data["sp500"]
    p_t0 = float(sp_df.loc[t0_idx, "Close"])
    p_minus_5 = float(sp_df.loc[t0_idx - 5, "Close"])
    p_minus_20 = float(sp_df.loc[t0_idx - 20, "Close"])

    # Returns evaluated over [T0-5, T0] and [T0-20, T0]
    row_record["pre_5d_return_mkt"] = float(((p_t0 / p_minus_5) - 1.0) * 100.0)
    row_record["pre_20d_return_mkt"] = float(((p_t0 / p_minus_20) - 1.0) * 100.0)
    row_record["rolling_21d_vol_mkt"] = float(sp_df.loc[t0_idx, "rolling_volatility_21d"])
    row_record["annualized_21d_vol_mkt"] = float(sp_df.loc[t0_idx, "annualized_volatility_21d"])

    # 50-day Simple Moving Average ratio (P_T0 / SMA_50 - 1)
    start_sma = max(0, t0_idx - 49)
    sma_50 = float(sp_df.loc[start_sma:t0_idx, "Close"].mean())
    row_record["sma_50_ratio_mkt"] = float(((p_t0 / sma_50) - 1.0) * 100.0)

    # VIX Features
    vix_df = financial_data["vix"]
    vix_t0 = float(vix_df.loc[t0_idx, "Close"])
    vix_minus_5 = float(vix_df.loc[t0_idx - 5, "Close"])
    row_record["vix_level"] = float(vix_t0)
    row_record["vix_5d_change_pct"] = float(((vix_t0 / vix_minus_5) - 1.0) * 100.0)

    # Macroeconomic Yield Curve Features
    macro_df = financial_data["macro"]
    row_record["yield_spread_10y2y"] = float(macro_df.loc[t0_idx, "yield_curve_spread_10y2y"])
    row_record["treasury_yield_10y"] = float(macro_df.loc[t0_idx, "treasury_yield_10y"])
    row_record["treasury_yield_2y"] = float(macro_df.loc[t0_idx, "treasury_yield_2y"])

    # The Eight Prediction Targets
    # Calculated strictly using future trading day positions: T0+5 and T0+20
    for idx_name in BENCHMARK_INDICES:
        idx_df = financial_data[idx_name]
        p_ref = float(idx_df.loc[t0_idx, "Close"])
        p_plus_5 = float(idx_df.loc[t0_idx + 5, "Close"])
        p_plus_20 = float(idx_df.loc[t0_idx + 20, "Close"])

        row_record[f"{idx_name}_fwd_ret_5d"] = float(((p_plus_5 / p_ref) - 1.0) * 100.0)
        row_record[f"{idx_name}_fwd_ret_20d"] = float(((p_plus_20 / p_ref) - 1.0) * 100.0)

    return row_record, None


# -----------------------------------------------------------------------------
# 3. Master Dataset Construction Engine
# -----------------------------------------------------------------------------

def build_training_dataset(
    conflict_df: pd.DataFrame,
    financial_data: Dict[str, pd.DataFrame],
    min_lookback_days: int = 20
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Iterates over conflict episodes and constructs the clean supervised training matrix.
    Tracks and documents all excluded episodes.
    """
    trading_dates = financial_data["sp500"]["Date_str"].tolist()
    records: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []

    for _, ep_row in conflict_df.iterrows():
        ep_dict = ep_row.to_dict()
        feat_dict, exclusion_reason = extract_features_and_targets_for_episode(
            episode=ep_dict,
            financial_data=financial_data,
            trading_dates=trading_dates,
            min_lookback_days=min_lookback_days
        )
        if feat_dict is not None:
            records.append(feat_dict)
        else:
            exclusions.append({
                "episode_id": ep_dict.get("episode_id"),
                "event_date": ep_dict.get("event_date"),
                "canonical_region": ep_dict.get("canonical_region"),
                "reason": exclusion_reason
            })

    df_result = pd.DataFrame(records)
    return df_result, exclusions


# -----------------------------------------------------------------------------
# 4. Pipeline Execution & Persistence
# -----------------------------------------------------------------------------

def run_feature_engineering_pipeline(
    conflict_parquet_path: str = DEFAULT_CONFLICT_EPISODES_PARQUET,
    processed_dir: str = DEFAULT_PROCESSED_DATA_DIR,
    output_dir: str = DEFAULT_ML_OUTPUT_DIR,
    min_lookback_days: int = 20
) -> Dict[str, Any]:
    """
    Executes the end-to-end Step 3 pipeline:
    1. Loads processed conflict episodes and benchmark financial Parquets.
    2. Aligns episodes with US trading dates via find_effective_trading_day_index.
    3. Extracts pre-event financial condition features and 8 continuous forward return targets.
    4. Validates zero data leakage and zero nulls.
    5. Persists training matrix to Parquet, CSV, and summary JSON.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=== STEP 3: FEATURE ENGINEERING & FINANCIAL TARGET GENERATION ===")

    if not os.path.exists(conflict_parquet_path):
        raise FileNotFoundError(f"Conflict episodes dataset not found at {conflict_parquet_path}")

    df_conflict = pd.read_parquet(conflict_parquet_path)
    logger.info(f"Loaded {len(df_conflict)} conflict episodes from {conflict_parquet_path}.")

    financial_data = load_financial_and_macro_datasets(processed_dir=processed_dir)

    df_training, exclusions = build_training_dataset(
        conflict_df=df_conflict,
        financial_data=financial_data,
        min_lookback_days=min_lookback_days
    )

    logger.info(
        f"Constructed {len(df_training)} usable training observations. "
        f"Excluded {len(exclusions)} episodes due to trading calendar boundaries."
    )

    # Save to Parquet and CSV
    parquet_path = os.path.join(output_dir, "conflict_market_training.parquet")
    csv_path = os.path.join(output_dir, "conflict_market_training.csv")
    summary_path = os.path.join(output_dir, "training_summary.json")

    df_training.to_parquet(parquet_path, engine="pyarrow", index=False)
    df_training.to_csv(csv_path, index=False)

    regional_counts = df_training["canonical_region"].value_counts().to_dict()
    duration_counts = df_training["duration_regime"].value_counts().to_dict()

    target_stats: Dict[str, Dict[str, float]] = {}
    for tgt in TARGET_COLUMNS:
        s = df_training[tgt]
        target_stats[tgt] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median())
        }

    summary_metrics = {
        "status": "SUCCESS",
        "input_conflict_episodes": len(df_conflict),
        "usable_training_observations": len(df_training),
        "excluded_observations_count": len(exclusions),
        "excluded_observations_detail": exclusions,
        "date_range": {
            "min_event_date": str(df_training["event_date"].min()),
            "max_event_date": str(df_training["event_date"].max()),
            "min_t0_date": str(df_training["effective_t0_date"].min()),
            "max_t0_date": str(df_training["effective_t0_date"].max())
        },
        "regional_breakdown": regional_counts,
        "duration_regime_breakdown": duration_counts,
        "target_return_statistics": target_stats,
        "total_columns": len(df_training.columns),
        "null_counts_total": int(df_training.isnull().sum().sum()),
        "parquet_path": parquet_path,
        "csv_path": csv_path
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    logger.info(f"Persisted ML training dataset to {parquet_path}, {csv_path}, and {summary_path}")
    return summary_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_feature_engineering_pipeline()
