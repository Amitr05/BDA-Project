"""
Conflict Impact Prediction Engine
=================================
Inference engine for Step 5: Accepts geopolitical conflict scenarios and prevailing
financial market conditions to generate forward return predictions across four US equity benchmarks:
- S&P 500
- NASDAQ Composite
- Dow Jones Industrial Average
- Russell 2000

Generates exactly eight numerical percentage returns:
- 5-trading-day predicted change (R_+5)
- 20-trading-day predicted change (R_+20)

Handles:
- Geographical region validation (5 canonical regions)
- Duration mapping (days -> Short/Tactical, Medium/Campaign, Protracted/War)
- Market regime conditioning (VIX, SMA trend, Treasury yields, yield spread)
- Data sparsity warnings for underrepresented regions (East Asia, North America)
- Evaluation metrics and feature importance retrieval for dashboard display
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ANALYTICS_MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "analytics", "ml_models")
PROCESSED_MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "ml", "models")

CANONICAL_REGIONS: List[str] = [
    "Middle East / MENA",
    "Eastern Europe",
    "South Asia",
    "East Asia",
    "North America"
]

DURATION_REGIMES: List[str] = [
    "Short / Tactical",
    "Medium / Campaign",
    "Protracted / War"
]

# Regional historical observation distribution in 2015-2025 dataset
REGIONAL_OBSERVATION_COUNTS: Dict[str, int] = {
    "Middle East / MENA": 121,
    "South Asia": 114,
    "Eastern Europe": 56,
    "North America": 16,
    "East Asia": 3
}

# Regional sparsity caveats
REGIONAL_SPARSITY_WARNINGS: Dict[str, str] = {
    "East Asia": (
        "⚠️ Severe Historical Data Sparsity: Only 3 historical conflict/escalation episodes occurred in East Asia "
        "(Taiwan Strait / Korean Peninsula military exercises) during the 2015–2025 observation window. Predictions "
        "rely heavily on cross-regional transfer learning and market-stress conditioning rather than regional sample depth."
    ),
    "North America": (
        "⚠️ Moderate Historical Data Sparsity: Only 16 historical incidents (domestic armed conflict / violent extremist episodes) "
        "are recorded for North America in the training dataset. Predictions reflect localized volatility rather than state-on-state warfare."
    )
}

BENCHMARK_NAMES_DISPLAY = [
    ("sp500", "S&P 500"),
    ("nasdaq", "NASDAQ Composite"),
    ("dow_jones", "Dow Jones"),
    ("russell2000", "Russell 2000")
]


def map_duration_to_regime(duration_days: Union[int, float]) -> str:
    """
    Maps an expected conflict duration in calendar days to one of three ex-ante operational regimes:
    - < 30 days: Short / Tactical
    - 30 to 180 days: Medium / Campaign
    - > 180 days: Protracted / War
    """
    days = float(duration_days)
    if days < 30.0:
        return "Short / Tactical"
    elif days <= 180.0:
        return "Medium / Campaign"
    else:
        return "Protracted / War"


class ConflictPredictor:
    """
    Production-grade inference engine for geopolitical conflict market impact prediction.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        metadata: Dict[str, Any],
        metrics: Dict[str, Any],
        feature_importances: List[Dict[str, Any]],
        model_dir: str
    ):
        self.pipeline = pipeline
        self.metadata = metadata
        self.metrics = metrics
        self.feature_importances = feature_importances
        self.model_dir = model_dir
        self.numeric_features = metadata.get("numeric_features", [])
        self.categorical_features = metadata.get("categorical_features", [])
        self.target_columns = metadata.get("target_columns", [])
        self.feature_medians = metadata.get("feature_medians", {})

    @classmethod
    def load(cls, model_dir: Optional[str] = None) -> "ConflictPredictor":
        """
        Loads the trained model pipeline and associated metadata from disk.
        Searches data/analytics/ml_models/ and data/processed/ml/models/.
        """
        search_dirs = [
            model_dir,
            ANALYTICS_MODEL_DIR,
            PROCESSED_MODEL_DIR
        ]
        chosen_dir = None
        for d in search_dirs:
            if d and os.path.exists(os.path.join(d, "random_forest_multitarget.joblib")):
                chosen_dir = d
                break

        if not chosen_dir:
            raise FileNotFoundError(
                f"Model artifact 'random_forest_multitarget.joblib' not found in: {search_dirs}"
            )

        pipeline_path = os.path.join(chosen_dir, "random_forest_multitarget.joblib")
        meta_path = os.path.join(chosen_dir, "model_metadata.json")
        metrics_path = os.path.join(chosen_dir, "evaluation_metrics.json")
        imp_path = os.path.join(chosen_dir, "feature_importance.json")

        pipeline = joblib.load(pipeline_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)

        feature_importances = []
        if os.path.exists(imp_path):
            with open(imp_path, "r", encoding="utf-8") as f:
                feature_importances = json.load(f)

        logger.info(f"Loaded ConflictPredictor from {chosen_dir}")
        return cls(
            pipeline=pipeline,
            metadata=metadata,
            metrics=metrics,
            feature_importances=feature_importances,
            model_dir=chosen_dir
        )

    def prepare_feature_row(
        self,
        region: str,
        expected_duration_days: float,
        fatalities: Optional[float] = None,
        incident_count: Optional[float] = None,
        recent_return_5d: Optional[float] = None,
        recent_return_20d: Optional[float] = None,
        volatility_annualized: Optional[float] = None,
        vix_level: Optional[float] = None,
        vix_5d_change_pct: Optional[float] = None,
        market_trend_sma_ratio: Optional[float] = None,
        yield_spread_10y2y: Optional[float] = None,
        treasury_yield_10y: Optional[float] = None,
        treasury_yield_2y: Optional[float] = None,
        violence_type: str = "State-based armed conflict"
    ) -> Tuple[pd.DataFrame, str]:
        """
        Validates scenario inputs, maps duration regime, imputes missing fields
        using training medians, and builds the canonical model input DataFrame.
        """
        # 1. Validate Region
        if region not in CANONICAL_REGIONS:
            raise ValueError(f"Invalid region '{region}'. Must be one of: {CANONICAL_REGIONS}")

        # 2. Map Duration
        duration_regime = map_duration_to_regime(expected_duration_days)

        # 3. Numeric values with fallbacks to training medians
        m = self.feature_medians
        best_fatalities = float(fatalities) if fatalities is not None else float(m.get("fatalities_best", 48.0))
        log_fatalities = float(np.log(1.0 + max(0.0, best_fatalities)))
        incidents = float(incident_count) if incident_count is not None else float(m.get("incident_count", 5.0))

        r5 = float(recent_return_5d) if recent_return_5d is not None else float(m.get("pre_5d_return_mkt", 0.35))
        r20 = float(recent_return_20d) if recent_return_20d is not None else float(m.get("pre_20d_return_mkt", 1.36))

        ann_vol = float(volatility_annualized) if volatility_annualized is not None else float(m.get("annualized_21d_vol_mkt", 12.14))
        # Daily rolling 21d vol from annualized (ann_vol / sqrt(252))
        daily_vol = (ann_vol / 100.0) / np.sqrt(252.0) if ann_vol > 0 else float(m.get("rolling_21d_vol_mkt", 0.0076))

        vix = float(vix_level) if vix_level is not None else float(m.get("vix_level", 16.69))
        vix_chg = float(vix_5d_change_pct) if vix_5d_change_pct is not None else float(m.get("vix_5d_change_pct", -1.56))
        sma_ratio = float(market_trend_sma_ratio) if market_trend_sma_ratio is not None else float(m.get("sma_50_ratio_mkt", 1.48))

        # Treasury Yields
        y_10y = float(treasury_yield_10y) if treasury_yield_10y is not None else float(m.get("treasury_yield_10y", 2.09))
        y_2y = float(treasury_yield_2y) if treasury_yield_2y is not None else float(m.get("treasury_yield_2y", 1.21))
        
        # If yield_spread explicitly provided, honor it and adjust 2y yield if not provided
        if yield_spread_10y2y is not None:
            spread = float(yield_spread_10y2y)
            if treasury_yield_2y is None:
                y_2y = y_10y - spread
        else:
            spread = float(m.get("yield_spread_10y2y", 0.70))

        # Construct row dict
        row_dict = {
            "fatalities_best": best_fatalities,
            "log_fatalities": log_fatalities,
            "incident_count": incidents,
            "pre_5d_return_mkt": r5,
            "pre_20d_return_mkt": r20,
            "rolling_21d_vol_mkt": daily_vol,
            "annualized_21d_vol_mkt": ann_vol,
            "sma_50_ratio_mkt": sma_ratio,
            "vix_level": vix,
            "vix_5d_change_pct": vix_chg,
            "yield_spread_10y2y": spread,
            "treasury_yield_10y": y_10y,
            "treasury_yield_2y": y_2y,
            "canonical_region": region,
            "duration_regime": duration_regime,
            "type_of_violence_desc": violence_type
        }

        feature_cols = self.numeric_features + self.categorical_features
        df_row = pd.DataFrame([row_dict])[feature_cols]
        return df_row, duration_regime

    def predict_scenario(
        self,
        region: str,
        expected_duration_days: float,
        fatalities: Optional[float] = None,
        incident_count: Optional[float] = None,
        recent_return_5d: Optional[float] = None,
        recent_return_20d: Optional[float] = None,
        volatility_annualized: Optional[float] = None,
        vix_level: Optional[float] = None,
        vix_5d_change_pct: Optional[float] = None,
        market_trend_sma_ratio: Optional[float] = None,
        yield_spread_10y2y: Optional[float] = None,
        treasury_yield_10y: Optional[float] = None,
        treasury_yield_2y: Optional[float] = None,
        violence_type: str = "State-based armed conflict"
    ) -> Dict[str, Any]:
        """
        Executes prediction on scenario and returns structured output table,
        regional diagnostics, and an econometric explanation.
        """
        df_input, mapped_regime = self.prepare_feature_row(
            region=region,
            expected_duration_days=expected_duration_days,
            fatalities=fatalities,
            incident_count=incident_count,
            recent_return_5d=recent_return_5d,
            recent_return_20d=recent_return_20d,
            volatility_annualized=volatility_annualized,
            vix_level=vix_level,
            vix_5d_change_pct=vix_5d_change_pct,
            market_trend_sma_ratio=market_trend_sma_ratio,
            yield_spread_10y2y=yield_spread_10y2y,
            treasury_yield_10y=treasury_yield_10y,
            treasury_yield_2y=treasury_yield_2y,
            violence_type=violence_type
        )

        # Raw model prediction of shape (1, 8)
        raw_pred = self.pipeline.predict(df_input)[0]

        # Extract predictions for the eight targets:
        # 0: sp500_5d, 1: sp500_20d
        # 2: nasdaq_5d, 3: nasdaq_20d
        # 4: dow_jones_5d, 5: dow_jones_20d
        # 6: russell2000_5d, 7: russell2000_20d
        display_rows = [
            {
                "Index": "S&P 500",
                "5-Day Predicted Change": f"{raw_pred[0]:+.2f}%",
                "20-Day Predicted Change": f"{raw_pred[1]:+.2f}%",
                "raw_5d": float(raw_pred[0]),
                "raw_20d": float(raw_pred[1])
            },
            {
                "Index": "NASDAQ Composite",
                "5-Day Predicted Change": f"{raw_pred[2]:+.2f}%",
                "20-Day Predicted Change": f"{raw_pred[3]:+.2f}%",
                "raw_5d": float(raw_pred[2]),
                "raw_20d": float(raw_pred[3])
            },
            {
                "Index": "Dow Jones",
                "5-Day Predicted Change": f"{raw_pred[4]:+.2f}%",
                "20-Day Predicted Change": f"{raw_pred[5]:+.2f}%",
                "raw_5d": float(raw_pred[4]),
                "raw_20d": float(raw_pred[5])
            },
            {
                "Index": "Russell 2000",
                "5-Day Predicted Change": f"{raw_pred[6]:+.2f}%",
                "20-Day Predicted Change": f"{raw_pred[7]:+.2f}%",
                "raw_5d": float(raw_pred[6]),
                "raw_20d": float(raw_pred[7])
            }
        ]

        display_df = pd.DataFrame(display_rows)[["Index", "5-Day Predicted Change", "20-Day Predicted Change"]]

        # Sparsity warning
        sparsity_warning = REGIONAL_SPARSITY_WARNINGS.get(region, None)
        obs_count = REGIONAL_OBSERVATION_COUNTS.get(region, 0)

        # Build Econometric Explanation
        explanation = self._generate_explanation(
            raw_pred=raw_pred,
            region=region,
            duration_regime=mapped_regime,
            vix=float(df_input["vix_level"].iloc[0]),
            sma_ratio=float(df_input["sma_50_ratio_mkt"].iloc[0]),
            yield_spread=float(df_input["yield_spread_10y2y"].iloc[0])
        )

        return {
            "prediction_table": display_df,
            "raw_predictions": {
                "sp500_fwd_ret_5d": float(raw_pred[0]),
                "sp500_fwd_ret_20d": float(raw_pred[1]),
                "nasdaq_fwd_ret_5d": float(raw_pred[2]),
                "nasdaq_fwd_ret_20d": float(raw_pred[3]),
                "dow_jones_fwd_ret_5d": float(raw_pred[4]),
                "dow_jones_fwd_ret_20d": float(raw_pred[5]),
                "russell2000_fwd_ret_5d": float(raw_pred[6]),
                "russell2000_fwd_ret_20d": float(raw_pred[7]),
            },
            "scenario": {
                "primary_category": "War and Armed Conflicts",
                "canonical_region": region,
                "expected_duration_days": expected_duration_days,
                "mapped_duration_regime": mapped_regime,
                "regional_sample_size": obs_count
            },
            "sparsity_warning": sparsity_warning,
            "explanation": explanation,
            "model_info": {
                "name": "Multi-Output RandomForestRegressor",
                "hyperparameters": self.metadata.get("primary_model", "RandomForestRegressor (depth=5, n=100)"),
                "train_samples": self.metadata.get("train_samples", 216),
                "test_samples": self.metadata.get("test_samples", 94)
            }
        }

    def _generate_explanation(
        self,
        raw_pred: np.ndarray,
        region: str,
        duration_regime: str,
        vix: float,
        sma_ratio: float,
        yield_spread: float
    ) -> str:
        """
        Synthesizes an econometric interpretation based on model feature importances
        (VIX level, 50D SMA trend, yield curve posture, and duration regime).
        """
        avg_5d = np.mean([raw_pred[0], raw_pred[2], raw_pred[4], raw_pred[6]])
        avg_20d = np.mean([raw_pred[1], raw_pred[3], raw_pred[5], raw_pred[7]])

        # Volatility condition
        if vix > 25.0:
            vol_comment = f"VIX is elevated at {vix:.1f}, indicating high prevailing market stress; historically, conflict shocks under high VIX exhibit rapid absorption with potential mean-reversion."
        elif vix < 15.0:
            vol_comment = f"VIX is relatively low at {vix:.1f} (complacency zone); unexpected geopolitical shocks from quiet regimes often induce sharper immediate reassessments."
        else:
            vol_comment = f"VIX is at {vix:.1f}, close to historical median market conditions."

        # Trend condition
        if sma_ratio > 2.0:
            trend_comment = f"The parent equity market was trading {sma_ratio:+.1f}% above its 50-day SMA, indicating upward underlying momentum."
        elif sma_ratio < -2.0:
            trend_comment = f"The parent equity market was trading {sma_ratio:+.1f}% below its 50-day SMA, indicating an existing downtrend prior to escalation."
        else:
            trend_comment = f"Market trend was neutral ({sma_ratio:+.1f}% relative to 50D SMA)."

        # Curve condition
        curve_comment = (
            f"The 10Y-2Y yield curve spread is {yield_spread:+.2f}%, indicating an inverted curve (recessionary conditioning)."
            if yield_spread < 0 else
            f"The 10Y-2Y yield curve spread is normal at {yield_spread:+.2f}%."
        )

        direction_5d = "slight positive drift" if avg_5d > 0.1 else ("modest negative pullback" if avg_5d < -0.1 else "near-neutral")
        direction_20d = "positive drift" if avg_20d > 0.2 else ("negative consolidation" if avg_20d < -0.2 else "subdued return")

        return (
            f"For a **{duration_regime}** conflict in **{region}**, the model predicts a {direction_5d} "
            f"across US equity benchmarks over 5 trading days (average: {avg_5d:+.2f}%), and a {direction_20d} "
            f"over 20 trading days (average: {avg_20d:+.2f}%).\n\n"
            f"**Key Drivers Identified by Random Forest**:\n"
            f"- **Implied Volatility (24.4% importance)**: {vol_comment}\n"
            f"- **Market Trend (13.7% importance)**: {trend_comment}\n"
            f"- **Monetary Conditioning (12.0% importance)**: {curve_comment}"
        )

    def get_feature_importances_df(self) -> pd.DataFrame:
        """Returns sorted feature importance table for visualization."""
        if not self.feature_importances:
            return pd.DataFrame(columns=["feature", "importance", "cumulative_importance"])
        df_imp = pd.DataFrame(self.feature_importances)
        return df_imp

    def get_evaluation_metrics_dict(self) -> Dict[str, Any]:
        """Returns full evaluation metrics dictionary."""
        return self.metrics


def load_conflict_predictor(model_dir: Optional[str] = None) -> ConflictPredictor:
    """Helper factory function to instantiate ConflictPredictor."""
    return ConflictPredictor.load(model_dir)


if __name__ == "__main__":
    predictor = load_conflict_predictor()
    print("ConflictPredictor loaded successfully.")
    result = predictor.predict_scenario(
        region="Middle East / MENA",
        expected_duration_days=45,
        fatalities=120,
        vix_level=22.0
    )
    print("\nPredicted Market Returns:")
    print(result["prediction_table"])
    print("\nExplanation:")
    print(result["explanation"])
