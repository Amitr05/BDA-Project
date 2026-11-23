"""
Page 6: ML Predictions
Supervised Machine Learning Forecasting: Predicting 5-Day and 20-Day US Equity Benchmark Forward Returns
Under War & Armed Conflict Scenarios (Trained on UCDP GED v26.1 & Point-in-Time Market Regimes).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ml.predictor import (
    load_conflict_predictor,
    ConflictPredictor,
    CANONICAL_REGIONS,
    REGIONAL_OBSERVATION_COUNTS,
    REGIONAL_SPARSITY_WARNINGS,
    map_duration_to_regime,
)
from dashboard.components.header import render_page_header
from dashboard.components.charts import apply_chart_theme, FINANCIAL_COLORS


def render_ml_predictions_page():
    render_page_header(
        title="AI/ML Geopolitical Conflict Impact Predictor",
        subtitle="Supervised multi-target forecasting: 5-Day and 20-Day forward returns across 4 US equity benchmarks trained on UCDP GED v26.1 & macro regimes.",
        badge="MODULE 4",
        badge_color="#dc2626"
    )

    # Domain Scope Warning Banner
    st.warning(
        "⚠️ **Specialized Domain Notice**: This machine learning model is trained exclusively on historical armed conflict "
        "and war episodes from the Uppsala Conflict Data Program (UCDP GED v26.1). It is **not** a general-purpose model "
        "and cannot predict responses to monetary policy, corporate earnings, elections, or natural disasters. "
        "All predictions are experimental academic research estimates and **do not constitute financial or investment advice**."
    )

    try:
        predictor = load_conflict_predictor()
    except Exception as e:
        st.error(f"Unable to load Machine Learning Predictor: {e}")
        st.info("Ensure ML models have been trained and serialized to data/analytics/ml_models/.")
        return

    st.markdown("---")
    col_input, col_results = st.columns([1, 1.25], gap="large")

    with col_input:
        st.subheader("1. Scenario Definition")

        # Primary Event Category (Fixed)
        st.text_input(
            "Primary Event Category (Fixed Scope):",
            value="War and Armed Conflicts",
            disabled=True,
            help="Fixed ML domain: War and Armed Conflicts derived from UCDP GED v26.1."
        )

        # Canonical Region (5 choices)
        selected_region = st.selectbox(
            "Geographical Subcategory (Region):",
            options=CANONICAL_REGIONS,
            index=0,
            help="Select one of the five canonical geopolitical subcategories."
        )

        # Sparsity notification right below dropdown
        if selected_region in REGIONAL_SPARSITY_WARNINGS:
            st.warning(REGIONAL_SPARSITY_WARNINGS[selected_region])
        else:
            obs = REGIONAL_OBSERVATION_COUNTS.get(selected_region, 0)
            st.caption(f"Historical Sample Depth: **{obs} independent episodes** recorded in {selected_region}.")

        # Expected Duration & Regime Mapping
        col_dur1, col_dur2 = st.columns([1.2, 1])
        with col_dur1:
            duration_days = st.number_input(
                "Expected Conflict Duration (Days):",
                min_value=1,
                max_value=730,
                value=30,
                step=5,
                help="Input expected duration in calendar days. Automatically mapped to operational regime."
            )
        with col_dur2:
            mapped_regime = map_duration_to_regime(duration_days)
            st.text_input(
                "Operational Regime:",
                value=mapped_regime,
                disabled=True,
                help="<30d: Short / Tactical | 30-180d: Medium / Campaign | >180d: Protracted / War"
            )

        # Conflict Intensity & Violence Type
        col_int1, col_int2 = st.columns(2)
        with col_int1:
            fatalities = st.number_input(
                "Expected Casualties / Fatalities:",
                min_value=0,
                max_value=25000,
                value=50,
                step=10,
                help="Best estimate of total battle and non-combatant fatalities in the episode."
            )
        with col_int2:
            violence_type = st.selectbox(
                "Violence Typology:",
                options=[
                    "State-based armed conflict",
                    "Non-state armed conflict",
                    "One-sided violence"
                ],
                index=0
            )

        st.subheader("2. Prevailing Market & Macro Conditions")
        st.caption("Conditioning features evaluated at event day T0 (historical medians pre-filled):")

        with st.expander("Adjust Financial & Macro Indicators (Optional)", expanded=True):
            col_mkt1, col_mkt2 = st.columns(2)
            with col_mkt1:
                vix_level = st.slider("Spot VIX Level (T0):", min_value=10.0, max_value=65.0, value=16.7, step=0.5)
                vix_5d_chg = st.number_input("5-Day VIX Change (%):", value=-1.5, step=1.0)
                mkt_sma_ratio = st.number_input("Market vs. 50D SMA (%):", value=1.5, step=0.5, help="Trend posture relative to 50-day moving average")
            with col_mkt2:
                yield_spread = st.number_input("Treasury 10Y–2Y Spread (%):", value=0.70, step=0.1, help="Yield curve slope: <0% denotes inversion")
                treasury_10y = st.number_input("10-Year Benchmark Yield (%):", value=2.10, step=0.1)
                pre_ret_5d = st.number_input("Pre-Event 5D Return (%):", value=0.35, step=0.5)
                pre_ret_20d = st.number_input("Pre-Event 20D Return (%):", value=1.36, step=0.5)

        run_predict = st.button("🔮 Generate Market Impact Forecast", type="primary", use_container_width=True)

    with col_results:
        st.subheader("3. Model Return Predictions")

        # Execute prediction using actual trained model
        result = predictor.predict_scenario(
            region=selected_region,
            expected_duration_days=duration_days,
            fatalities=fatalities,
            recent_return_5d=pre_ret_5d,
            recent_return_20d=pre_ret_20d,
            vix_level=vix_level,
            vix_5d_change_pct=vix_5d_chg,
            market_trend_sma_ratio=mkt_sma_ratio,
            yield_spread_10y2y=yield_spread,
            treasury_yield_10y=treasury_10y,
            violence_type=violence_type
        )

        table_df = result["prediction_table"]

        # Display Styled Prediction Table in Exact Format Required
        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True
        )

        # Plotly Comparison Bar Chart Across the 4 Indices and 2 Horizons
        raw_dict = result["raw_predictions"]
        chart_data = pd.DataFrame([
            {"Index": "S&P 500", "Horizon": "5-Day Return", "Return (%)": raw_dict["sp500_fwd_ret_5d"]},
            {"Index": "S&P 500", "Horizon": "20-Day Return", "Return (%)": raw_dict["sp500_fwd_ret_20d"]},
            {"Index": "NASDAQ", "Horizon": "5-Day Return", "Return (%)": raw_dict["nasdaq_fwd_ret_5d"]},
            {"Index": "NASDAQ", "Horizon": "20-Day Return", "Return (%)": raw_dict["nasdaq_fwd_ret_20d"]},
            {"Index": "Dow Jones", "Horizon": "5-Day Return", "Return (%)": raw_dict["dow_jones_fwd_ret_5d"]},
            {"Index": "Dow Jones", "Horizon": "20-Day Return", "Return (%)": raw_dict["dow_jones_fwd_ret_20d"]},
            {"Index": "Russell 2000", "Horizon": "5-Day Return", "Return (%)": raw_dict["russell2000_fwd_ret_5d"]},
            {"Index": "Russell 2000", "Horizon": "20-Day Return", "Return (%)": raw_dict["russell2000_fwd_ret_20d"]},
        ])

        fig_pred = px.bar(
            chart_data,
            x="Index",
            y="Return (%)",
            color="Horizon",
            barmode="group",
            text_auto=".2f",
            title=f"Predicted Return Impact Across Four US Benchmarks ({selected_region})",
            color_discrete_map={"5-Day Return": FINANCIAL_COLORS["primary"], "20-Day Return": FINANCIAL_COLORS["secondary"]}
        )
        fig_pred.add_hline(y=0, line_color="gray", line_dash="dash")
        apply_chart_theme(fig_pred, height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_pred, use_container_width=True)

        # Econometric Interpretation Card
        st.subheader("4. Econometric Explanation & Key Drivers")
        st.markdown(result["explanation"])

        # Academic Disclaimer
        st.info(
            "ℹ️ **Experimental Statistical Estimate**: Predictions are generated by a multi-output "
            "Random Forest trained on 2015–2022 conflict data. This output is an econometric research estimate, "
            "not a guaranteed market forecast, trading recommendation, or causal conclusion."
        )

    # Technical Diagnostics & Feature Importance Tabs
    st.markdown("---")
    st.subheader("5. Model Architecture & Empirical Validation Diagnostics")

    diag_tab1, diag_tab2, diag_tab3 = st.tabs([
        "📊 Feature Importance (Random Forest)",
        "🎯 Out-of-Sample Evaluation Scorecard (2023–2025)",
        "⚙️ Model Parameters & Metadata"
    ])

    with diag_tab1:
        st.write("**Top Predictive Drivers (Gini Impurity Reduction Across All 8 Targets):**")
        imp_df = predictor.get_feature_importances_df()
        if not imp_df.empty:
            fig_imp = px.bar(
                imp_df.head(10).sort_values("importance", ascending=True),
                x="importance",
                y="feature",
                orientation="h",
                title="Top 10 Feature Importances",
                text=imp_df.head(10).sort_values("importance", ascending=True)["importance"].apply(lambda x: f"{x*100:.2f}%"),
                color="importance",
                color_continuous_scale="Viridis"
            )
            apply_chart_theme(fig_imp, height=380, margin=dict(l=20, r=20, t=40, b=20))
            fig_imp.update_layout(xaxis_title="Relative Importance Weight")
            st.plotly_chart(fig_imp, use_container_width=True)
            st.caption("Notice: Financial conditioning features (VIX level, SMA ratio, and Treasury yields) account for >73% of predictive power.")

    with diag_tab2:
        st.write("**Out-of-Sample Performance Comparison (2023–2025 Test Partition, 94 Episodes):**")
        metrics_dict = predictor.get_evaluation_metrics_dict()
        if metrics_dict and "models" in metrics_dict:
            rf_targets = metrics_dict["models"]["random_forest"]["out_of_sample_test_metrics"]["targets"]
            ridge_targets = metrics_dict["models"]["ridge_regression"]["out_of_sample_test_metrics"]["targets"]

            rows = []
            for t_col, display_label in [
                ("sp500_fwd_ret_5d", "S&P 500 (5-Day)"),
                ("sp500_fwd_ret_20d", "S&P 500 (20-Day)"),
                ("nasdaq_fwd_ret_5d", "NASDAQ Composite (5-Day)"),
                ("nasdaq_fwd_ret_20d", "NASDAQ Composite (20-Day)"),
                ("dow_jones_fwd_ret_5d", "Dow Jones (5-Day)"),
                ("dow_jones_fwd_ret_20d", "Dow Jones (20-Day)"),
                ("russell2000_fwd_ret_5d", "Russell 2000 (5-Day)"),
                ("russell2000_fwd_ret_20d", "Russell 2000 (20-Day)"),
            ]:
                rows.append({
                    "Target Benchmark": display_label,
                    "Random Forest MAE": f"{rf_targets[t_col]['mae']:.3f}%",
                    "Ridge Baseline MAE": f"{ridge_targets[t_col]['mae']:.3f}%",
                    "Random Forest RMSE": f"{rf_targets[t_col]['rmse']:.3f}%",
                    "Ridge Baseline RMSE": f"{ridge_targets[t_col]['rmse']:.3f}%",
                    "Random Forest R²": f"{rf_targets[t_col]['r2']:+.3f}",
                    "Ridge Baseline R²": f"{ridge_targets[t_col]['r2']:+.3f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with diag_tab3:
        meta = predictor.metadata
        st.write("**Model Architecture Specifications:**")
        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric("Primary Algorithm", "RandomForestRegressor")
        col_p1.caption("Depth=5, Trees=100, Min Split=5")
        col_p2.metric("Training Samples", f"{meta.get('train_samples', 216)} (2015–2022)")
        col_p2.caption("30 Unique Conflict IDs")
        col_p3.metric("Test Samples", f"{meta.get('test_samples', 94)} (2023–2025)")
        col_p3.caption("19 Unique Conflict IDs")


if __name__ == "__main__":
    render_ml_predictions_page()
