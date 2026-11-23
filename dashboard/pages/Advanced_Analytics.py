"""
Page 7: Advanced Analytics
Cross-company event studies, intra-sector return dispersion, equity resilience analysis,
event category comparisons, and econometric hypothesis testing.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.app import (
    load_company_event_studies,
    load_cross_company_dispersion,
    load_category_comparisons,
    load_significance_summary,
    load_events_database,
    _format_num,
)
from dashboard.components.header import render_page_header
from dashboard.components.charts import apply_chart_theme, FINANCIAL_COLORS


def render_advanced_analytics_page():
    render_page_header(
        title="Advanced Analytics & Econometric Cross-Sections",
        subtitle="350-equity cross-sectional studies, intra-sector return dispersion ranges, resilience rankings, category comparative dynamics, and hypothesis testing.",
        badge="MODULE 6",
        badge_color="#4f46e5"
    )

    comp_events_df = load_company_event_studies()
    disp_df = load_cross_company_dispersion()
    cat_comp_df = load_category_comparisons()
    sig_df = load_significance_summary()

    if comp_events_df is None or comp_events_df.empty:
        st.warning("Layer 3 analytics datasets missing in data/analytics/. Run python main.py first.")
        return

    # 1. Cross-Company Event Studies (350 Constituents)
    st.subheader("1. Cross-Company Event Studies & Beta Distributions")
    st.caption("Micro-level equity response data across S&P 500, NASDAQ, Dow Jones, and Russell 2000 constituents.")

    c1, c2 = st.columns([1.5, 1.5])
    with c1:
        avail_events = ["All Events"] + sorted(comp_events_df["event_name"].dropna().unique().tolist())
        sel_ev = st.selectbox("Filter by Historical Event:", options=avail_events, index=0)

    with c2:
        avail_sectors = ["All Sectors"] + sorted(comp_events_df["sector"].dropna().unique().tolist())
        sel_sec = st.selectbox("Filter by Sector:", options=avail_sectors, index=0)

    filtered_comp = comp_events_df.copy()
    if sel_ev != "All Events":
        filtered_comp = filtered_comp[filtered_comp["event_name"] == sel_ev]
    if sel_sec != "All Sectors":
        filtered_comp = filtered_comp[filtered_comp["sector"] == sel_sec]

    # Metrics row for cross-company sample
    cm1, cm2, cm3, cm4, cm5 = st.columns(5)
    cm1.metric("Analyzed Companies", len(filtered_comp))
    cm2.metric("Mean Post-5D Return", _format_num(filtered_comp["post_5d_return_pct"].mean(), "%"))
    cm3.metric("Mean Post-30D Return", _format_num(filtered_comp["post_30d_return_pct"].mean(), "%"))
    cm4.metric("Mean CAR (+5D)", _format_num(filtered_comp["car_post_5d_pct"].mean(), "%"))
    cm5.metric("Average Market Beta", _format_num(filtered_comp["estimated_beta"].mean()))

    # Company response table
    st.write(f"**Constituent Equities Response Table ({len(filtered_comp)} records):**")
    disp_cols = [
        "ticker", "company_name", "sector", "event_name", "price_t0",
        "pre_5d_return_pct", "post_5d_return_pct", "post_30d_return_pct",
        "spread_vs_sp500_5d_pct", "estimated_beta", "car_post_5d_pct"
    ]
    st.dataframe(
        filtered_comp[[c for c in disp_cols if c in filtered_comp.columns]].rename(columns={
            "ticker": "Ticker", "company_name": "Company", "sector": "Sector", "event_name": "Event",
            "price_t0": "Close T0", "pre_5d_return_pct": "Pre-5D %", "post_5d_return_pct": "Post-5D %",
            "post_30d_return_pct": "Post-30D %", "spread_vs_sp500_5d_pct": "Spread vs SPX %",
            "estimated_beta": "Beta", "car_post_5d_pct": "CAR +5D %"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Beta vs Post-5D Spread Scatter Plot
    if len(filtered_comp) >= 5:
        scatter_fig = px.scatter(
            filtered_comp,
            x="estimated_beta",
            y="post_5d_return_pct",
            color="sector",
            hover_name="company_name",
            hover_data=["ticker", "post_30d_return_pct"],
            title="Equity Post-5D Return vs. Market Beta (Colored by Sector)",
            labels={"estimated_beta": "Estimated Market Model Beta", "post_5d_return_pct": "Post-5D Return (%)"}
        )
        scatter_fig.add_hline(y=0, line_color="gray", line_dash="dash")
        scatter_fig.add_vline(x=1.0, line_color="gray", line_dash="dash")
        apply_chart_theme(scatter_fig, height=380)
        st.plotly_chart(scatter_fig, use_container_width=True)

    st.markdown("---")

    # 2. Intra-Sector Market Dispersion & Resilience Analysis
    st.subheader("2. Intra-Sector Market Return Dispersion & Resilience Analysis")
    st.caption("Quantifies the spread between sector winners and losers around major historical event shocks.")

    if disp_df is not None and not disp_df.empty:
        col_d1, col_d2 = st.columns([1.2, 0.8])

        with col_d1:
            disp_events = disp_df["event_name"].dropna().unique().tolist()
            sel_disp_ev = st.selectbox("Select Event for Dispersion Breakdown:", options=disp_events, index=0)

            disp_sub = disp_df[disp_df["event_name"] == sel_disp_ev].sort_values("return_range_pct", ascending=True)
            if not disp_sub.empty:
                d_bar = px.bar(
                    disp_sub,
                    x="return_range_pct",
                    y="sector",
                    orientation="h",
                    title=f"Intra-Sector Return Range % (Max - Min Return) - {sel_disp_ev}",
                    color="return_range_pct",
                    color_continuous_scale="Purples",
                    labels={"return_range_pct": "Return Range Spread (%)", "sector": "Market Sector"}
                )
                apply_chart_theme(d_bar, height=360)
                st.plotly_chart(d_bar, use_container_width=True)

        with col_d2:
            st.write(f"**Sector Resilience Rankings ({sel_disp_ev})**")
            resilience_cols = ["sector", "top_resilient_ticker", "top_resilient_return_pct", "most_vulnerable_ticker", "most_vulnerable_return_pct"]
            if all(c in disp_sub.columns for c in resilience_cols):
                st.dataframe(
                    disp_sub[resilience_cols].rename(columns={
                        "sector": "Sector",
                        "top_resilient_ticker": "Top Resilient",
                        "top_resilient_return_pct": "Top Return %",
                        "most_vulnerable_ticker": "Most Vulnerable",
                        "most_vulnerable_return_pct": "Worst Return %"
                    }),
                    use_container_width=True,
                    hide_index=True
                )

    st.markdown("---")

    # 3. Event Category Comparative Dynamics
    st.subheader("3. Cross-Category Comparative Macro Statistics")
    st.caption("Macroeconomic and market-level return comparisons across standard event taxonomy categories.")

    if cat_comp_df is not None and not cat_comp_df.empty:
        col_cat1, col_cat2 = st.columns([1.1, 0.9])

        with col_cat1:
            st.write("**Event Category Performance Summary:**")
            st.dataframe(
                cat_comp_df.rename(columns={
                    "event_category": "Category", "event_count": "Event Count",
                    "avg_pre_5d_return_pct": "Avg Pre-5D %", "avg_post_5d_return_pct": "Avg Post-5D %",
                    "avg_post_30d_return_pct": "Avg Post-30D %", "avg_max_drawdown_30d_pct": "Avg 30D Drawdown %",
                    "vix_at_event_mean": "Mean VIX at T0"
                }),
                use_container_width=True,
                hide_index=True
            )

        with col_cat2:
            cat_fig = px.bar(
                cat_comp_df,
                x="event_category",
                y="avg_max_drawdown_30d_pct",
                title="Average 30-Day Maximum Drawdown Across Categories",
                color="avg_max_drawdown_30d_pct",
                color_continuous_scale="Reds_r",
                labels={"avg_max_drawdown_30d_pct": "Average 30D Drawdown (%)", "event_category": "Category"}
            )
            apply_chart_theme(cat_fig, height=300)
            st.plotly_chart(cat_fig, use_container_width=True)

    st.markdown("---")

    # 4. Econometric Statistical Significance & Hypothesis Testing
    st.subheader("4. Econometric Statistical Significance & Hypothesis Testing")
    st.caption("Parametric two-tailed cross-sectional t-tests evaluating H0: Mean(CAR) = 0 across sample scopes.")

    if sig_df is not None and not sig_df.empty:
        st.dataframe(
            sig_df[[
                "scope", "metric", "sample_size", "mean_car_pct", "standard_error",
                "t_statistic", "p_value", "statistically_significant_5pct", "interpretation"
            ]].rename(columns={
                "scope": "Sample Scope", "metric": "Metric", "sample_size": "N",
                "mean_car_pct": "Mean CAR %", "standard_error": "Std Error",
                "t_statistic": "t-stat", "p_value": "p-value",
                "statistically_significant_5pct": "Significant at 5%?", "interpretation": "Econometric Interpretation"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "📌 **Methodological Context**: Statistical significance tests evaluate whether observed Cumulative Abnormal Returns "
            "(CAR) significantly deviate from zero under the null hypothesis of market model efficiency. Statistical association "
            "does not establish deterministic causal attribution."
        )


if __name__ == "__main__":
    render_advanced_analytics_page()
