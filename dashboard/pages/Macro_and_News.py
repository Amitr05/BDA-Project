"""
Page 8: Macro & News
FRED macroeconomic indicators, Treasury yield curve and inversion dynamics,
monetary policy regime context, and GDELT daily news sentiment tracking.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from dashboard.app import (
    load_macro_aligned_data,
    load_macro_context_analysis,
    load_news_articles,
    _format_num,
    PROCESSED_DIR,
)
from dashboard.components.header import render_page_header
from dashboard.components.charts import apply_chart_theme, FINANCIAL_COLORS


def load_daily_news_sentiment() -> pd.DataFrame:
    """Load daily news sentiment aggregate if present."""
    p = PROCESSED_DIR / "daily_news_sentiment.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if "published_date" in df.columns:
            df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
            df = df.dropna(subset=["published_date"]).sort_values("published_date").reset_index(drop=True)
        return df
    return pd.DataFrame()


def render_macro_news_page():
    render_page_header(
        title="Macroeconomic Indicators & News Sentiment",
        subtitle="Point-in-time FRED macroeconomic series, Treasury yield curve slope & inversion, monetary regime context, and GDELT sentiment dynamics.",
        badge="MODULE 7",
        badge_color="#059669"
    )

    macro_df = load_macro_aligned_data()
    macro_context_df = load_macro_context_analysis()
    daily_sent_df = load_daily_news_sentiment()
    news_df = load_news_articles()

    if macro_df is None or macro_df.empty:
        st.warning("Aligned macroeconomic dataset missing in data/processed/macro/. Run ingestion pipeline first.")
        return

    # Date filter for Macro series
    min_d = macro_df["Date"].min().date()
    max_d = macro_df["Date"].max().date()

    fc1, fc2 = st.columns([1.5, 2.5])
    with fc1:
        date_range = st.date_input(
            "Filter Macro Date Range:",
            value=[min_d, max_d],
            min_value=min_d,
            max_value=max_d,
            key="macro_date_filter"
        )

    filt_macro = macro_df.copy()
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        s_d, e_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filt_macro = filt_macro[(filt_macro["Date"] >= s_d) & (filt_macro["Date"] <= e_d)].reset_index(drop=True)

    # 1. Macro KPI Metrics Row
    st.subheader("1. Key Macroeconomic & Monetary Indicators")

    latest_row = filt_macro.iloc[-1]
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    ff_rate = latest_row.get("fed_funds_rate_daily", np.nan)
    k1.metric("Fed Funds Rate", _format_num(ff_rate, "%"))

    t10y = latest_row.get("treasury_yield_10y", np.nan)
    k2.metric("10Y Treasury Yield", _format_num(t10y, "%"))

    t2y = latest_row.get("treasury_yield_2y", np.nan)
    k3.metric("2Y Treasury Yield", _format_num(t2y, "%"))

    curve_spread = latest_row.get("yield_curve_spread_10y2y", np.nan)
    inverted = latest_row.get("yield_curve_inverted", False)
    inv_label = "INVERTED" if inverted else "NORMAL"
    k4.metric("10Y–2Y Yield Spread", _format_num(curve_spread, "%"), delta=inv_label, delta_color="inverse" if inverted else "normal")

    cpi = latest_row.get("cpi_headline_yoy_pct", np.nan)
    k5.metric("Headline CPI (YoY)", _format_num(cpi, "%"))

    unrate = latest_row.get("unemployment_rate", np.nan)
    k6.metric("Unemployment Rate", _format_num(unrate, "%"))

    st.markdown("---")

    # 2. Treasury Yield Curve & Inversion Dynamics
    st.subheader("2. Treasury Yield Curve & Inversion Dynamics")
    st.caption("Point-in-time aligned benchmark Treasury yields (10-Year, 2-Year, 3-Month) and slope of the yield curve.")

    col_y1, col_y2 = st.columns(2)

    with col_y1:
        yield_fig = go.Figure()
        if "treasury_yield_10y" in filt_macro.columns:
            yield_fig.add_trace(go.Scatter(x=filt_macro["Date"], y=filt_macro["treasury_yield_10y"], mode="lines", name="10Y Yield", line=dict(color="#2563eb", width=2)))
        if "treasury_yield_2y" in filt_macro.columns:
            yield_fig.add_trace(go.Scatter(x=filt_macro["Date"], y=filt_macro["treasury_yield_2y"], mode="lines", name="2Y Yield", line=dict(color="#f59e0b", width=1.5)))
        if "treasury_yield_3m" in filt_macro.columns:
            yield_fig.add_trace(go.Scatter(x=filt_macro["Date"], y=filt_macro["treasury_yield_3m"], mode="lines", name="3M Yield", line=dict(color="#10b981", width=1.5, dash="dot")))
        apply_chart_theme(yield_fig, height=360, title="Benchmark US Treasury Yields (% p.a.)")
        yield_fig.update_layout(yaxis_title="Yield (%)")
        st.plotly_chart(yield_fig, use_container_width=True)

    with col_y2:
        spread_fig = go.Figure()
        if "yield_curve_spread_10y2y" in filt_macro.columns:
            spread_series = filt_macro["yield_curve_spread_10y2y"].dropna()
            spread_fig.add_trace(go.Scatter(
                x=filt_macro["Date"],
                y=filt_macro["yield_curve_spread_10y2y"],
                mode="lines",
                name="10Y - 2Y Spread",
                line=dict(color="#9333ea", width=2)
            ))
            spread_fig.add_hline(y=0.0, line_color="crimson", line_dash="dash", annotation_text="Inversion Boundary (Spread = 0%)")
        apply_chart_theme(spread_fig, height=360, title="10Y – 2Y Treasury Yield Curve Spread (%)")
        spread_fig.update_layout(yaxis_title="Spread (% points)")
        st.plotly_chart(spread_fig, use_container_width=True)

    st.markdown("---")

    # 3. Inflation & Macro Regimes
    st.subheader("3. Macroeconomic Regime Analysis & Market Impact")
    st.caption("Empirical equity performance under different macroeconomic yield curve regimes from Layer 3 analytics.")

    col_r1, col_r2 = st.columns([1.1, 0.9])

    with col_r1:
        if macro_context_df is not None and not macro_context_df.empty:
            st.write("**Macro Regime Summary:**")
            st.dataframe(
                macro_context_df.rename(columns={
                    "macro_dimension": "Dimension", "regime": "Regime",
                    "trading_sessions": "Trading Days", "avg_daily_return_pct": "Avg Daily Return %",
                    "annualized_volatility_pct": "Ann. Volatility %", "avg_vix_level": "Avg VIX Level"
                }),
                use_container_width=True,
                hide_index=True
            )

    with col_r2:
        if "cpi_headline_yoy_pct" in filt_macro.columns and "cpi_core_yoy_pct" in filt_macro.columns:
            cpi_fig = go.Figure()
            cpi_fig.add_trace(go.Scatter(x=filt_macro["Date"], y=filt_macro["cpi_headline_yoy_pct"], mode="lines", name="Headline CPI YoY", line=dict(color="#ef4444", width=2)))
            cpi_fig.add_trace(go.Scatter(x=filt_macro["Date"], y=filt_macro["cpi_core_yoy_pct"], mode="lines", name="Core CPI YoY", line=dict(color="#3b82f6", width=2, dash="dash")))
            cpi_fig.add_hline(y=2.0, line_color="green", line_dash="dot", annotation_text="Fed 2% Target")
            apply_chart_theme(cpi_fig, height=280, title="Inflation Dynamics: Headline vs. Core CPI (% YoY)")
            st.plotly_chart(cpi_fig, use_container_width=True)

    st.markdown("---")

    # 4. GDELT News Sentiment & Market Volatility Dynamics
    st.subheader("4. GDELT Global News Sentiment & Volatility Tracking")
    st.caption("Investigates the relationship between global news sentiment polarity, GDELT tone, and stock market volatility.")

    if not daily_sent_df.empty:
        col_ns1, col_ns2 = st.columns(2)

        with col_ns1:
            tone_fig = px.bar(
                daily_sent_df,
                x="published_date",
                y="avg_daily_tone",
                title="GDELT Daily Average Tone Trend",
                color="avg_daily_tone",
                color_continuous_scale="RdYlGn",
                labels={"published_date": "Publication Date", "avg_daily_tone": "GDELT Tone"}
            )
            tone_fig.add_hline(y=0, line_color="gray")
            apply_chart_theme(tone_fig, height=320)
            st.plotly_chart(tone_fig, use_container_width=True)

        with col_ns2:
            neg_fig = px.line(
                daily_sent_df,
                x="published_date",
                y="negative_ratio",
                title="Negative News Article Ratio Over Time",
                markers=True,
                labels={"published_date": "Date", "negative_ratio": "Negative Ratio"}
            )
            apply_chart_theme(neg_fig, height=320)
            st.plotly_chart(neg_fig, use_container_width=True)

    if not news_df.empty:
        st.write("**GDELT Standardized News Dataset:**")
        n_disp = [c for c in ["published_date", "title", "domain", "sourcecountry", "sentiment_label", "gdelt_tone", "polarity", "timing_phase"] if c in news_df.columns]
        st.dataframe(news_df[n_disp].head(25), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render_macro_news_page()
