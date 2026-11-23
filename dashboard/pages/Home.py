"""
Page 1: Home / Overview
Landing page for the Big Data Analytics Financial Intelligence Platform.
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.app import (
    INDEX_CONFIG,
    load_index_data,
    load_vix_data,
    load_events_database,
    load_macro_aligned_data,
    load_news_articles,
    _format_num,
)
from dashboard.components.header import render_page_header
from dashboard.components.charts import apply_chart_theme, FINANCIAL_COLORS


def render_home_page():
    st.title("Impact of Global Events on the US Stock Market")
    st.caption("Institutional Financial Intelligence Platform · Big Data Analytics & Econometric Engine")

    render_page_header(
        title="Executive Overview & Market Pulse",
        subtitle="Multi-Asset Preprocessing, Econometric Event Studies, Macro Regime Alignment & Supervised Geopolitical ML Forecasting",
        badge="HOME / OVERVIEW",
        badge_color="#2563eb"
    )

    # Load core baseline datasets
    index_frames = {k: load_index_data(k) for k in INDEX_CONFIG}
    vix_df = load_vix_data()
    events_df = load_events_database()
    macro_df = load_macro_aligned_data()
    news_df = load_news_articles()

    # 1. Key Financial Metrics Bar across 4 US Benchmarks + VIX
    st.subheader("1. US Benchmark Indices & Volatility Monitor")

    col1, col2, col3, col4, col5 = st.columns(5)

    sp = index_frames.get("sp500")
    if sp is not None and not sp.empty:
        p_latest = sp.iloc[-1]["Close"]
        p_chg = sp.iloc[-1].get("daily_return_pct", 0.0)
        col1.metric("S&P 500 (^GSPC)", f"${p_latest:,.2f}", f"{p_chg:+.2f}% (1D)")

    nasdaq = index_frames.get("nasdaq")
    if nasdaq is not None and not nasdaq.empty:
        p_latest = nasdaq.iloc[-1]["Close"]
        p_chg = nasdaq.iloc[-1].get("daily_return_pct", 0.0)
        col2.metric("NASDAQ (^IXIC)", f"${p_latest:,.2f}", f"{p_chg:+.2f}% (1D)")

    dow = index_frames.get("dow_jones")
    if dow is not None and not dow.empty:
        p_latest = dow.iloc[-1]["Close"]
        p_chg = dow.iloc[-1].get("daily_return_pct", 0.0)
        col3.metric("Dow Jones (^DJI)", f"${p_latest:,.2f}", f"{p_chg:+.2f}% (1D)")

    rut = index_frames.get("russell2000")
    if rut is not None and not rut.empty:
        p_latest = rut.iloc[-1]["Close"]
        p_chg = rut.iloc[-1].get("daily_return_pct", 0.0)
        col4.metric("Russell 2000 (^RUT)", f"${p_latest:,.2f}", f"{p_chg:+.2f}% (1D)")

    if vix_df is not None and not vix_df.empty:
        v_latest = vix_df.iloc[-1]["Close"]
        v_diff = (vix_df.iloc[-1]["Close"] - vix_df.iloc[-2]["Close"]) if len(vix_df) > 1 else 0.0
        regime = "Calm (<20)" if v_latest < 20 else ("Elevated (20-30)" if v_latest <= 30 else "Stress (>30)")
        col5.metric("CBOE VIX Volatility", f"{v_latest:.2f}", f"{v_diff:+.2f} ({regime})", delta_color="inverse")

    st.markdown("---")

    # 2. Normalized Multi-Asset Benchmark Comparison (1-Year Context)
    st.subheader("2. Multi-Asset Comparative Performance (Trailing 1-Year)")
    st.caption("Normalized performance rebased to 100 for all 4 major US equity benchmarks alongside CBOE VIX movements.")

    if sp is not None and not sp.empty:
        max_d = sp["Date"].max()
        one_year_ago = max_d - pd.DateOffset(years=1)

        bench_fig = make_subplots(specs=[[{"secondary_y": True}]])
        colors_map = {
            "sp500": FINANCIAL_COLORS["primary"],
            "nasdaq": FINANCIAL_COLORS["secondary"],
            "dow_jones": FINANCIAL_COLORS["success"],
            "russell2000": FINANCIAL_COLORS["accent"]
        }

        for k, cfg in INDEX_CONFIG.items():
            df_k = index_frames.get(k)
            if df_k is not None and not df_k.empty:
                sub_df = df_k[df_k["Date"] >= one_year_ago].copy().reset_index(drop=True)
                if not sub_df.empty:
                    base = sub_df.iloc[0]["Close"]
                    if base > 0:
                        sub_df["rebased"] = (sub_df["Close"] / base) * 100.0
                        bench_fig.add_trace(go.Scatter(
                            x=sub_df["Date"],
                            y=sub_df["rebased"],
                            mode="lines",
                            name=f"{cfg['label']} ({cfg['ticker']})",
                            line=dict(color=colors_map.get(k, "#ffffff"), width=2)
                        ), secondary_y=False)

        if vix_df is not None and not vix_df.empty:
            vix_sub = vix_df[vix_df["Date"] >= one_year_ago].copy().reset_index(drop=True)
            if not vix_sub.empty:
                bench_fig.add_trace(go.Scatter(
                    x=vix_sub["Date"],
                    y=vix_sub["Close"],
                    mode="lines",
                    name="CBOE VIX Level",
                    line=dict(color=FINANCIAL_COLORS["vix"], width=1.5, dash="dot")
                ), secondary_y=True)

        bench_fig.add_hline(y=100, line_dash="dash", line_color="gray", secondary_y=False)
        bench_fig.update_yaxes(title_text="Normalized Index Level (Base=100)", secondary_y=False)
        bench_fig.update_yaxes(title_text="VIX Level", secondary_y=True)
        apply_chart_theme(bench_fig, height=390, title="Trailing 1-Year US Equity Benchmarks vs. CBOE VIX")
        st.plotly_chart(bench_fig, use_container_width=True)

    st.markdown("---")

    # 3. Recent Global Events Feed & Filter
    st.subheader("3. Curated Global Events Database & Intelligence Feed")
    st.caption("Historical database of geopolitical conflicts, macroeconomic decisions, elections, and crises.")

    if not events_df.empty:
        c_filter1, c_filter2 = st.columns([1.5, 2.5])
        with c_filter1:
            all_cats = ["All Categories"] + sorted(events_df["category"].dropna().unique().tolist() if "category" in events_df.columns else [])
            sel_cat = st.selectbox(
                "Filter Events Feed by Category:",
                options=all_cats,
                index=0,
                key="home_cat_filter"
            )

        with c_filter2:
            st.caption(f"Total Events Catalogued: **{len(events_df)}** across **{len(all_cats)-1}** distinct institutional taxonomy domains.")

        filtered_feed = events_df.copy()
        if sel_cat != "All Categories" and "category" in filtered_feed.columns:
            filtered_feed = filtered_feed[filtered_feed["category"] == sel_cat]

        disp_cols = [c for c in ["event_name", "category", "event_date", "region", "severity_score", "related_companies", "related_sectors"] if c in filtered_feed.columns]
        st.dataframe(
            filtered_feed[disp_cols].sort_values("event_date", ascending=False).head(15),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    # 4. Navigation Gateway to Platform Analytical Modules
    st.subheader("4. Analytical Modules & Research Workflows")
    st.caption("Select a dedicated page from the sidebar to launch specialized analytics.")

    card1, card2, card3, card4 = st.columns(4)

    with card1:
        st.markdown("""
        **📈 Market Explorer**
        - Benchmark indices & individual equities
        - OHLCV price action, returns & rolling vol
        - CBOE VIX volatility regimes
        - Relative performance spreads vs parent index
        """)

    with card2:
        st.markdown("""
        **🌍 Global Event Explorer**
        - Curated 7-category global event registry
        - Geographic, date & keyword filters
        - Affected corporate entities & sectors
        - GDELT news sentiment dynamics
        """)

    with card3:
        st.markdown("""
        **💥 Event Impact Analyzer**
        - Econometric event-study engine
        - Market Model AR & CAR trajectory
        - Statistical significance tests (p-values)
        - Multi-event comparative overlay [-5, +30]
        """)

    with card4:
        st.markdown("""
        **🏢 Sector Analysis**
        - 8 Select SPDR Sector ETFs
        - Normalized cumulative performance
        - Return correlation matrix heatmap
        - Intra-sector peer company comparisons
        """)

    card5, card6, card7, card8 = st.columns(4)

    with card5:
        st.markdown("""
        **🤖 ML Predictions**
        - Supervised multi-target forecasting
        - War & armed conflict scenario simulation
        - Random Forest vs Ridge regression baselines
        - Gini feature importance & diagnostics
        """)

    with card6:
        st.markdown("""
        **🔬 Advanced Analytics**
        - 350-equity cross-sectional response table
        - Intra-sector return dispersion ranges
        - Cross-category comparative macro statistics
        - Hypothesis testing & resilience analysis
        """)

    with card7:
        st.markdown("""
        **📊 Macro & News**
        - FRED point-in-time macroeconomic dataset
        - Treasury yield curve & inversion monitor
        - Monetary regime context analysis
        - GDELT daily sentiment & tone dynamics
        """)

    with card8:
        st.markdown("""
        **⚙️ Data Architecture**
        - **Layer 1:** Immutable Raw Ingestion (`data/raw/`)
        - **Layer 2:** Distributed Parquet (`data/processed/`)
        - **Layer 3:** Econometric Analytics (`data/analytics/`)
        - Zero look-ahead bias with point-in-time alignment
        """)


if __name__ == "__main__":
    render_home_page()
