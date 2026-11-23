"""
Page 2: Market Explorer
Interactive exploration of benchmark indices, constituent equities, return distributions,
volatility regimes, and relative performance spreads.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.app import (
    INDEX_CONFIG,
    load_index_data,
    load_vix_data,
    load_stock_data,
    get_available_companies_for_index,
    _format_num,
)
from dashboard.components.header import render_page_header
from dashboard.components.charts import apply_chart_theme, FINANCIAL_COLORS


def render_market_explorer_page():
    render_page_header(
        title="Market Explorer & Equities Analysis",
        subtitle="Benchmark equity indices, constituent equities, return distributions, rolling volatility, and CBOE VIX movements.",
        badge="MODULE 1",
        badge_color="#0284c7"
    )

    index_frames = {k: load_index_data(k) for k in INDEX_CONFIG}
    vix_df = load_vix_data()

    missing_core = [k for k, df in index_frames.items() if df is None]
    if missing_core or vix_df is None:
        st.warning("Processed market datasets are missing. Run the ingestion pipeline first.")
        return

    # 1. Controls: Select Index, Select Company, Date Range
    ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.8, 1.2])

    with ctrl1:
        sel_idx_key = st.selectbox(
            "1. Select Market Index:",
            options=list(INDEX_CONFIG.keys()),
            format_func=lambda k: f"{INDEX_CONFIG[k]['label']} ({INDEX_CONFIG[k]['ticker']})",
            key="m1_index_select"
        )
        idx_cfg = INDEX_CONFIG[sel_idx_key]
        idx_df = index_frames[sel_idx_key].copy()

    with ctrl2:
        avail_companies = get_available_companies_for_index(sel_idx_key)
        comp_options = ["(Index Only - No Company Selected)"]
        if not avail_companies.empty:
            comp_options += [
                f"{r['ticker']} - {r['company_name']} ({r.get('sector', 'Unknown')})"
                for _, r in avail_companies.iterrows()
            ]

        sel_comp_str = st.selectbox(
            "2. Select Individual Company:",
            options=comp_options,
            key="m1_company_select"
        )

    sel_ticker = None
    comp_df = None
    comp_meta = {}

    if sel_comp_str != "(Index Only - No Company Selected)":
        sel_ticker = sel_comp_str.split(" - ")[0].strip()
        comp_df = load_stock_data(sel_ticker)
        if not avail_companies.empty:
            match_meta = avail_companies[avail_companies["ticker"] == sel_ticker]
            if not match_meta.empty:
                comp_meta = match_meta.iloc[0].to_dict()

    with ctrl3:
        # Determine global date bounds
        base_series = comp_df if comp_df is not None else idx_df
        min_date = base_series["Date"].min().date()
        max_date = base_series["Date"].max().date()

        date_range = st.date_input(
            "Filter Date Range:",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date,
            key="m1_date_filter"
        )

    # Filter by date range
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        idx_df = idx_df[(idx_df["Date"] >= start_d) & (idx_df["Date"] <= end_d)].reset_index(drop=True)
        if comp_df is not None:
            comp_df = comp_df[(comp_df["Date"] >= start_d) & (comp_df["Date"] <= end_d)].reset_index(drop=True)

    # 2. Key Metrics Row
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

    if comp_df is not None and not comp_df.empty:
        c_p0 = comp_df.iloc[0]["Close"]
        c_p1 = comp_df.iloc[-1]["Close"]
        c_ret = ((c_p1 - c_p0) / c_p0) * 100.0 if c_p0 > 0 else 0.0
        c_vol = comp_df["rolling_vol_21d"].iloc[-1] if "rolling_vol_21d" in comp_df.columns else np.nan

        idx_p0 = idx_df.iloc[0]["Close"] if not idx_df.empty else 1.0
        idx_p1 = idx_df.iloc[-1]["Close"] if not idx_df.empty else 1.0
        idx_ret = ((idx_p1 - idx_p0) / idx_p0) * 100.0 if idx_p0 > 0 else 0.0
        spread = c_ret - idx_ret

        m_col1.metric("Stock Latest Close", f"${c_p1:,.2f}", f"{comp_df['daily_return_pct'].iloc[-1]:+.2f}% (1D)")
        m_col2.metric("Company Total Return", f"{c_ret:,.2f}%")
        m_col3.metric("21D Volatility (Ann.)", f"{c_vol:,.2f}%" if pd.notna(c_vol) else "N/A")
        m_col4.metric(f"{idx_cfg['label']} Return", f"{idx_ret:,.2f}%")
        m_col5.metric("Relative Spread (Alpha)", f"{spread:+,.2f}%", delta_color="normal")
    else:
        i_p0 = idx_df.iloc[0]["Close"] if not idx_df.empty else 1.0
        i_p1 = idx_df.iloc[-1]["Close"] if not idx_df.empty else 1.0
        i_ret = ((i_p1 - i_p0) / i_p0) * 100.0 if i_p0 > 0 else 0.0
        i_vol = idx_df["rolling_volatility_21d"].iloc[-1] if "rolling_volatility_21d" in idx_df.columns else np.nan
        latest_vix = vix_df.iloc[-1]["Close"] if not vix_df.empty else np.nan

        m_col1.metric(f"{idx_cfg['label']} Level", f"{i_p1:,.2f}", f"{idx_df['daily_return_pct'].iloc[-1]:+.2f}% (1D)")
        m_col2.metric("Index Period Return", f"{i_ret:,.2f}%")
        m_col3.metric("21D Volatility (Ann.)", f"{i_vol:,.2f}%" if pd.notna(i_vol) else "N/A")
        m_col4.metric("CBOE VIX Level", f"{latest_vix:,.2f}" if pd.notna(latest_vix) else "N/A")
        m_col5.metric("Trading Sessions", f"{len(idx_df):,}")

    st.markdown("---")

    # 1. Historical Price Movement
    sec_title = f"{comp_meta.get('company_name', sel_ticker)} ({sel_ticker})" if sel_ticker else idx_cfg["label"]
    st.subheader(f"1. Historical Stock Price Movements: {sec_title}")

    active_plot_df = comp_df if comp_df is not None else idx_df
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(
        x=active_plot_df["Date"],
        y=active_plot_df["Close"],
        mode="lines",
        name=f"{sec_title} Close",
        line=dict(color=FINANCIAL_COLORS["primary"], width=2)
    ))
    apply_chart_theme(price_fig, height=420, title=f"{sec_title} - Historical Closing Price (USD)")
    price_fig.update_layout(xaxis_title="Date", yaxis_title="Close Price (USD)")
    st.plotly_chart(price_fig, use_container_width=True)

    # 2. Daily Returns and 3. Volatility Side by Side
    col_ret, col_vol = st.columns(2)

    with col_ret:
        st.subheader("2. Daily Returns & Distribution")
        ret_series = active_plot_df["daily_return_pct"].dropna()

        # Daily bar chart
        bar_colors = [FINANCIAL_COLORS["success"] if r >= 0 else FINANCIAL_COLORS["danger"] for r in ret_series]
        ret_fig = go.Figure()
        ret_fig.add_trace(go.Bar(
            x=active_plot_df.dropna(subset=["daily_return_pct"])["Date"],
            y=ret_series,
            marker_color=bar_colors,
            name="Daily Return %"
        ))
        ret_fig.add_hline(y=0, line_color="gray", line_width=1)
        apply_chart_theme(ret_fig, height=320, title=f"{sec_title} - Daily Percentage Returns")
        st.plotly_chart(ret_fig, use_container_width=True)

        # Return distribution histogram
        hist_fig = px.histogram(
            ret_series,
            nbins=50,
            marginal="box",
            title=f"{sec_title} - Return Distribution (Mean: {ret_series.mean():.2f}%, Vol: {ret_series.std():.2f}%)",
            labels={"value": "Daily Return (%)"}
        )
        apply_chart_theme(hist_fig, height=280)
        st.plotly_chart(hist_fig, use_container_width=True)

    with col_vol:
        st.subheader("3. Annualized Volatility Dynamics")
        vol_col = "rolling_vol_21d" if "rolling_vol_21d" in active_plot_df.columns else "rolling_volatility_21d"
        vol_fig = go.Figure()
        if vol_col in active_plot_df.columns:
            vol_fig.add_trace(go.Scatter(
                x=active_plot_df["Date"],
                y=active_plot_df[vol_col],
                mode="lines",
                name="21D Rolling Vol (Ann. %)",
                line=dict(color=FINANCIAL_COLORS["secondary"], width=2)
            ))
            vol_fig.add_hline(y=active_plot_df[vol_col].mean(), line_dash="dash", line_color="gray", annotation_text="Period Mean Vol")
        apply_chart_theme(vol_fig, height=320, title=f"{sec_title} - 21-Day Rolling Volatility (% p.a.)")
        st.plotly_chart(vol_fig, use_container_width=True)

        # 4. VIX Movements in Market Context
        st.subheader("4. CBOE Volatility Index (VIX) Movements")
        vix_sub = vix_df[(vix_df["Date"] >= active_plot_df["Date"].min()) & (vix_df["Date"] <= active_plot_df["Date"].max())]
        vix_fig = go.Figure()
        vix_fig.add_trace(go.Scatter(
            x=vix_sub["Date"],
            y=vix_sub["Close"],
            mode="lines",
            name="VIX Level",
            line=dict(color=FINANCIAL_COLORS["vix"], width=1.5)
        ))
        vix_fig.add_hline(y=20, line_dash="dot", line_color="orange", annotation_text="Elevated Vol (>20)")
        vix_fig.add_hline(y=30, line_dash="dash", line_color="crimson", annotation_text="High Vol / Stress (>30)")
        apply_chart_theme(vix_fig, height=280, title="CBOE Volatility Index (VIX) Over Period")
        st.plotly_chart(vix_fig, use_container_width=True)

    # 5. Compare Company Performance with Parent Index
    if comp_df is not None and not comp_df.empty:
        st.markdown("---")
        st.subheader(f"5. Company Performance vs Parent Index ({idx_cfg['label']})")
        st.caption("Normalized performance comparison (Rebased to 100.0) and relative daily return spread.")

        cmp_merged = pd.merge(
            comp_df[["Date", "Close", "daily_return_pct"]].rename(columns={"Close": "Close_sec", "daily_return_pct": "ret_sec"}),
            idx_df[["Date", "Close", "daily_return_pct"]].rename(columns={"Close": "Close_idx", "daily_return_pct": "ret_idx"}),
            on="Date",
            how="inner"
        ).sort_values("Date").reset_index(drop=True)

        if not cmp_merged.empty:
            cmp_merged["norm_sec"] = (cmp_merged["Close_sec"] / cmp_merged["Close_sec"].iloc[0]) * 100.0
            cmp_merged["norm_idx"] = (cmp_merged["Close_idx"] / cmp_merged["Close_idx"].iloc[0]) * 100.0
            cmp_merged["daily_spread_pct"] = cmp_merged["ret_sec"] - cmp_merged["ret_idx"]
            cmp_merged["cum_spread_pct"] = cmp_merged["norm_sec"] - cmp_merged["norm_idx"]

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                norm_fig = go.Figure()
                norm_fig.add_trace(go.Scatter(x=cmp_merged["Date"], y=cmp_merged["norm_sec"], mode="lines", name=f"{sel_ticker} (Base=100)"))
                norm_fig.add_trace(go.Scatter(x=cmp_merged["Date"], y=cmp_merged["norm_idx"], mode="lines", name=f"{idx_cfg['label']} (Base=100)", line=dict(dash="dash")))
                apply_chart_theme(norm_fig, height=360, title=f"Cumulative Performance: {sel_ticker} vs {idx_cfg['label']}")
                st.plotly_chart(norm_fig, use_container_width=True)

            with col_c2:
                spread_fig = go.Figure()
                spread_colors = [FINANCIAL_COLORS["success"] if s >= 0 else FINANCIAL_COLORS["danger"] for s in cmp_merged["daily_spread_pct"]]
                spread_fig.add_trace(go.Bar(x=cmp_merged["Date"], y=cmp_merged["daily_spread_pct"], marker_color=spread_colors, name="Daily Spread %"))
                spread_fig.add_trace(go.Scatter(x=cmp_merged["Date"], y=cmp_merged["cum_spread_pct"], mode="lines", name="Cumulative Spread %", line=dict(color=FINANCIAL_COLORS["primary"], width=2)))
                spread_fig.add_hline(y=0, line_color="gray")
                apply_chart_theme(spread_fig, height=360, title="Relative Excess Spread (Stock - Index)")
                st.plotly_chart(spread_fig, use_container_width=True)


if __name__ == "__main__":
    render_market_explorer_page()
