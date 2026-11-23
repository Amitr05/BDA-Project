"""
Page 4: Event Impact Analyzer
Quantitative econometric event study engine: Analyze stock/index price movements,
pre/post returns, drawdowns, Market Model abnormal returns (AR/CAR), VIX volatility,
and multi-event comparative trajectory overlay.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from dashboard.app import (
    INDEX_CONFIG,
    RAW_DIR,
    load_index_data,
    load_vix_data,
    load_stock_data,
    load_events_database,
    load_company_event_studies,
    load_event_study,
    load_sector_event_study,
    load_abnormal_returns_detail,
    load_significance_summary,
    load_cross_company_dispersion,
    load_companies_registry,
    load_news_articles,
    _format_num,
    _match_event_subset,
)
from src.analytics.advanced_event_study import (
    calculate_abnormal_returns,
    find_effective_trading_day_index,
)
from dashboard.components.header import render_page_header
from dashboard.components.charts import apply_chart_theme, FINANCIAL_COLORS


def render_event_impact_analyzer_page():
    render_page_header(
        title="Event Impact Analyzer & Econometric Event Study Engine",
        subtitle="Market Model abnormal return estimation, CAR cumulative trajectory [-5, +30], parametric t-test statistical significance, and multi-event overlays.",
        badge="MODULE 3",
        badge_color="#f59e0b"
    )

    index_frames = {k: load_index_data(k) for k in INDEX_CONFIG}
    vix_df = load_vix_data()
    events_df = load_events_database()
    comp_events_df = load_company_event_studies()
    idx_events_df = load_event_study()
    sector_events_df = load_sector_event_study()
    ar_detail_df = load_abnormal_returns_detail()
    sig_df = load_significance_summary()

    if events_df.empty:
        st.warning("Event database missing. Run python main.py to populate.")
        return

    # 1. Target Security Selection: Index vs Company
    sec_col1, sec_col2, sec_col3 = st.columns([1.2, 1.3, 1.5])

    with sec_col1:
        target_mode = st.radio("1. Security Type:", ["Market Index", "Individual Company"], horizontal=True, key="m3_sec_type")

    with sec_col2:
        if target_mode == "Market Index":
            target_idx_key = st.selectbox("Select Benchmark Index:", options=list(INDEX_CONFIG.keys()), format_func=lambda k: INDEX_CONFIG[k]["label"], key="m3_idx_select")
            target_symbol = INDEX_CONFIG[target_idx_key]["label"]
            target_ticker = INDEX_CONFIG[target_idx_key]["ticker"]
            target_price_df = index_frames[target_idx_key].copy()
        else:
            comp_reg = load_companies_registry()
            stocks_dir = RAW_DIR / "stocks"
            avail_tickers = [f.replace("_raw.csv", "") for f in os.listdir(stocks_dir) if f.endswith("_raw.csv")] if stocks_dir.exists() else []
            # Sort with high profile first
            favs = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM", "XOM", "JNJ", "BA", "CAT"]
            top_tickers = [t for t in favs if t in avail_tickers] + sorted([t for t in avail_tickers if t not in favs])

            target_ticker = st.selectbox("Select Company:", options=top_tickers, key="m3_comp_select")
            target_symbol = comp_reg[comp_reg["ticker"] == target_ticker]["company_name"].values[0] if comp_reg is not None and len(comp_reg[comp_reg["ticker"] == target_ticker]) > 0 else target_ticker
            target_price_df = load_stock_data(target_ticker)

    with sec_col3:
        # 2. Select Event Category & Event
        avail_cats = ["All Categories"] + sorted(events_df["category"].dropna().unique().tolist() if "category" in events_df.columns else [])
        sel_ev_cat = st.selectbox("2. Filter Events by Category:", options=avail_cats, key="m3_cat_filter")

    # Filter events for dropdown
    ev_pool = events_df.copy()
    if sel_ev_cat != "All Categories" and "category" in ev_pool.columns:
        ev_pool = ev_pool[ev_pool["category"] == sel_ev_cat]

    # Build a display-label map so event names containing parentheses are handled safely.
    ev_option_map = {}
    for _, r in ev_pool.sort_values("event_date", ascending=False).iterrows():
        label = f"{r['event_name']} ({pd.to_datetime(r['event_date']).strftime('%Y-%m-%d')})"
        ev_option_map[label] = r["event_name"]

    ev_options = list(ev_option_map.keys())
    if not ev_options:
        st.info("No events match the selected category filter.")
        return

    # Default to the Russia-Ukraine invasion when it is available.
    default_ev_idx = 0
    for idx_opt, opt_name in enumerate(ev_options):
        if "Russia" in opt_name and "Invasion" in opt_name:
            default_ev_idx = idx_opt
            break

    sel_ev_full = st.selectbox(
        "3. Select Specific Event to Analyze:",
        options=ev_options,
        index=default_ev_idx,
        key="m3_event_select"
    )

    sel_ev_name = ev_option_map.get(sel_ev_full, "")
    matched_events = events_df[events_df["event_name"] == sel_ev_name]
    if matched_events.empty:
        st.info(f"Event details not found for '{sel_ev_name}'.")
        return

    ev_row = matched_events.iloc[0]
    ev_date_str = pd.to_datetime(ev_row["event_date"]).strftime("%Y-%m-%d")

    # Compute or lookup event metrics
    t0_idx = find_effective_trading_day_index(ev_date_str, target_price_df["Date"].dt.strftime("%Y-%m-%d").tolist()) if target_price_df is not None else None

    # Calculate returns and drawdowns around event
    pre_5d_val = np.nan
    post_5d_val = np.nan
    post_30d_val = np.nan
    max_dd_val = np.nan
    vix_t0_val = np.nan
    vix_5d_chg = np.nan

    if target_price_df is not None and t0_idx is not None:
        closes = target_price_df["Close"].values
        dates_list = target_price_df["Date"].dt.strftime("%Y-%m-%d").tolist()
        t0_date = dates_list[t0_idx]
        n_days = len(closes)

        p0 = closes[t0_idx]
        pre_idx = max(0, t0_idx - 5)
        p_pre = closes[pre_idx]
        if p_pre > 0:
            pre_5d_val = ((p0 - p_pre) / p_pre) * 100.0

        if (t0_idx + 5) < n_days:
            p_post5 = closes[t0_idx + 5]
            post_5d_val = ((p_post5 - p0) / p0) * 100.0

        if (t0_idx + 30) < n_days:
            p_post30 = closes[t0_idx + 30]
            post_30d_val = ((p_post30 - p0) / p0) * 100.0
            lowest = min(closes[t0_idx : t0_idx + 31])
            max_dd_val = ((lowest - p0) / p0) * 100.0

        # VIX metrics
        vix_dates = vix_df["Date"].dt.strftime("%Y-%m-%d").tolist() if vix_df is not None else []
        v_idx = find_effective_trading_day_index(ev_date_str, vix_dates)
        if v_idx is not None and vix_df is not None:
            v_p0 = vix_df.iloc[v_idx]["Close"]
            vix_t0_val = v_p0
            if (v_idx + 5) < len(vix_df):
                v_p5 = vix_df.iloc[v_idx + 5]["Close"]
                vix_5d_chg = v_p5 - v_p0

    # 1. Display 5 Key Metrics Cards & Trajectory
    st.markdown("---")
    st.subheader(f"1. Historical Event Metrics & Price Trajectory: {target_symbol} around {sel_ev_name}")

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Pre-Event 5D Return", _format_num(pre_5d_val, "%"), delta_color="normal")
    mc2.metric("Post-Event 5D Return", _format_num(post_5d_val, "%"), delta_color="normal")
    mc3.metric("Post-Event 30D Return", _format_num(post_30d_val, "%"), delta_color="normal")
    mc4.metric("30D Max Drawdown", _format_num(max_dd_val, "%"), delta_color="inverse")
    mc5.metric("VIX at Event (T0)", _format_num(vix_t0_val), delta=_format_num(vix_5d_chg, " (5D)"), delta_color="inverse")

    # Event Window Trajectory Chart [-30 to +30] with VIX Overlay
    if target_price_df is not None and t0_idx is not None:
        start_pos = max(0, t0_idx - 30)
        end_pos = min(len(target_price_df) - 1, t0_idx + 30)
        win_df = target_price_df.iloc[start_pos : end_pos + 1].copy()
        base_p = target_price_df.iloc[t0_idx]["Close"]

        if base_p > 0:
            win_df["norm_price"] = (win_df["Close"] / base_p) * 100.0
            traj_fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Security Trajectory
            traj_fig.add_trace(go.Scatter(
                x=win_df["Date"],
                y=win_df["norm_price"],
                mode="lines",
                name=f"{target_symbol} (T0=100)",
                line=dict(color=FINANCIAL_COLORS["primary"], width=2.5)
            ), secondary_y=False)

            # If Company, also overlay Parent S&P 500
            if target_mode == "Individual Company":
                sp500_df = index_frames.get("sp500")
                if sp500_df is not None:
                    sp_t0_idx = find_effective_trading_day_index(ev_date_str, sp500_df["Date"].dt.strftime("%Y-%m-%d").tolist())
                    if sp_t0_idx is not None:
                        sp_start = max(0, sp_t0_idx - 30)
                        sp_end = min(len(sp500_df) - 1, sp_t0_idx + 30)
                        sp_win = sp500_df.iloc[sp_start : sp_end + 1].copy()
                        sp_base = sp500_df.iloc[sp_t0_idx]["Close"]
                        if sp_base > 0:
                            sp_win["norm_sp"] = (sp_win["Close"] / sp_base) * 100.0
                            traj_fig.add_trace(go.Scatter(
                                x=sp_win["Date"],
                                y=sp_win["norm_sp"],
                                mode="lines",
                                name="S&P 500 (Base=100)",
                                line=dict(color=FINANCIAL_COLORS["benchmark"], dash="dash")
                            ), secondary_y=False)

            # VIX Overlay
            if vix_df is not None:
                vix_win = vix_df[vix_df["Date"].isin(win_df["Date"])].dropna(subset=["Close"])
                if not vix_win.empty:
                    traj_fig.add_trace(go.Scatter(
                        x=vix_win["Date"],
                        y=vix_win["Close"],
                        mode="lines",
                        name="VIX Level",
                        line=dict(color=FINANCIAL_COLORS["vix"], dash="dot")
                    ), secondary_y=True)

            traj_fig.add_vline(x=target_price_df.iloc[t0_idx]["Date"], line_dash="dash", line_color="crimson", annotation_text="T0 Event")
            traj_fig.add_hline(y=100, line_dash="dot", line_color="gray", secondary_y=False)

            apply_chart_theme(traj_fig, height=440, title=f"Event Window Dynamics: {target_symbol} around {sel_ev_name} (±30 Trading Days)")
            traj_fig.update_yaxes(title_text="Normalized Price (T0 = 100)", secondary_y=False)
            traj_fig.update_yaxes(title_text="VIX Level", secondary_y=True)
            st.plotly_chart(traj_fig, use_container_width=True)

    # 2. Abnormal Returns & Cumulative Abnormal Returns (CAR)
    st.markdown("---")
    st.subheader("2. Econometric Abnormal Returns (AR) & Cumulative Abnormal Returns (CAR)")
    st.caption("Benchmark Market Model estimated over [T0-120, T0-11] clean pre-event window against the S&P 500 index.")

    sp_benchmark = index_frames.get("sp500")
    ar_res = calculate_abnormal_returns(
        security_df=target_price_df,
        market_df=sp_benchmark,
        event_date=ev_date_str
    ) if target_price_df is not None and sp_benchmark is not None else {}

    if ar_res:
        ac1, ac2, ac3, ac4 = st.columns(4)
        ac1.metric("Pre-5D CAR", _format_num(ar_res.get("car_pre_5d_pct"), "%"))
        ac2.metric("Post-5D CAR", _format_num(ar_res.get("car_post_5d_pct"), "%"))
        ac3.metric("Post-30D CAR", _format_num(ar_res.get("car_post_30d_pct"), "%"))
        ac4.metric("Estimated Market Beta", _format_num(ar_res.get("beta")))

        daily_ar = ar_res.get("daily_ar_series", [])
        if daily_ar:
            ar_df = pd.DataFrame(daily_ar)
            car_plot = px.line(
                ar_df,
                x="relative_trading_day",
                y="car_pct",
                title=f"Cumulative Abnormal Return (CAR %) Path: {target_symbol} around {sel_ev_name}",
                labels={"relative_trading_day": "Relative Trading Day (T0 = 0)", "car_pct": "Cumulative Abnormal Return (%)"},
            )
            car_plot.add_vline(x=0, line_dash="dash", line_color="crimson", annotation_text="T0 Event Date")
            car_plot.add_hline(y=0.0, line_dash="dot", line_color="gray")
            apply_chart_theme(car_plot, height=360)
            st.plotly_chart(car_plot, use_container_width=True)

    # Statistical significance summary
    if sig_df is not None and not sig_df.empty:
        st.write("**Cross-Sectional Statistical Significance Tests (CAR)**")
        st.dataframe(sig_df[[
            "scope", "metric", "sample_size", "mean_car_pct", "standard_error", "t_statistic", "p_value", "statistically_significant_5pct"
        ]].rename(columns={
            "scope": "Sample Scope", "metric": "Metric", "sample_size": "N",
            "mean_car_pct": "Mean CAR %", "standard_error": "Std Error", "t_statistic": "t-stat", "p_value": "p-value", "statistically_significant_5pct": "Sig at 5%?"
        }), use_container_width=True, hide_index=True)

        st.caption("Econometric Note: Evaluates parametric cross-sectional two-tailed t-test H0: Mean(CAR) = 0. Statistical significance reflects empirical market association, not definitive causal proof.")

    # 3. Compare Multiple Events
    st.markdown("---")
    st.subheader(f"3. Multi-Event Comparative Performance: {target_symbol}")
    st.caption("Overlay normalized price trajectories across multiple historical events on a synchronized relative trading day scale [-5, +30].")

    multi_ev_choices = events_df["event_name"].unique().tolist()
    default_multi = [sel_ev_name]
    for key_pattern in ["Brexit", "Ukraine", "Pandemic", "Banking Crisis"]:
        for c in multi_ev_choices:
            if key_pattern in c and c not in default_multi:
                default_multi.append(c)
                break
        if len(default_multi) >= 3:
            break

    selected_multi_events = st.multiselect("Select Events to Compare:", options=multi_ev_choices, default=default_multi, key="m3_multi_ev")

    if selected_multi_events and target_price_df is not None:
        comp_ev_fig = go.Figure()
        summary_rows = []

        for ev_n in selected_multi_events:
            ev_sub_row = events_df[events_df["event_name"] == ev_n].iloc[0]
            ev_d = pd.to_datetime(ev_sub_row["event_date"]).strftime("%Y-%m-%d")

            idx_pos = find_effective_trading_day_index(ev_d, target_price_df["Date"].dt.strftime("%Y-%m-%d").tolist())
            if idx_pos is not None:
                st_p = max(0, idx_pos - 5)
                en_p = min(len(target_price_df) - 1, idx_pos + 30)
                sub_w = target_price_df.iloc[st_p : en_p + 1].copy()
                base_close = target_price_df.iloc[idx_pos]["Close"]

                if base_close > 0:
                    sub_w["norm"] = (sub_w["Close"] / base_close) * 100.0
                    sub_w["rel_day"] = range(st_p - idx_pos, en_p - idx_pos + 1)

                    comp_ev_fig.add_trace(go.Scatter(
                        x=sub_w["rel_day"],
                        y=sub_w["norm"],
                        mode="lines",
                        name=f"{ev_n} ({ev_d})"
                    ))

                    p_5 = target_price_df.iloc[idx_pos + 5]["Close"] if (idx_pos + 5) < len(target_price_df) else np.nan
                    p_30 = target_price_df.iloc[idx_pos + 30]["Close"] if (idx_pos + 30) < len(target_price_df) else np.nan
                    lowest_close = min(target_price_df.iloc[idx_pos : idx_pos + 31]["Close"]) if (idx_pos + 30) < len(target_price_df) else np.nan

                    summary_rows.append({
                        "Event": ev_n,
                        "Event Date": ev_d,
                        "Category": ev_sub_row.get("category", "General"),
                        "Pre-5D %": round(((base_close - target_price_df.iloc[st_p]["Close"]) / target_price_df.iloc[st_p]["Close"]) * 100.0, 2) if target_price_df.iloc[st_p]["Close"] > 0 else None,
                        "Post-5D %": round(((p_5 - base_close) / base_close) * 100.0, 2) if pd.notna(p_5) else None,
                        "Post-30D %": round(((p_30 - base_close) / base_close) * 100.0, 2) if pd.notna(p_30) else None,
                        "30D Drawdown %": round(((lowest_close - base_close) / base_close) * 100.0, 2) if pd.notna(lowest_close) else None
                    })

        comp_ev_fig.add_vline(x=0, line_dash="dash", line_color="crimson", annotation_text="T0 Event")
        comp_ev_fig.add_hline(y=100, line_dash="dot", line_color="gray")
        apply_chart_theme(comp_ev_fig, height=400, title=f"Multi-Event Price Trajectories: {target_symbol} (T0 = 100)")
        comp_ev_fig.update_layout(xaxis_title="Relative Trading Days (T0 = 0)", yaxis_title="Normalized Price")
        st.plotly_chart(comp_ev_fig, use_container_width=True)

        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # 4. Related News Coverage & Sentiment for this event
    st.markdown("---")
    st.subheader(f"4. Related GDELT News Coverage & Sentiment: {sel_ev_name}")
    news_df = load_news_articles()

    if not news_df.empty:
        ev_id = ev_row.get("event_id", "")
        n_sub = _match_event_subset(news_df, sel_ev_name, ev_date_str)
        if n_sub.empty and "matched_event_name" in news_df.columns:
            n_sub = news_df[
                (news_df["matched_event_id"] == ev_id) |
                (news_df["matched_event_name"].astype(str).str.contains(sel_ev_name[:15], case=False, na=False))
            ]
        if not n_sub.empty:
            st.dataframe(n_sub[[
                c for c in ["published_date", "title", "domain", "sentiment_label", "gdelt_tone", "polarity", "timing_phase"]
                if c in n_sub.columns
            ]], use_container_width=True, hide_index=True)
        else:
            st.info(f"No specific news records matching '{sel_ev_name}' in the current sample.")


if __name__ == "__main__":
    render_event_impact_analyzer_page()
