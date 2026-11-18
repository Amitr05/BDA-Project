"""
Streamlit Dashboard: Multi-Asset Global Event & US Stock Market Analytics Platform
==================================================================================
Comprehensive interactive analytics dashboard featuring three major modules:
  - MODULE 1: MARKET EXPLORER
  - MODULE 2: GLOBAL EVENT EXPLORER
  - MODULE 3: EVENT IMPACT ANALYZER

Built with Streamlit, Plotly, Apache Spark, and PyArrow.
Run from repository root:
    python -m streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Resolve paths from repository root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.advanced_event_study import (
    calculate_abnormal_returns,
    find_effective_trading_day_index,
    estimate_market_model_parameters,
    compute_statistical_tests
)

# Storage paths
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ANALYTICS_DIR = PROJECT_ROOT / "data" / "analytics"

INDEX_CONFIG = {
    "sp500": {
        "label": "S&P 500",
        "ticker": "^GSPC",
        "path": PROCESSED_DIR / "sp500_processed.parquet",
    },
    "nasdaq": {
        "label": "NASDAQ Composite",
        "ticker": "^IXIC",
        "path": PROCESSED_DIR / "nasdaq_processed.parquet",
    },
    "dow_jones": {
        "label": "Dow Jones Industrial Average",
        "ticker": "^DJI",
        "path": PROCESSED_DIR / "dow_jones_processed.parquet",
    },
    "russell2000": {
        "label": "Russell 2000",
        "ticker": "^RUT",
        "path": PROCESSED_DIR / "russell2000_processed.parquet",
    },
}

VIX_PATH = PROCESSED_DIR / "vix_processed.parquet"
EVENT_STUDY_PATH = PROCESSED_DIR / "event_study_results.csv"
SECTOR_ETFS_PATH = PROCESSED_DIR / "etfs" / "sector_etfs_processed.parquet"
MACRO_ALIGNED_PATH = PROCESSED_DIR / "macro" / "macro_indicators_aligned.parquet"
SECTOR_EVENT_STUDY_PATH = PROCESSED_DIR / "sector_event_study_results.csv"

# Layer 3 Analytics Paths
COMPANY_EVENT_STUDY_PATH = ANALYTICS_DIR / "event_study_companies.parquet"
ABNORMAL_RETURNS_DETAIL_PATH = ANALYTICS_DIR / "abnormal_returns_detail.parquet"
SIGNIFICANCE_SUMMARY_PATH = ANALYTICS_DIR / "abnormal_returns_significance.parquet"
CROSS_COMPANY_DISPERSION_PATH = ANALYTICS_DIR / "cross_company_dispersion.parquet"
MACRO_CONTEXT_PATH = ANALYTICS_DIR / "macro_context_analysis.parquet"
CATEGORY_COMPARISONS_PATH = ANALYTICS_DIR / "event_category_comparisons.parquet"

# Global Event Categories (The 7 Standard Categories)
EVENT_CATEGORIES_7 = [
    "War and Geopolitical Conflicts",
    "Politics and Elections",
    "Economic Events",
    "Technology and Gaming",
    "Sports and Entertainment",
    "Natural Disasters",
    "Corporate Events"
]

METRIC_COLUMNS = {
    "Pre-5D Return (%)": "pre_5d_return_pct",
    "Post-5D Return (%)": "post_5d_return_pct",
    "Post-30D Return (%)": "post_30d_return_pct",
}

DISPLAY_COLUMNS = [
    "event_name",
    "calendar_event_date",
    "effective_trading_date",
    "event_category",
    "index",
    "ticker",
    "price_t0",
    "pre_5d_return_pct",
    "post_5d_return_pct",
    "post_30d_return_pct",
    "max_drawdown_30d_pct",
    "vix_at_event",
    "vix_5d_change",
    "vix_max_30d",
    "short_description",
]

DISPLAY_NAMES = {
    "event_name": "Event",
    "calendar_event_date": "Calendar Event Date",
    "effective_trading_date": "Effective Trading Date (T0)",
    "event_category": "Category",
    "index": "Index",
    "ticker": "Ticker",
    "price_t0": "Close at T0",
    "pre_5d_return_pct": "Pre-5D Return (%)",
    "post_5d_return_pct": "Post-5D Return (%)",
    "post_30d_return_pct": "Post-30D Return (%)",
    "max_drawdown_30d_pct": "30D Minimum Return / Drawdown (%)",
    "vix_at_event": "VIX at T0",
    "vix_5d_change": "VIX Change over 5D",
    "vix_max_30d": "VIX Maximum over 30D",
    "short_description": "Description",
}


# =============================================================================
# DATA LOADERS & CACHING
# =============================================================================

def _read_parquet(path: Path) -> Optional[pd.DataFrame]:
    """Read a processed Parquet dataset and normalise its Date column."""
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if "Date" not in frame.columns:
        return frame
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).sort_values("Date")
    frame = frame.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return frame


@st.cache_data(show_spinner=False)
def load_index_data(index_key: str) -> Optional[pd.DataFrame]:
    """Load one of the four processed equity-index datasets."""
    if index_key not in INDEX_CONFIG:
        raise ValueError(f"Unsupported index key: {index_key}")
    return _read_parquet(INDEX_CONFIG[index_key]["path"])


@st.cache_data(show_spinner=False)
def load_vix_data() -> Optional[pd.DataFrame]:
    """Load the market-wide VIX dataset independently of the equity indices."""
    return _read_parquet(VIX_PATH)


@st.cache_data(show_spinner=False)
def load_market_data():
    """Return the S&P 500 and VIX frames for backwards compatibility."""
    return load_index_data("sp500"), load_vix_data()


@st.cache_data(show_spinner=False)
def load_event_study() -> Optional[pd.DataFrame]:
    """Load and validate the long-format event-study CSV."""
    if not EVENT_STUDY_PATH.exists():
        return None
    frame = pd.read_csv(EVENT_STUDY_PATH)
    required = {
        "event_name",
        "effective_trading_date",
        "index",
        "index_key",
        "pre_5d_return_pct",
        "post_5d_return_pct",
        "post_30d_return_pct",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in event study CSV: {', '.join(sorted(missing))}")
    frame = frame.copy()
    frame["effective_trading_date"] = pd.to_datetime(frame["effective_trading_date"], errors="coerce")
    if "calendar_event_date" in frame.columns:
        frame["calendar_event_date"] = pd.to_datetime(frame["calendar_event_date"], errors="coerce")
    return frame


@st.cache_data(show_spinner=False)
def load_sector_etfs_data() -> Optional[pd.DataFrame]:
    """Load processed sector ETF panel dataset."""
    if not SECTOR_ETFS_PATH.exists():
        return None
    frame = pd.read_parquet(SECTOR_ETFS_PATH)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    return frame.dropna(subset=["Date"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_macro_aligned_data() -> Optional[pd.DataFrame]:
    """Load point-in-time aligned FRED macroeconomic indicators."""
    if not MACRO_ALIGNED_PATH.exists():
        return None
    frame = pd.read_parquet(MACRO_ALIGNED_PATH)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    return frame.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_sector_event_study() -> Optional[pd.DataFrame]:
    """Load sector ETF event-study metrics."""
    if not SECTOR_EVENT_STUDY_PATH.exists():
        return None
    frame = pd.read_csv(SECTOR_EVENT_STUDY_PATH)
    frame["effective_trading_date"] = pd.to_datetime(frame["effective_trading_date"], errors="coerce")
    return frame


@st.cache_data(show_spinner=False)
def load_company_event_studies() -> Optional[pd.DataFrame]:
    """Load company-level event study results from Layer 3 analytics."""
    if not COMPANY_EVENT_STUDY_PATH.exists():
        return None
    return pd.read_parquet(COMPANY_EVENT_STUDY_PATH)


@st.cache_data(show_spinner=False)
def load_abnormal_returns_detail() -> Optional[pd.DataFrame]:
    """Load daily abnormal returns time series from Layer 3 analytics."""
    if not ABNORMAL_RETURNS_DETAIL_PATH.exists():
        return None
    return pd.read_parquet(ABNORMAL_RETURNS_DETAIL_PATH)


@st.cache_data(show_spinner=False)
def load_significance_summary() -> Optional[pd.DataFrame]:
    """Load econometric statistical significance test summary from Layer 3 analytics."""
    if not SIGNIFICANCE_SUMMARY_PATH.exists():
        return None
    return pd.read_parquet(SIGNIFICANCE_SUMMARY_PATH)


@st.cache_data(show_spinner=False)
def load_cross_company_dispersion() -> Optional[pd.DataFrame]:
    """Load intra-sector cross-company dispersion metrics from Layer 3 analytics."""
    if not CROSS_COMPANY_DISPERSION_PATH.exists():
        return None
    return pd.read_parquet(CROSS_COMPANY_DISPERSION_PATH)


@st.cache_data(show_spinner=False)
def load_macro_context_analysis() -> Optional[pd.DataFrame]:
    """Load macroeconomic regime analysis from Layer 3 analytics."""
    if not MACRO_CONTEXT_PATH.exists():
        return None
    return pd.read_parquet(MACRO_CONTEXT_PATH)


@st.cache_data(show_spinner=False)
def load_category_comparisons() -> Optional[pd.DataFrame]:
    """Load event category comparisons from Layer 3 analytics."""
    if not CATEGORY_COMPARISONS_PATH.exists():
        return None
    return pd.read_parquet(CATEGORY_COMPARISONS_PATH)


@st.cache_data(show_spinner=False)
def load_companies_registry() -> Optional[pd.DataFrame]:
    """Load company metadata registry."""
    path = RAW_DIR / "companies.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_index_constituents() -> Optional[pd.DataFrame]:
    """Load historical index constituents mapping."""
    path = RAW_DIR / "index_constituents.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_events_database() -> pd.DataFrame:
    """Load comprehensive events database."""
    p_exp = RAW_DIR / "events_expanded.csv"
    if p_exp.exists():
        df = pd.read_csv(p_exp)
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
        return df
    p_cur = RAW_DIR / "events.csv"
    if p_cur.exists():
        df = pd.read_csv(p_cur)
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
        if "category" not in df.columns and "event_category" in df.columns:
            df["category"] = df["event_category"]
        return df
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_news_articles() -> pd.DataFrame:
    """Load GDELT processed news articles with sentiment."""
    path_pq = PROCESSED_DIR / "news_articles.parquet"
    if path_pq.exists():
        df = pd.read_parquet(path_pq)
        if "published_date" in df.columns:
            df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
        return df
    path_csv = PROCESSED_DIR / "news_articles.csv"
    if path_csv.exists():
        df = pd.read_csv(path_csv)
        if "published_date" in df.columns:
            df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
        return df
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_stock_data(ticker: str) -> Optional[pd.DataFrame]:
    """Load individual equity OHLCV data with returns and rolling volatility."""
    path = RAW_DIR / "stocks" / f"{ticker}_raw.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["daily_return"] = df["Close"].pct_change()
    df["daily_return_pct"] = df["daily_return"] * 100.0
    df["rolling_vol_21d"] = df["daily_return"].rolling(21).std() * (252 ** 0.5) * 100.0
    return df


@st.cache_data(show_spinner=False)
def get_available_companies_for_index(index_key: str) -> pd.DataFrame:
    """Filter company registry for constituents of selected index with local raw data."""
    comp_df = load_companies_registry()
    cst_df = load_index_constituents()
    stocks_dir = RAW_DIR / "stocks"

    if comp_df is None or not stocks_dir.exists():
        return pd.DataFrame()

    avail_tickers = {f.replace("_raw.csv", "") for f in os.listdir(stocks_dir) if f.endswith("_raw.csv")}
    index_ticker = INDEX_CONFIG.get(index_key, {}).get("ticker", "^GSPC")

    if cst_df is not None and not cst_df.empty:
        idx_cst = cst_df[cst_df["index_id"] == index_ticker]["ticker"].unique()
        matched = comp_df[comp_df["ticker"].isin(idx_cst) & comp_df["ticker"].isin(avail_tickers)]
        if not matched.empty:
            return matched.sort_values("company_name").reset_index(drop=True)

    index_name_map = {
        "sp500": "S&P 500",
        "nasdaq": "NASDAQ",
        "dow_jones": "Dow Jones",
        "russell2000": "Russell 2000"
    }
    target_name = index_name_map.get(index_key, "S&P 500")
    matched = comp_df[comp_df["indices"].str.contains(target_name, na=False, case=False) & comp_df["ticker"].isin(avail_tickers)]
    if not matched.empty:
        return matched.sort_values("company_name").reset_index(drop=True)

    return comp_df[comp_df["ticker"].isin(avail_tickers)].sort_values("company_name").reset_index(drop=True)


def _match_event_subset(df: Optional[pd.DataFrame], ev_name: str, ev_date_str: str) -> pd.DataFrame:
    """Safely match and filter an event subset across datasets with varying naming conventions."""
    if df is None or df.empty:
        return pd.DataFrame()

    # 1. Exact event_name match
    if "event_name" in df.columns:
        sub = df[df["event_name"] == ev_name]
        if not sub.empty:
            return sub

    # 2. Date-based match (effective_trading_date, event_date, calendar_event_date)
    for d_col in ["effective_trading_date", "event_date", "calendar_event_date"]:
        if d_col in df.columns:
            sub = df[pd.to_datetime(df[d_col], errors="coerce").dt.strftime("%Y-%m-%d") == ev_date_str]
            if not sub.empty:
                return sub

    # 3. Canonical key / normalized title match
    if "event_name" in df.columns:
        norm_ev = ev_name.lower().replace("–", "-")
        for key in ["russia-ukraine", "brexit", "covid", "banking crisis", "svb", "trade war", "gaza", "israel-hamas", "yen carry"]:
            if key in norm_ev:
                sub = df[df["event_name"].str.lower().str.replace("–", "-").str.contains(key, na=False)]
                if not sub.empty:
                    return sub

    return pd.DataFrame()


def _format_num(val: Any, suffix: str = "", digits: int = 2) -> str:
    if pd.isna(val) or val is None:
        return "N/A"
    try:
        return f"{float(val):,.{digits}f}{suffix}"
    except (ValueError, TypeError):
        return str(val)


# =============================================================================
# MODULE 1: MARKET EXPLORER
# =============================================================================

def render_market_explorer(index_frames: Dict[str, pd.DataFrame], vix_df: pd.DataFrame):
    st.subheader("Module 1: Market Explorer")
    st.caption(
        "Interactive exploration of market benchmark indices, individual constituent equities, "
        "parent-index relative performance, intra-sector peer comparisons, and CBOE VIX movements."
    )

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
    st.write(f"### 1. Historical Stock Price Movements: {sec_title}")

    active_plot_df = comp_df if comp_df is not None else idx_df
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(
        x=active_plot_df["Date"],
        y=active_plot_df["Close"],
        mode="lines",
        name=f"{sec_title} Close",
        line=dict(color="#1f77b4", width=2)
    ))
    price_fig.update_layout(
        title=f"{sec_title} - Historical Closing Price",
        xaxis_title="Date",
        yaxis_title="Close Price (USD)",
        hovermode="x unified",
        height=420
    )
    st.plotly_chart(price_fig, use_container_width=True)

    # 2. Daily Returns and 3. Volatility Side by Side
    col_ret, col_vol = st.columns(2)

    with col_ret:
        st.write("### 2. Daily Returns & Distribution")
        ret_series = active_plot_df["daily_return_pct"].dropna()

        # Daily bar chart
        bar_colors = ["#2ca02c" if r >= 0 else "#d62728" for r in ret_series]
        ret_fig = go.Figure()
        ret_fig.add_trace(go.Bar(
            x=active_plot_df.dropna(subset=["daily_return_pct"])["Date"],
            y=ret_series,
            marker_color=bar_colors,
            name="Daily Return %"
        ))
        ret_fig.add_hline(y=0, line_color="gray", line_width=1)
        ret_fig.update_layout(title=f"{sec_title} - Daily Percentage Returns", height=320, hovermode="x unified")
        st.plotly_chart(ret_fig, use_container_width=True)

        # Return distribution histogram
        hist_fig = px.histogram(
            ret_series,
            nbins=50,
            marginal="box",
            title=f"{sec_title} - Return Distribution (Mean: {ret_series.mean():.2f}%, Vol: {ret_series.std():.2f}%)",
            labels={"value": "Daily Return (%)"}
        )
        hist_fig.update_layout(height=280)
        st.plotly_chart(hist_fig, use_container_width=True)

    with col_vol:
        st.write("### 3. Annualized Volatility Dynamics")
        vol_col = "rolling_vol_21d" if "rolling_vol_21d" in active_plot_df.columns else "rolling_volatility_21d"
        vol_fig = go.Figure()
        if vol_col in active_plot_df.columns:
            vol_fig.add_trace(go.Scatter(
                x=active_plot_df["Date"],
                y=active_plot_df[vol_col],
                mode="lines",
                name="21D Rolling Vol (Ann. %)",
                line=dict(color="#ff7f0e", width=2)
            ))
            vol_fig.add_hline(y=active_plot_df[vol_col].mean(), line_dash="dash", line_color="gray", annotation_text="Period Mean Vol")
        vol_fig.update_layout(title=f"{sec_title} - 21-Day Rolling Volatility (% p.a.)", height=320, hovermode="x unified")
        st.plotly_chart(vol_fig, use_container_width=True)

        # 4. VIX Movements in Market Context
        st.write("### 4. CBOE Volatility Index (VIX) Movements")
        vix_sub = vix_df[(vix_df["Date"] >= active_plot_df["Date"].min()) & (vix_df["Date"] <= active_plot_df["Date"].max())]
        vix_fig = go.Figure()
        vix_fig.add_trace(go.Scatter(
            x=vix_sub["Date"],
            y=vix_sub["Close"],
            mode="lines",
            name="VIX Level",
            line=dict(color="#9467bd", width=1.5)
        ))
        vix_fig.add_hline(y=20, line_dash="dot", line_color="orange", annotation_text="Elevated Vol (>20)")
        vix_fig.add_hline(y=30, line_dash="dash", line_color="crimson", annotation_text="High Vol / Stress (>30)")
        vix_fig.update_layout(title="CBOE Volatility Index (VIX) Over Period", height=280, hovermode="x unified")
        st.plotly_chart(vix_fig, use_container_width=True)

    # 5. Compare Company Performance with Parent Index
    if comp_df is not None and not comp_df.empty:
        st.markdown("---")
        st.write(f"### 5. Company Performance vs Parent Index ({idx_cfg['label']})")
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
                norm_fig.update_layout(title=f"Cumulative Performance: {sel_ticker} vs {idx_cfg['label']}", height=360, hovermode="x unified")
                st.plotly_chart(norm_fig, use_container_width=True)

            with col_c2:
                spread_fig = go.Figure()
                spread_colors = ["#2ca02c" if s >= 0 else "#d62728" for s in cmp_merged["daily_spread_pct"]]
                spread_fig.add_trace(go.Bar(x=cmp_merged["Date"], y=cmp_merged["daily_spread_pct"], marker_color=spread_colors, name="Daily Spread %"))
                spread_fig.add_trace(go.Scatter(x=cmp_merged["Date"], y=cmp_merged["cum_spread_pct"], mode="lines", name="Cumulative Spread %", line=dict(color="#1f77b4", width=2)))
                spread_fig.add_hline(y=0, line_color="gray")
                spread_fig.update_layout(title=f"Relative Excess Spread (Stock - Index)", height=360, hovermode="x unified")
                st.plotly_chart(spread_fig, use_container_width=True)

    # 6. Compare Multiple Companies within the Same Sector
    st.markdown("---")
    st.write("### 6. Intra-Sector Peer Company Comparison")
    comp_registry = load_companies_registry()

    if comp_registry is not None and not comp_registry.empty:
        all_sectors = sorted([s for s in comp_registry["sector"].dropna().unique().tolist() if s != "Unknown"])
        default_sec = comp_meta.get("sector", "Information Technology")
        if default_sec not in all_sectors:
            default_sec = all_sectors[0]

        sel_sector_peers = st.selectbox("Select Sector for Peer Comparison:", options=all_sectors, index=all_sectors.index(default_sec))
        sector_stocks = comp_registry[comp_registry["sector"] == sel_sector_peers]["ticker"].tolist()

        # Check local files availability
        stocks_dir = RAW_DIR / "stocks"
        avail_sector_stocks = [t for t in sector_stocks if (stocks_dir / f"{t}_raw.csv").exists()]

        if len(avail_sector_stocks) >= 2:
            default_peers = avail_sector_stocks[:min(4, len(avail_sector_stocks))]
            if sel_ticker and sel_ticker in avail_sector_stocks and sel_ticker not in default_peers:
                default_peers = [sel_ticker] + default_peers[:3]

            chosen_peers = st.multiselect("Select Peer Companies to Compare:", options=avail_sector_stocks, default=default_peers)

            if len(chosen_peers) >= 2:
                peer_dfs = {}
                for t in chosen_peers:
                    df_t = load_stock_data(t)
                    if df_t is not None and not df_t.empty:
                        peer_dfs[t] = df_t

                if len(peer_dfs) >= 2:
                    # Rebased comparison chart
                    peer_pivot = pd.concat([df.set_index("Date")["Close"].rename(t) for t, df in peer_dfs.items()], axis=1).dropna()
                    norm_peers = (peer_pivot / peer_pivot.iloc[0]) * 100.0

                    col_p1, col_p2 = st.columns([1.2, 0.8])
                    with col_p1:
                        peer_fig = px.line(
                            norm_peers.reset_index(),
                            x="Date",
                            y=chosen_peers,
                            title=f"Intra-Sector Normalized Performance (Base=100) - {sel_sector_peers}",
                            labels={"value": "Indexed Performance", "variable": "Company"}
                        )
                        peer_fig.update_layout(height=380, hovermode="x unified")
                        st.plotly_chart(peer_fig, use_container_width=True)

                    with col_p2:
                        ret_pivot = pd.concat([df.set_index("Date")["daily_return"].rename(t) for t, df in peer_dfs.items()], axis=1).dropna()
                        corr_peers = ret_pivot.corr().round(2)
                        p_corr_fig = px.imshow(
                            corr_peers,
                            text_auto=True,
                            aspect="auto",
                            color_continuous_scale="RdBu_r",
                            zmin=-1.0,
                            zmax=1.0,
                            title=f"Daily Return Correlation Matrix ({sel_sector_peers})"
                        )
                        p_corr_fig.update_layout(height=380)
                        st.plotly_chart(p_corr_fig, use_container_width=True)

                    # Quantitative peer table
                    peer_stats = []
                    for t in chosen_peers:
                        sdf = peer_dfs[t]
                        p_0 = sdf.iloc[0]["Close"]
                        p_1 = sdf.iloc[-1]["Close"]
                        t_ret = ((p_1 - p_0) / p_0) * 100.0 if p_0 > 0 else 0.0
                        v_ann = float(sdf["daily_return"].std() * (252 ** 0.5) * 100.0)
                        sr = (t_ret / v_ann) if v_ann > 0 else 0.0
                        # max drawdown
                        roll_max = sdf["Close"].cummax()
                        dd = ((sdf["Close"] - roll_max) / roll_max).min() * 100.0
                        peer_stats.append({
                            "Ticker": t,
                            "Company": comp_registry[comp_registry["ticker"] == t]["company_name"].values[0] if len(comp_registry[comp_registry["ticker"] == t]) > 0 else t,
                            "Total Return (%)": round(t_ret, 2),
                            "Annual Volatility (%)": round(v_ann, 2),
                            "Sharpe Ratio": round(sr, 2),
                            "Max Drawdown (%)": round(dd, 2)
                        })
                    st.dataframe(pd.DataFrame(peer_stats), use_container_width=True, hide_index=True)
        else:
            st.info(f"Insufficient local stock records for sector '{sel_sector_peers}' to conduct intra-sector peer comparison.")

    # 7. Explore Sector ETF Performance
    st.markdown("---")
    st.write("### 7. Sector ETF Benchmark Performance (8 Select SPDR ETFs)")
    st.caption("Benchmark ETFs: XLK (Tech), XLF (Financials), XLE (Energy), XLV (Health), XLI (Industrials), XLY (Discretionary), XLP (Staples), XLU (Utilities)")

    sector_etfs_df = load_sector_etfs_data()
    if sector_etfs_df is not None and not sector_etfs_df.empty:
        etf_pivot = sector_etfs_df.pivot(index="Date", columns="Ticker", values="Close").dropna()
        if not etf_pivot.empty:
            norm_etf = (etf_pivot / etf_pivot.iloc[0]) * 100.0
            col_e1, col_e2 = st.columns([1.2, 0.8])
            with col_e1:
                e_fig = px.line(
                    norm_etf.reset_index(),
                    x="Date",
                    y=[c for c in norm_etf.columns if c != "Date"],
                    title="Comparative Sector ETF Cumulative Performance (Rebased to 100)",
                    labels={"value": "Normalized Return", "variable": "Sector ETF"}
                )
                e_fig.update_layout(height=380, hovermode="x unified")
                st.plotly_chart(e_fig, use_container_width=True)

            with col_e2:
                etf_ret_pivot = sector_etfs_df.pivot(index="Date", columns="Ticker", values="daily_return").dropna()
                etf_corr = etf_ret_pivot.corr().round(2)
                etf_corr_fig = px.imshow(
                    etf_corr,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    zmin=-1.0,
                    zmax=1.0,
                    title="Sector ETF Daily Return Correlation Matrix"
                )
                etf_corr_fig.update_layout(height=380)
                st.plotly_chart(etf_corr_fig, use_container_width=True)


# =============================================================================
# MODULE 2: GLOBAL EVENT EXPLORER
# =============================================================================

def render_global_event_explorer():
    st.subheader("Module 2: Global Event Explorer")
    st.caption(
        "Browse, filter, and inspect global geopolitical conflicts, elections, macroeconomic shocks, "
        "and natural disasters, along with associated corporate entities and GDELT news sentiment."
    )

    events_df = load_events_database()
    if events_df.empty:
        st.warning("No global events found in data/raw/events_expanded.csv or events.csv.")
        return

    # Clean dates and categories
    events_df["event_date_dt"] = pd.to_datetime(events_df["event_date"], errors="coerce")
    if "category" not in events_df.columns and "event_category" in events_df.columns:
        events_df["category"] = events_df["event_category"]

    # 1. Filters Row: Categories, Regions, Date Range, Search
    f1, f2, f3 = st.columns([1.5, 1.2, 1.3])

    with f1:
        # Filter by 7 event categories
        avail_cats = sorted(events_df["category"].dropna().unique().tolist())
        selected_cats = st.multiselect(
            "Filter by Event Category (7 Core Categories):",
            options=avail_cats,
            default=avail_cats,
            key="m2_cat_filter"
        )

    with f2:
        # Filter by geographical region
        regions = ["All Regions"]
        if "region" in events_df.columns:
            regions += sorted(events_df["region"].dropna().unique().tolist())
        selected_region = st.selectbox("Filter by Region:", options=regions, key="m2_reg_filter")

    with f3:
        # Search for specific events
        search_query = st.text_input("Search Events (Name, Keywords, Tickers):", placeholder="e.g. Tariffs, Ukraine, Brexit...", key="m2_search")

    # Date range filter
    min_ev_d = events_df["event_date_dt"].min().date()
    max_ev_d = events_df["event_date_dt"].max().date()
    ev_date_range = st.date_input("Filter by Event Date Range:", value=[min_ev_d, max_ev_d], min_value=min_ev_d, max_value=max_ev_d, key="m2_date_filter")

    # Apply all filters
    filtered_events = events_df.copy()
    if selected_cats:
        filtered_events = filtered_events[filtered_events["category"].isin(selected_cats)]

    if selected_region != "All Regions" and "region" in filtered_events.columns:
        filtered_events = filtered_events[filtered_events["region"] == selected_region]

    if isinstance(ev_date_range, (list, tuple)) and len(ev_date_range) == 2:
        d_start, d_end = pd.to_datetime(ev_date_range[0]), pd.to_datetime(ev_date_range[1])
        filtered_events = filtered_events[(filtered_events["event_date_dt"] >= d_start) & (filtered_events["event_date_dt"] <= d_end)]

    if search_query:
        q = search_query.lower()
        search_mask = (
            filtered_events["event_name"].astype(str).str.lower().str.contains(q) |
            filtered_events.get("description", pd.Series([""] * len(filtered_events))).astype(str).str.lower().str.contains(q) |
            filtered_events.get("tags", pd.Series([""] * len(filtered_events))).astype(str).str.lower().str.contains(q) |
            filtered_events.get("related_companies", pd.Series([""] * len(filtered_events))).astype(str).str.lower().str.contains(q)
        )
        filtered_events = filtered_events[search_mask]

    # Metrics summary
    em1, em2, em3, em4 = st.columns(4)
    em1.metric("Matching Global Events", len(filtered_events))
    em2.metric("Categories Represented", filtered_events["category"].nunique())
    em3.metric("Earliest Event", filtered_events["event_date_dt"].min().strftime("%Y-%m-%d") if not filtered_events.empty else "N/A")
    em4.metric("Latest Event", filtered_events["event_date_dt"].max().strftime("%Y-%m-%d") if not filtered_events.empty else "N/A")

    st.markdown("---")

    # 1. Browse Global Events Table
    st.write("### 1. Global Events Directory")
    if filtered_events.empty:
        st.info("No global events match the current filter criteria. Try expanding the date range or clearing the search query.")
        return

    disp_cols = [c for c in ["event_id", "event_name", "category", "subcategory", "event_date", "region", "severity_score", "related_companies", "related_sectors"] if c in filtered_events.columns]
    st.dataframe(filtered_events[disp_cols].sort_values("event_date", ascending=False), use_container_width=True, hide_index=True)

    # 2. View Event Descriptions & Entity Associations
    st.markdown("---")
    st.write("### 2. Detailed Event Inspector & Associated Entities")

    event_choices = filtered_events["event_name"].tolist()
    sel_event_name = st.selectbox("Select Event to Inspect in Detail:", options=event_choices, key="m2_event_select")
    ev_row = filtered_events[filtered_events["event_name"] == sel_event_name].iloc[0]

    card_col1, card_col2 = st.columns([1.6, 1.4])
    with card_col1:
        st.markdown(f"#### {ev_row['event_name']}")
        st.markdown(f"**Category:** `{ev_row.get('category', 'General')}` | **Subcategory:** `{ev_row.get('subcategory', 'N/A')}`")
        st.markdown(f"**Date:** `{pd.to_datetime(ev_row['event_date']).strftime('%Y-%m-%d')}` | **Region:** `{ev_row.get('region', 'Global')}`")

        sev = ev_row.get("severity_score", np.nan)
        if pd.notna(sev):
            st.markdown(f"**Impact Severity Score:** `{sev} / 5.0`")

        desc = ev_row.get("description", ev_row.get("short_description", "No description available."))
        st.info(f"**Description:**\n\n{desc}")

    with card_col2:
        st.markdown("#### Associated Corporate Entities & Sectors")
        rel_comp = ev_row.get("related_companies", "")
        rel_sec = ev_row.get("related_sectors", "")
        tags = ev_row.get("tags", "")
        source_ref = ev_row.get("source_ref", "")

        st.write(f"**Affected Companies / Tickers:**")
        if pd.notna(rel_comp) and str(rel_comp).strip():
            tickers_list = str(rel_comp).split("|")
            st.markdown(" ".join([f"`{t}`" for t in tickers_list]))
        else:
            st.write("*(Broad market impact / Unspecified)*")

        st.write(f"**Affected Sectors:**")
        if pd.notna(rel_sec) and str(rel_sec).strip():
            sectors_list = str(rel_sec).split("|")
            st.markdown(" ".join([f"`{s}`" for s in sectors_list]))
        else:
            st.write("*(All market sectors)*")

        if pd.notna(tags) and str(tags).strip():
            st.write(f"**Tags:** `{tags}`")
        if pd.notna(source_ref) and str(source_ref).strip():
            st.caption(f"**Source Reference:** {source_ref}")

    # 3. View Related News Articles & Sentiment
    st.markdown("---")
    st.write(f"### 3. GDELT News Coverage & Sentiment Dynamics: {sel_event_name}")
    news_df = load_news_articles()

    if not news_df.empty:
        ev_id = ev_row.get("event_id", "")
        ev_date_str = pd.to_datetime(ev_row["event_date"]).strftime("%Y-%m-%d")
        # Match news by event_id, event_name, or _match_event_subset
        news_match = _match_event_subset(news_df, sel_event_name, ev_date_str)
        if news_match.empty:
            news_match = news_df[
                (news_df["matched_event_id"] == ev_id) |
                (news_df["matched_event_name"].astype(str).str.contains(sel_event_name[:15], case=False, na=False))
            ]

        if not news_match.empty:
            n1, n2, n3, n4 = st.columns(4)
            n1.metric("Articles Analyzed", len(news_match))
            avg_tone = news_match["gdelt_tone"].mean() if "gdelt_tone" in news_match.columns else news_match["tone"].mean()
            n2.metric("Average GDELT Tone", f"{avg_tone:.2f}", delta_color="normal")
            dominant_sent = news_match["sentiment_label"].mode().values[0] if "sentiment_label" in news_match.columns else "Neutral"
            n3.metric("Dominant Sentiment", dominant_sent)
            n4.metric("Sources/Publishers", news_match["domain"].nunique() if "domain" in news_match.columns else "N/A")

            # Timing phase and sentiment breakdown chart
            col_n1, col_n2 = st.columns(2)
            with col_n1:
                if "timing_phase" in news_match.columns:
                    timing_counts = news_match["timing_phase"].value_counts().reset_index()
                    timing_counts.columns = ["Timing Phase", "Articles"]
                    time_fig = px.bar(
                        timing_counts,
                        x="Timing Phase",
                        y="Articles",
                        title="News Distribution Across Event Timing Phases",
                        color="Timing Phase",
                        color_discrete_map={"pre-event": "#636EFA", "day-of": "#EF553B", "post-event": "#00CC96"}
                    )
                    time_fig.update_layout(height=300)
                    st.plotly_chart(time_fig, use_container_width=True)

            with col_n2:
                if "sentiment_label" in news_match.columns:
                    sent_counts = news_match["sentiment_label"].value_counts().reset_index()
                    sent_counts.columns = ["Sentiment", "Articles"]
                    sent_fig = px.pie(
                        sent_counts,
                        names="Sentiment",
                        values="Articles",
                        title="News Sentiment Label Breakdown",
                        color="Sentiment",
                        color_discrete_map={"Positive": "#2ca02c", "Neutral": "#7f7f7f", "Negative": "#d62728"}
                    )
                    sent_fig.update_layout(height=300)
                    st.plotly_chart(sent_fig, use_container_width=True)

            # News articles table
            st.write("**Related News Articles Table**")
            news_disp_cols = [c for c in ["published_date", "title", "domain", "sourcecountry", "sentiment_label", "gdelt_tone", "polarity", "timing_phase"] if c in news_match.columns]
            st.dataframe(news_match[news_disp_cols].sort_values("published_date", ascending=True), use_container_width=True, hide_index=True)
        else:
            st.info(f"No specific GDELT news records indexed for '{sel_event_name}' in the sample collection. Showing category news:")
            cat_news = news_df[news_df["event_category"] == ev_row.get("category")].head(5)
            if not cat_news.empty:
                st.dataframe(cat_news[[c for c in ["published_date", "title", "domain", "sentiment_label", "gdelt_tone"] if c in cat_news.columns]], use_container_width=True, hide_index=True)


# =============================================================================
# MODULE 3: EVENT IMPACT ANALYZER
# =============================================================================

def render_event_impact_analyzer(index_frames: Dict[str, pd.DataFrame], vix_df: pd.DataFrame):
    st.subheader("Module 3: Event Impact Analyzer")
    st.caption(
        "Quantitative econometric event study engine: Analyze stock/index price movements, pre/post returns, "
        "drawdowns, Market Model abnormal returns (AR/CAR), VIX volatility, multi-event comparisons, and intra-sector dispersion."
    )

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
            avail_tickers = [f.replace("_raw.csv", "") for f in os.listdir(RAW_DIR / "stocks") if f.endswith("_raw.csv")]
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

    ev_options = [
        f"{r['event_name']} ({pd.to_datetime(r['event_date']).strftime('%Y-%m-%d')})"
        for _, r in ev_pool.sort_values("event_date", ascending=False).iterrows()
    ]
    if not ev_options:
        st.info("No events match the selected category filter.")
        return

    # Default to premier shock (e.g. Russia-Ukraine) when present so all panels populate immediately
    default_ev_idx = 0
    for idx_opt, opt_name in enumerate(ev_options):
        if "Russia" in opt_name and "Invasion" in opt_name:
            default_ev_idx = idx_opt
            break

    sel_ev_full = st.selectbox("3. Select Specific Event to Analyze:", options=ev_options, index=default_ev_idx, key="m3_event_select")
    sel_ev_name = sel_ev_full.rsplit(" (", 1)[0].strip() if " (" in sel_ev_full else sel_ev_full
    ev_matches = events_df[events_df["event_name"] == sel_ev_name]
    ev_row = ev_matches.iloc[0] if len(ev_matches) > 0 else events_df.iloc[0]
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
        vix_dates = vix_df["Date"].dt.strftime("%Y-%m-%d").tolist()
        v_idx = find_effective_trading_day_index(ev_date_str, vix_dates)
        if v_idx is not None:
            v_p0 = vix_df.iloc[v_idx]["Close"]
            vix_t0_val = v_p0
            if (v_idx + 5) < len(vix_df):
                v_p5 = vix_df.iloc[v_idx + 5]["Close"]
                vix_5d_chg = v_p5 - v_p0

    # 1. Display 5 Key Metrics Cards & Trajectory
    st.markdown("---")
    st.write(f"### 1. Historical Event Metrics & Price Trajectory: {target_symbol} around {sel_ev_name}")

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Pre-Event 5D Return", _format_num(pre_5d_val, "%"), delta_color="normal")
    mc2.metric("Post-Event 5D Return", _format_num(post_5d_val, "%"), delta_color="normal")
    mc3.metric("Post-Event 30D Return", _format_num(post_30d_val, "%"), delta_color="normal")
    mc4.metric("30D Max Drawdown", _format_num(max_dd_val, "%"), delta_color="inverse")
    mc5.metric("VIX at Event (T0)", _format_num(vix_t0_val), delta=_format_num(vix_5d_chg, " (5D)"), delta_color="inverse")

    # 5. Event Window Trajectory Chart [-30 to +30] with VIX Overlay
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
                line=dict(color="#1f77b4", width=2.5)
            ), secondary_y=False)

            # If Company, also overlay Parent S&P 500
            if target_mode == "Individual Company":
                sp500_df = index_frames["sp500"]
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
                            line=dict(color="#7f7f7f", dash="dash")
                        ), secondary_y=False)

            # VIX Overlay
            vix_win = vix_df[vix_df["Date"].isin(win_df["Date"])].dropna(subset=["Close"])
            if not vix_win.empty:
                traj_fig.add_trace(go.Scatter(
                    x=vix_win["Date"],
                    y=vix_win["Close"],
                    mode="lines",
                    name="VIX Level",
                    line=dict(color="#e377c2", dash="dot")
                ), secondary_y=True)

            traj_fig.add_vline(x=target_price_df.iloc[t0_idx]["Date"], line_dash="dash", line_color="crimson", annotation_text="T0 Event")
            traj_fig.add_hline(y=100, line_dash="dot", line_color="gray", secondary_y=False)

            traj_fig.update_layout(
                title=f"Event Window Dynamics: {target_symbol} around {sel_ev_name} (±30 Trading Days)",
                hovermode="x unified",
                height=440
            )
            traj_fig.update_yaxes(title_text="Normalized Price (T0 = 100)", secondary_y=False)
            traj_fig.update_yaxes(title_text="VIX Level", secondary_y=True)
            st.plotly_chart(traj_fig, use_container_width=True)

    # 2. Abnormal Returns & Cumulative Abnormal Returns (CAR)
    st.markdown("---")
    st.write(f"### 2. Econometric Abnormal Returns (AR) & Cumulative Abnormal Returns (CAR)")
    st.caption("Benchmark Market Model estimated over [T0-120, T0-11] clean pre-event window against the S&P 500 index.")

    sp_benchmark = index_frames["sp500"]
    ar_res = calculate_abnormal_returns(
        security_df=target_price_df,
        market_df=sp_benchmark,
        event_date=ev_date_str
    ) if target_price_df is not None else {}

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
            car_plot.update_layout(height=360, hovermode="x unified")
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
    st.write(f"### 3. Multi-Event Comparative Performance: {target_symbol}")
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
        comp_ev_fig.update_layout(title=f"Multi-Event Price Trajectories: {target_symbol} (T0 = 100)", xaxis_title="Relative Trading Days (T0 = 0)", yaxis_title="Normalized Price", height=400, hovermode="x unified")
        st.plotly_chart(comp_ev_fig, use_container_width=True)

        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # 4. Compare Companies and Sectors Affected by the Same Event
    st.markdown("---")
    st.write(f"### 4. Cross-Sector & Intra-Sector Company Impact: {sel_ev_name}")

    col_sec1, col_sec2 = st.columns(2)

    with col_sec1:
        st.write("**Sector ETF Returns Around Event**")
        sec_ev_sub = _match_event_subset(sector_events_df, sel_ev_name, ev_date_str)
        if not sec_ev_sub.empty:
            s_melt = sec_ev_sub.melt(
                id_vars=["sector", "ticker"],
                value_vars=["pre_5d_return_pct", "post_5d_return_pct", "post_30d_return_pct"],
                var_name="window",
                value_name="ret_pct"
            )
            s_melt["Window"] = s_melt["window"].map({
                "pre_5d_return_pct": "Pre-5D %",
                "post_5d_return_pct": "Post-5D %",
                "post_30d_return_pct": "Post-30D %"
            })
            s_fig = px.bar(
                s_melt,
                x="sector",
                y="ret_pct",
                color="Window",
                barmode="group",
                title=f"8 Sector ETFs Performance ({sel_ev_name})"
            )
            s_fig.add_hline(y=0, line_color="gray")
            s_fig.update_layout(height=360, xaxis_tickangle=-20)
            st.plotly_chart(s_fig, use_container_width=True)
        else:
            st.info(f"Sector event study data not available for '{sel_ev_name}'.")

    with col_sec2:
        st.write("**Intra-Sector Dispersion Range**")
        disp_df = load_cross_company_dispersion()
        disp_sub = _match_event_subset(disp_df, sel_ev_name, ev_date_str)
        if not disp_sub.empty:
            d_fig = px.bar(
                disp_sub.sort_values("return_range_pct", ascending=True),
                x="return_range_pct",
                y="sector",
                orientation="h",
                title=f"Intra-Sector Return Dispersion Range % ({sel_ev_name})",
                color="return_range_pct",
                color_continuous_scale="Purples"
            )
            d_fig.update_layout(height=360)
            st.plotly_chart(d_fig, use_container_width=True)
        else:
            st.info(f"Cross-company dispersion data not available for '{sel_ev_name}'.")

    # Company response table for this event
    comp_ev_sub = _match_event_subset(comp_events_df, sel_ev_name, ev_date_str)
    if not comp_ev_sub.empty:
        st.write(f"**Company Cross-Sectional Performance Table ({len(comp_ev_sub)} Equities):**")
        st.dataframe(comp_ev_sub[[
            "ticker", "company_name", "sector", "price_t0",
            "pre_5d_return_pct", "post_5d_return_pct", "post_30d_return_pct",
            "spread_vs_sp500_5d_pct", "estimated_beta", "car_post_5d_pct"
        ]].rename(columns={
            "ticker": "Ticker", "company_name": "Company", "sector": "Sector", "price_t0": "Close T0",
            "pre_5d_return_pct": "Pre-5D %", "post_5d_return_pct": "Post-5D %", "post_30d_return_pct": "Post-30D %",
            "spread_vs_sp500_5d_pct": "Spread vs SPX %", "estimated_beta": "Beta", "car_post_5d_pct": "CAR +5D %"
        }), use_container_width=True, hide_index=True)

    # 5. Related News Coverage & Sentiment
    st.markdown("---")
    st.write(f"### 5. Related GDELT News Coverage & Sentiment: {sel_ev_name}")
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


# =============================================================================
# MAIN CONTROLLER & APPLICATION ENTRY POINT
# =============================================================================

def main():
    st.set_page_config(
        page_title="Impact of Global Events on the US Stock Market",
        page_icon="📈",
        layout="wide",
    )

    st.title("Impact of Global Events on the US Stock Market")
    st.caption("Big Data Analytics Platform: Multi-Asset Preprocessing, Econometric Event Studies & Point-in-Time Macro Alignment")

    # Load core datasets
    with st.spinner("Initializing platform datasets..."):
        index_frames = {k: load_index_data(k) for k in INDEX_CONFIG}
        vix_df = load_vix_data()
        event_df = load_event_study()

    # Verify baseline market datasets exist
    missing_core = [k for k, df in index_frames.items() if df is None]
    if missing_core or vix_df is None:
        st.warning("Processed market datasets are missing. Run the full pipeline via python main.py first.")
        st.code("python main.py", language="powershell")
        return

    # Sidebar: Platform Information & Global Settings
    st.sidebar.title("BDA Platform")
    st.sidebar.markdown("**Big Data Analytics Platform**")
    st.sidebar.caption("PySpark SQL · PyArrow · Streamlit · Plotly")

    nav_mode = st.sidebar.radio(
        "Navigate Module:",
        options=[
            "All Modules (Tabs)",
            "Module 1: Market Explorer",
            "Module 2: Global Event Explorer",
            "Module 3: Event Impact Analyzer"
        ],
        index=0,
        key="global_nav_mode"
    )

    st.sidebar.markdown("---")
    st.sidebar.write("**Storage Architecture:**")
    st.sidebar.caption("• Layer 1: data/raw/ (Immutable raw ingestion)")
    st.sidebar.caption("• Layer 2: data/processed/ (Partitioned Parquet)")
    st.sidebar.caption("• Layer 3: data/analytics/ (Econometric event studies)")

    show_debug_table = st.sidebar.checkbox("Show Recent Processed S&P 500 Records", value=False)

    # Render Modules via Tabs or Direct Selection
    if nav_mode == "All Modules (Tabs)":
        tab1, tab2, tab3 = st.tabs([
            "📈 MODULE 1: MARKET EXPLORER",
            "🌍 MODULE 2: GLOBAL EVENT EXPLORER",
            "💥 MODULE 3: EVENT IMPACT ANALYZER"
        ])
        with tab1:
            render_market_explorer(index_frames, vix_df)
        with tab2:
            render_global_event_explorer()
        with tab3:
            render_event_impact_analyzer(index_frames, vix_df)

    elif nav_mode == "Module 1: Market Explorer":
        render_market_explorer(index_frames, vix_df)
    elif nav_mode == "Module 2: Global Event Explorer":
        render_global_event_explorer()
    elif nav_mode == "Module 3: Event Impact Analyzer":
        render_event_impact_analyzer(index_frames, vix_df)

    if show_debug_table and index_frames.get("sp500") is not None:
        st.markdown("---")
        st.subheader("Recent Processed S&P 500 Records")
        st.dataframe(index_frames["sp500"].tail(20), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
