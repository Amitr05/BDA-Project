"""
Streamlit Dashboard: Multi-Asset Global Event & US Stock Market Analytics Platform
==================================================================================
Institutional Financial Intelligence Platform featuring 8 dedicated analytical pages:
  1. Home / Overview
  2. Market Explorer
  3. Global Event Explorer
  4. Event Impact Analyzer
  5. Sector Analysis
  6. ML Predictions
  7. Advanced Analytics
  8. Macro & News

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
import streamlit as st

# Resolve paths from repository root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


from dashboard.components.event_matcher import match_event_subset, _match_event_subset
from dashboard.components.metrics import format_num, _format_num


# =============================================================================
# BACKWARD COMPATIBLE RENDERING FUNCTION ALIASES
# =============================================================================

def render_market_explorer(index_frames: Dict[str, pd.DataFrame], vix_df: pd.DataFrame):
    """Backwards compatibility shim for existing unit tests."""
    from dashboard.pages.Market_Explorer import render_market_explorer_page
    render_market_explorer_page()


def render_global_event_explorer():
    """Backwards compatibility shim for existing unit tests."""
    from dashboard.pages.Global_Event_Explorer import render_global_event_explorer_page
    render_global_event_explorer_page()


def render_event_impact_analyzer(index_frames: Dict[str, pd.DataFrame], vix_df: pd.DataFrame):
    """Backwards compatibility shim for existing unit tests."""
    from dashboard.pages.Event_Impact_Analyzer import render_event_impact_analyzer_page
    render_event_impact_analyzer_page()


def render_conflict_impact_predictor():
    """Backwards compatibility shim for existing unit tests."""
    from dashboard.pages.ML_Predictions import render_ml_predictions_page
    render_ml_predictions_page()


# =============================================================================
# MULTI-PAGE APPLICATION CONTROLLER (Streamlit Native st.navigation)
# =============================================================================

def main():
    st.set_page_config(
        page_title="Impact of Global Events on the US Stock Market",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Global custom styling for modern institutional aesthetics
    st.markdown(
        """
        <style>
            /* Clean Metric card styling */
            [data-testid="stMetric"] {
                background: rgba(128, 128, 128, 0.05);
                border: 1px solid rgba(128, 128, 128, 0.15);
                border-radius: 8px;
                padding: 12px 16px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }
            [data-testid="stMetricLabel"] {
                font-weight: 600;
                font-size: 0.85rem;
                color: #64748b;
            }
            [data-testid="stMetricValue"] {
                font-weight: 700;
                font-size: 1.4rem;
            }
            /* Clean section dividers */
            hr {
                margin: 1.25rem 0;
                border-color: rgba(128, 128, 128, 0.2);
            }
            /* Subtle sidebar polish */
            [data-testid="stSidebar"] {
                border-right: 1px solid rgba(128, 128, 128, 0.15);
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar: Platform Information & Storage Architecture
    st.sidebar.markdown("### 📈 BDA Financial Intelligence")
    st.sidebar.caption("Big Data Analytics & Econometric Engine")
    st.sidebar.markdown("---")

    # Native Multi-page Navigation Definition
    nav_pages = {
        "Overview": [
            st.Page("pages/Home.py", title="Home / Overview", icon="🏠", default=True),
        ],
        "Market & Events": [
            st.Page("pages/Market_Explorer.py", title="Market Explorer", icon="📈"),
            st.Page("pages/Global_Event_Explorer.py", title="Global Event Explorer", icon="🌍"),
            st.Page("pages/Event_Impact_Analyzer.py", title="Event Impact Analyzer", icon="💥"),
        ],
        "Specialized Intelligence": [
            st.Page("pages/Sector_Analysis.py", title="Sector Analysis", icon="🏢"),
            st.Page("pages/ML_Predictions.py", title="ML Predictions", icon="🤖"),
            st.Page("pages/Advanced_Analytics.py", title="Advanced Analytics", icon="🔬"),
            st.Page("pages/Macro_and_News.py", title="Macro & News", icon="📊"),
        ]
    }

    pg = st.navigation(nav_pages)

    # Sidebar metadata badges
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Storage Layer Architecture:**")
    st.sidebar.caption("• **Layer 1:** `data/raw/` (Immutable Raw)")
    st.sidebar.caption("• **Layer 2:** `data/processed/` (Spark Parquet)")
    st.sidebar.caption("• **Layer 3:** `data/analytics/` (Econometric Models)")
    st.sidebar.markdown("---")
    st.sidebar.caption("Python 3.11 · PySpark · PyArrow · Streamlit · Plotly · scikit-learn")

    # Execute the selected page
    pg.run()


if __name__ == "__main__":
    main()
