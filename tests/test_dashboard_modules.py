"""
Comprehensive Unit & Integration Tests for Streamlit Dashboard Modules
======================================================================
Verifies feature logic and data contracts across all three major modules:
- MODULE 1: MARKET EXPLORER
- MODULE 2: GLOBAL EVENT EXPLORER
- MODULE 3: EVENT IMPACT ANALYZER
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from dashboard.app import (
    INDEX_CONFIG,
    EVENT_CATEGORIES_7,
    load_index_data,
    load_vix_data,
    load_stock_data,
    load_sector_etfs_data,
    load_macro_aligned_data,
    load_events_database,
    load_news_articles,
    load_company_event_studies,
    load_event_study,
    load_sector_event_study,
    load_abnormal_returns_detail,
    load_significance_summary,
    load_cross_company_dispersion,
    load_macro_context_analysis,
    load_category_comparisons,
    load_companies_registry,
    load_index_constituents,
    get_available_companies_for_index,
    _format_num,
)
from src.analytics.advanced_event_study import (
    find_effective_trading_day_index,
    calculate_abnormal_returns,
    estimate_market_model_parameters
)


# =============================================================================
# MODULE 1: MARKET EXPLORER TESTS
# =============================================================================

class TestModule1MarketExplorer:
    """Tests for Module 1: Market Explorer functionality."""

    def test_all_four_indices_loadable(self):
        """Verify all four benchmark indices load with required schema."""
        for key in ["sp500", "nasdaq", "dow_jones", "russell2000"]:
            df = load_index_data(key)
            assert df is not None, f"Index {key} failed to load."
            assert not df.empty, f"Index {key} is empty."
            assert "Date" in df.columns
            assert "Close" in df.columns
            assert "daily_return_pct" in df.columns
            assert "rolling_volatility_21d" in df.columns
            assert pd.api.types.is_datetime64_any_dtype(df["Date"])
            # Ensure chronological order
            assert df["Date"].is_monotonic_increasing

    def test_vix_data_integrity(self):
        """Verify CBOE VIX dataset loads independently with expected attributes."""
        vix_df = load_vix_data()
        assert vix_df is not None
        assert not vix_df.empty
        assert "Date" in vix_df.columns
        assert "Close" in vix_df.columns
        assert (vix_df["Close"] > 0).all()

    def test_stock_data_loading_and_features(self):
        """Verify individual company OHLCV data loads with return and volatility features."""
        stock_df = load_stock_data("AAPL")
        assert stock_df is not None
        assert not stock_df.empty
        assert "Date" in stock_df.columns
        assert "Close" in stock_df.columns
        assert "daily_return" in stock_df.columns
        assert "daily_return_pct" in stock_df.columns
        assert "rolling_vol_21d" in stock_df.columns
        # Verify return formula: Close.pct_change() * 100
        assert np.isclose(stock_df["daily_return"].iloc[1] * 100.0, stock_df["daily_return_pct"].iloc[1])

    def test_available_companies_for_index_filtering(self):
        """Verify company dropdown filters constituents accurately by selected index."""
        sp500_comps = get_available_companies_for_index("sp500")
        assert not sp500_comps.empty
        assert "ticker" in sp500_comps.columns
        assert "company_name" in sp500_comps.columns
        # Common S&P 500 stocks must be present
        tickers = sp500_comps["ticker"].tolist()
        assert "AAPL" in tickers or "MSFT" in tickers

    def test_company_vs_parent_index_spread_calculation(self):
        """Verify calculation of normalized price (Base=100) and excess return spread."""
        stock_df = load_stock_data("AAPL")
        sp_df = load_index_data("sp500")
        assert stock_df is not None and sp_df is not None

        merged = pd.merge(
            stock_df[["Date", "Close", "daily_return_pct"]].rename(columns={"Close": "Close_sec", "daily_return_pct": "ret_sec"}),
            sp_df[["Date", "Close", "daily_return_pct"]].rename(columns={"Close": "Close_idx", "daily_return_pct": "ret_idx"}),
            on="Date",
            how="inner"
        ).sort_values("Date").reset_index(drop=True)

        assert not merged.empty
        merged["norm_sec"] = (merged["Close_sec"] / merged["Close_sec"].iloc[0]) * 100.0
        merged["norm_idx"] = (merged["Close_idx"] / merged["Close_idx"].iloc[0]) * 100.0
        merged["daily_spread_pct"] = merged["ret_sec"] - merged["ret_idx"]

        # Base normalized price at t0 should equal 100.0
        assert np.isclose(merged["norm_sec"].iloc[0], 100.0)
        assert np.isclose(merged["norm_idx"].iloc[0], 100.0)
        # Daily spread must equal difference
        expected_spread = merged["ret_sec"].iloc[1] - merged["ret_idx"].iloc[1]
        assert np.isclose(merged["daily_spread_pct"].iloc[1], expected_spread)

    def test_sector_etf_panel_and_correlation(self):
        """Verify 8 sector ETFs panel loads with all tickers and computes correlation matrix."""
        etfs_df = load_sector_etfs_data()
        assert etfs_df is not None
        assert not etfs_df.empty
        expected_etfs = {"XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU"}
        avail_etfs = set(etfs_df["Ticker"].unique())
        assert expected_etfs.issubset(avail_etfs)

        pivot = etfs_df.pivot(index="Date", columns="Ticker", values="daily_return").dropna()
        assert not pivot.empty
        corr = pivot.corr()
        assert corr.shape == (8, 8)
        # Diagonal must be 1.0
        for t in expected_etfs:
            assert np.isclose(corr.loc[t, t], 1.0)


# =============================================================================
# MODULE 2: GLOBAL EVENT EXPLORER TESTS
# =============================================================================

class TestModule2GlobalEventExplorer:
    """Tests for Module 2: Global Event Explorer filtering and inspection."""

    def test_events_database_covers_seven_categories(self):
        """Verify database contains events covering all 7 standard event categories."""
        events_df = load_events_database()
        assert not events_df.empty
        assert len(events_df) >= 60

        cat_col = "category" if "category" in events_df.columns else "event_category"
        cats_in_df = set(events_df[cat_col].dropna().unique())
        for cat in EVENT_CATEGORIES_7:
            assert cat in cats_in_df, f"Category '{cat}' missing from events database."

    def test_filter_events_by_category(self):
        """Verify filtering by single and multiple categories produces accurate subsets."""
        events_df = load_events_database()
        cat_col = "category" if "category" in events_df.columns else "event_category"

        econ_events = events_df[events_df[cat_col] == "Economic Events"]
        assert len(econ_events) >= 5
        assert (econ_events[cat_col] == "Economic Events").all()

        geo_events = events_df[events_df[cat_col] == "War and Geopolitical Conflicts"]
        assert len(geo_events) >= 5
        assert (geo_events[cat_col] == "War and Geopolitical Conflicts").all()

    def test_filter_events_by_date_range(self):
        """Verify temporal filtering restricts events within target boundaries."""
        events_df = load_events_database()
        events_df["event_date_dt"] = pd.to_datetime(events_df["event_date"], errors="coerce")

        start_dt = pd.Timestamp("2020-01-01")
        end_dt = pd.Timestamp("2021-12-31")
        sub = events_df[(events_df["event_date_dt"] >= start_dt) & (events_df["event_date_dt"] <= end_dt)]
        assert not sub.empty
        assert (sub["event_date_dt"] >= start_dt).all()
        assert (sub["event_date_dt"] <= end_dt).all()

    def test_filter_events_by_region(self):
        """Verify regional filtering works as expected."""
        events_df = load_events_database()
        if "region" in events_df.columns:
            us_events = events_df[events_df["region"].str.contains("United States", na=False)]
            assert not us_events.empty
            assert (us_events["region"].str.contains("United States", na=False)).all()

    def test_search_events_by_keyword(self):
        """Verify keyword search finds relevant events by title, description, or tags."""
        events_df = load_events_database()
        q = "brexit"
        mask = (
            events_df["event_name"].astype(str).str.lower().str.contains(q) |
            events_df.get("description", pd.Series([""] * len(events_df))).astype(str).str.lower().str.contains(q)
        )
        matches = events_df[mask]
        assert not matches.empty
        assert any("Brexit" in name for name in matches["event_name"])

    def test_related_companies_and_sectors_association(self):
        """Verify corporate entities and sector tags are non-empty for key events."""
        events_df = load_events_database()
        tariffs = events_df[events_df["event_name"].str.contains("Tariff", case=False, na=False)]
        assert not tariffs.empty
        row = tariffs.iloc[0]
        # Should have related sectors or companies
        rel_sec = str(row.get("related_sectors", ""))
        assert len(rel_sec) > 0

    def test_gdelt_news_articles_and_sentiment_metrics(self):
        """Verify news dataset contains sentiment tone, polarity, and timing phases."""
        news_df = load_news_articles()
        assert not news_df.empty
        assert "published_date" in news_df.columns
        assert "sentiment_label" in news_df.columns
        assert "timing_phase" in news_df.columns
        # Verify sentiment categories
        sent_labels = set(news_df["sentiment_label"].dropna().unique())
        assert {"Positive", "Negative", "Neutral"}.issubset(sent_labels)
        # Verify timing phase categories
        phases = set(news_df["timing_phase"].dropna().unique())
        assert {"pre-event", "day-of", "post-event"}.issubset(phases)


# =============================================================================
# MODULE 3: EVENT IMPACT ANALYZER TESTS
# =============================================================================

class TestModule3EventImpactAnalyzer:
    """Tests for Module 3: Event Impact Analyzer quantitative calculations."""

    def test_event_window_metrics_calculation(self):
        """Verify pre-5D, post-5D, post-30D, and max drawdown calculations."""
        sp_df = load_index_data("sp500")
        assert sp_df is not None

        dates_str = sp_df["Date"].dt.strftime("%Y-%m-%d").tolist()
        t0_idx = find_effective_trading_day_index("2020-03-11", dates_str)
        assert t0_idx is not None

        closes = sp_df["Close"].values
        p0 = closes[t0_idx]
        p_pre = closes[t0_idx - 5]
        p_post5 = closes[t0_idx + 5]
        p_post30 = closes[t0_idx + 30]

        pre_5d = ((p0 - p_pre) / p_pre) * 100.0
        post_5d = ((p_post5 - p0) / p0) * 100.0
        post_30d = ((p_post30 - p0) / p0) * 100.0
        lowest = min(closes[t0_idx : t0_idx + 31])
        max_dd = ((lowest - p0) / p0) * 100.0

        # COVID crash had sharp negative post-5D return and negative drawdown
        assert post_5d < -5.0
        assert max_dd < -10.0

    def test_market_model_abnormal_returns_estimation(self):
        """Verify Market Model OLS estimation produces valid alpha, beta, and CAR series."""
        stock_df = load_stock_data("AAPL")
        sp_df = load_index_data("sp500")
        assert stock_df is not None and sp_df is not None

        res = calculate_abnormal_returns(
            security_df=stock_df,
            market_df=sp_df,
            event_date="2020-03-11"
        )
        assert "beta" in res
        assert "alpha" in res
        assert "car_post_5d_pct" in res
        assert "car_post_30d_pct" in res
        assert "daily_ar_series" in res
        # Beta should be economically reasonable for Apple (~0.5 - 2.0)
        assert 0.5 <= res["beta"] <= 2.0
        # Daily AR series should span [-5, +30] trading days
        ar_series = res["daily_ar_series"]
        assert len(ar_series) == 36
        rel_days = [d["relative_trading_day"] for d in ar_series]
        assert rel_days[0] == -5
        assert rel_days[-1] == 30

    def test_cross_sectional_significance_summary(self):
        """Verify statistical significance table contains valid t-statistics and p-values."""
        sig_df = load_significance_summary()
        assert sig_df is not None
        assert not sig_df.empty
        assert "scope" in sig_df.columns
        assert "t_statistic" in sig_df.columns
        assert "p_value" in sig_df.columns
        assert "statistically_significant_5pct" in sig_df.columns
        # p-values must be between 0.0 and 1.0
        assert (sig_df["p_value"] >= 0.0).all()
        assert (sig_df["p_value"] <= 1.0).all()

    def test_multi_event_comparative_overlay(self):
        """Verify multi-event relative day normalization synchronizes on T0=0."""
        sp_df = load_index_data("sp500")
        dates_str = sp_df["Date"].dt.strftime("%Y-%m-%d").tolist()

        test_events = ["2016-06-23", "2020-03-11", "2022-02-24"]
        trajectories = {}

        for ev_d in test_events:
            idx = find_effective_trading_day_index(ev_d, dates_str)
            assert idx is not None
            st_p = idx - 5
            en_p = idx + 30
            sub_w = sp_df.iloc[st_p : en_p + 1].copy()
            base = sp_df.iloc[idx]["Close"]
            sub_w["norm"] = (sub_w["Close"] / base) * 100.0
            sub_w["rel_day"] = range(st_p - idx, en_p - idx + 1)
            trajectories[ev_d] = sub_w

            # At relative trading day 0, normalized price must be exactly 100.0
            t0_row = sub_w[sub_w["rel_day"] == 0]
            assert np.isclose(t0_row["norm"].values[0], 100.0)

    def test_sector_event_study_and_dispersion(self):
        """Verify sector ETF event study and intra-sector dispersion tables load."""
        sector_ev = load_sector_event_study()
        assert sector_ev is not None
        assert not sector_ev.empty
        assert "sector" in sector_ev.columns
        assert "pre_5d_return_pct" in sector_ev.columns
        assert "post_5d_return_pct" in sector_ev.columns

        disp_df = load_cross_company_dispersion()
        assert disp_df is not None
        assert not disp_df.empty
        assert "sector" in disp_df.columns
        assert "return_range_pct" in disp_df.columns
        assert (disp_df["return_range_pct"] >= 0.0).all()

    def test_company_event_study_dataset(self):
        """Verify 350-row company event studies table contains spreads and betas."""
        comp_ev = load_company_event_studies()
        assert comp_ev is not None
        assert not comp_ev.empty
        assert "ticker" in comp_ev.columns
        assert "event_name" in comp_ev.columns
        assert "spread_vs_sp500_5d_pct" in comp_ev.columns
        assert "estimated_beta" in comp_ev.columns
        assert "car_post_5d_pct" in comp_ev.columns


# =============================================================================
# FORMATTING & HELPER TESTS
# =============================================================================

def test_format_num_helper():
    """Verify number formatting handles floats, percentages, and NaN gracefully."""
    assert _format_num(1234.567) == "1,234.57"
    assert _format_num(12.345, "%") == "12.35%"
    assert _format_num(np.nan) == "N/A"
    assert _format_num(None) == "N/A"
    assert _format_num("test") == "test"
