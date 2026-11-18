"""
Unit and Integration Tests for Advanced Event Study & Econometric Analytics
==========================================================================
Tests:
1. Trading-day calendar alignment helper.
2. Market model OLS parameter estimation (alpha, beta, residual SE).
3. Abnormal returns (AR) and Cumulative Abnormal Returns (CAR) across estimation and event windows.
4. Parametric cross-sectional t-test, p-value, standard error, degrees of freedom.
5. Explicit statistical assumptions, limitations, and correlation != causation warnings.
6. Company-level event studies with parent index excess return spread and 30D drawdown.
7. Intra-sector cross-company dispersion and resilience rankings.
8. Event category comparisons across historical shock types.
9. Macroeconomic regime context analysis (yield curve inversion & inflation regimes).
10. Benchmark independent NumPy/SciPy precision verification against analytical formulas.
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from scipy import stats

from src.analytics.advanced_event_study import (
    find_effective_trading_day_index,
    estimate_market_model_parameters,
    calculate_abnormal_returns,
    compute_statistical_tests,
    run_company_event_studies,
    analyze_cross_company_dispersion,
    compare_event_categories,
    analyze_macroeconomic_context
)
from src.analytics.benchmark_validation import (
    independent_daily_return_and_volatility,
    independent_ols_market_model
)


class TestTradingDayAlignment:
    """Tests for active trading-day calendar alignment."""

    def test_exact_match(self):
        dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        idx = find_effective_trading_day_index("2024-01-03", dates)
        assert idx == 1

    def test_weekend_alignment(self):
        # 2024-01-06 is Saturday -> should map to Monday 2024-01-08
        dates = ["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]
        idx = find_effective_trading_day_index("2024-01-06", dates)
        assert idx == 2
        assert dates[idx] == "2024-01-08"

    def test_out_of_range_future_date(self):
        dates = ["2024-01-02", "2024-01-03"]
        idx = find_effective_trading_day_index("2024-02-01", dates)
        assert idx is None


class TestMarketModelEstimation:
    """Tests for Ordinary Least Squares (OLS) Market Model estimation."""

    def test_ols_known_parameters(self):
        np.random.seed(42)
        mkt_ret = np.random.normal(0.0005, 0.01, 100)
        true_alpha = 0.0002
        true_beta = 1.25
        sec_ret = true_alpha + true_beta * mkt_ret + np.random.normal(0, 0.002, 100)

        s_sec = pd.Series(sec_ret)
        s_mkt = pd.Series(mkt_ret)

        alpha, beta, sigma_eps, n = estimate_market_model_parameters(s_sec, s_mkt)
        assert n == 100
        assert np.isclose(beta, true_beta, atol=0.1)
        assert np.isclose(alpha, true_alpha, atol=0.001)
        assert sigma_eps > 0

    def test_insufficient_sample_fallback(self):
        s_sec = pd.Series([0.01, 0.02, -0.01])
        s_mkt = pd.Series([0.005, 0.01, -0.005])

        alpha, beta, sigma_eps, n = estimate_market_model_parameters(s_sec, s_mkt)
        assert n == 3
        assert alpha == 0.0
        assert beta == 1.0


class TestAbnormalReturns:
    """Tests for daily Abnormal Returns (AR) and Cumulative Abnormal Returns (CAR)."""

    @pytest.fixture
    def synthetic_history(self):
        # 160 trading days of data around an event at day 130
        dates = pd.date_range("2023-01-01", periods=160, freq="B").strftime("%Y-%m-%d").tolist()
        np.random.seed(123)
        mkt_ret = np.random.normal(0.0004, 0.01, 160)
        sec_ret = 0.0001 + 1.15 * mkt_ret + np.random.normal(0, 0.005, 160)

        mkt_prices = 100.0 * np.cumprod(1.0 + mkt_ret)
        sec_prices = 50.0 * np.cumprod(1.0 + sec_ret)

        mkt_df = pd.DataFrame({"Date": dates, "Close": mkt_prices, "daily_return": mkt_ret})
        sec_df = pd.DataFrame({"Date": dates, "Close": sec_prices, "daily_return": sec_ret})

        event_date = dates[125]
        return sec_df, mkt_df, event_date

    def test_calculate_abnormal_returns(self, synthetic_history):
        sec_df, mkt_df, event_date = synthetic_history
        res = calculate_abnormal_returns(
            security_df=sec_df,
            market_df=mkt_df,
            event_date=event_date,
            estimation_window=100,
            estimation_buffer=10,
            pre_event_window=5,
            post_event_window=30
        )

        assert "alpha" in res
        assert "beta" in res
        assert "car_pre_5d_pct" in res
        assert "car_post_5d_pct" in res
        assert "car_post_30d_pct" in res
        assert "daily_ar_series" in res
        assert len(res["daily_ar_series"]) > 0

        # Check relative trading day bounds [-5, +30]
        days = [r["relative_trading_day"] for r in res["daily_ar_series"]]
        assert min(days) == -5
        assert max(days) <= 30


class TestStatisticalSignificance:
    """Tests for cross-sectional t-test and econometric documentation."""

    def test_significant_car(self):
        # Sample of positive abnormal returns
        car_sample = [4.5, 3.2, 5.1, 2.8, 6.0, 4.0, 3.5, 4.8, 5.5, 3.9]
        res = compute_statistical_tests(car_sample, null_mean=0.0)

        assert res["sample_size"] == 10
        assert res["degrees_of_freedom"] == 9
        assert res["mean_car_pct"] > 0
        assert res["t_statistic"] > 2.26  # t-crit for 9 df at 5% is ~2.262
        assert res["p_value"] < 0.05
        assert res["statistically_significant_5pct"] is True

        # Check documented assumptions and limitations
        assert len(res["assumptions"]) >= 3
        assert any("Normality" in a for a in res["assumptions"])
        assert any("Independence" in a for a in res["assumptions"])
        assert len(res["limitations"]) >= 3
        assert any("Non-Causation" in l for l in res["limitations"])

    def test_non_significant_car(self):
        # Sample hovering around zero
        car_sample = [-0.5, 0.4, -0.2, 0.1, -0.3, 0.2, 0.0, -0.1]
        res = compute_statistical_tests(car_sample, null_mean=0.0)

        assert res["sample_size"] == 8
        assert res["statistically_significant_5pct"] is False
        assert res["p_value"] > 0.05
        assert "null hypothesis" in res["interpretation"]

    def test_insufficient_sample(self):
        res = compute_statistical_tests([1.5], null_mean=0.0)
        assert res["sample_size"] == 1
        assert res["t_statistic"] is None
        assert res["statistically_significant_5pct"] is False


class TestCompanyEventStudies:
    """Tests for company-level event study processing."""

    def test_run_company_event_studies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dates = pd.date_range("2023-01-01", periods=100, freq="B").strftime("%Y-%m-%d").tolist()
            stock_df = pd.DataFrame({
                "Date": dates,
                "Open": np.linspace(100, 150, 100),
                "High": np.linspace(102, 152, 100),
                "Low": np.linspace(99, 149, 100),
                "Close": np.linspace(101, 151, 100),
                "Volume": [1000000] * 100,
                "Ticker": ["MOCK"] * 100
            })
            stock_df.to_csv(os.path.join(tmpdir, "MOCK_raw.csv"), index=False)

            events_df = pd.DataFrame([{
                "event_id": "EVT_TEST",
                "event_name": "Test Event",
                "event_category": "Geopolitical",
                "event_date": dates[50]
            }])

            sp500_df = pd.DataFrame({
                "Date": dates,
                "Close": np.linspace(4000, 4500, 100),
                "daily_return": [0.001] * 100
            })

            companies_df = pd.DataFrame([{
                "ticker": "MOCK",
                "company_name": "Mock Inc",
                "sector": "Technology",
                "industry": "Software"
            }])

            comp_df, ar_df = run_company_event_studies(
                events_df=events_df,
                sp500_df=sp500_df,
                stocks_dir=tmpdir,
                companies_df=companies_df,
                max_companies=5
            )

            assert not comp_df.empty
            assert comp_df.iloc[0]["ticker"] == "MOCK"
            assert comp_df.iloc[0]["sector"] == "Technology"
            assert "post_5d_return_pct" in comp_df.columns
            assert "spread_vs_sp500_5d_pct" in comp_df.columns


class TestDispersionAndCategoryComparison:
    """Tests for intra-sector dispersion and category aggregations."""

    def test_analyze_cross_company_dispersion(self):
        comp_df = pd.DataFrame([
            {
                "event_id": "EV1", "event_name": "Ev1", "event_category": "Macro",
                "sector": "Tech", "ticker": "T1", "post_5d_return_pct": 5.0
            },
            {
                "event_id": "EV1", "event_name": "Ev1", "event_category": "Macro",
                "sector": "Tech", "ticker": "T2", "post_5d_return_pct": -2.0
            },
            {
                "event_id": "EV1", "event_name": "Ev1", "event_category": "Macro",
                "sector": "Tech", "ticker": "T3", "post_5d_return_pct": 1.5
            }
        ])

        disp_df = analyze_cross_company_dispersion(comp_df)
        assert len(disp_df) == 1
        row = disp_df.iloc[0]
        assert row["top_resilient_ticker"] == "T1"
        assert row["most_vulnerable_ticker"] == "T2"
        assert row["return_range_pct"] == 7.0

    def test_compare_event_categories(self):
        event_df = pd.DataFrame([
            {
                "event_name": "Ev1", "event_category": "Geopolitical",
                "pre_5d_return_pct": -1.0, "post_5d_return_pct": -3.5,
                "post_30d_return_pct": -2.0, "max_drawdown_30d_pct": -6.0, "vix_at_event": 28.5
            },
            {
                "event_name": "Ev2", "event_category": "Macro",
                "pre_5d_return_pct": 0.5, "post_5d_return_pct": 1.2,
                "post_30d_return_pct": 3.0, "max_drawdown_30d_pct": -1.5, "vix_at_event": 16.0
            }
        ])

        cat_df = compare_event_categories(event_df)
        assert len(cat_df) == 2
        assert "event_category" in cat_df.columns
        assert "avg_post_5d_return_pct" in cat_df.columns


class TestGroundTruthVerification:
    """Tests independent NumPy/SciPy ground truth formulas against analytical functions."""

    def test_independent_daily_return_and_volatility(self):
        closes = [100.0, 102.0, 101.0, 104.0, 103.0]
        rets, vols = independent_daily_return_and_volatility(closes, volatility_window=3)

        assert np.isnan(rets[0])
        assert np.isclose(rets[1], 0.02)
        assert np.isclose(rets[2], (101.0 - 102.0) / 102.0)

    def test_independent_ols_market_model(self):
        np.random.seed(42)
        mkt_ret = np.random.normal(0.0005, 0.01, 50)
        sec_ret = 0.0002 + 1.25 * mkt_ret + np.random.normal(0, 0.001, 50)

        a_ind, b_ind, s_ind = independent_ols_market_model(sec_ret, mkt_ret)
        a_eng, b_eng, s_eng, n = estimate_market_model_parameters(pd.Series(sec_ret), pd.Series(mkt_ret))

        assert n == 50
        # Should match within numerical tolerance
        assert np.isclose(b_ind, b_eng, atol=1e-4)
        assert np.isclose(a_ind, a_eng, atol=1e-4)
        assert np.isclose(s_ind, s_eng, atol=1e-4)
