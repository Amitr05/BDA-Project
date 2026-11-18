"""
Unit Tests for Sector-Level Market Analysis and Alignment Modules (Prompt 5)
Verifies:
- Sector-level performance analysis (cumulative return, annualized metrics, Sharpe, max drawdown).
- Multi-sector ETF comparison and normalized price tracking (rebased to 100).
- Sector return correlation matrix.
- Performance ranking across sectors.
- Alignment of sector ETFs with the four market indices and VIX.
- Sector ETF event-study metrics around global events.
"""

import os
import tempfile
import pandas as pd
import numpy as np
import pytest

from src.analytics.sector_analysis import (
    SECTOR_NAME_MAP,
    calculate_sector_performance_summary,
    compare_multiple_sector_etfs,
    compute_sector_correlation_matrix,
    rank_sector_performance,
    align_sector_etfs_with_market_and_vix,
    run_sector_event_study
)


def _create_mock_etf_frame(ticker: str, sector: str, start_val: float, growth: float, n_days: int = 100) -> pd.DataFrame:
    """Helper to generate synthetic ETF price time series."""
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B").strftime("%Y-%m-%d").tolist()
    # Linear price path
    closes = [start_val * (1.0 + growth * (i / n_days)) for i in range(n_days)]
    daily_returns = [0.0] + [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, n_days)]

    return pd.DataFrame({
        "Date": dates,
        "Open": closes,
        "High": [c * 1.01 for c in closes],
        "Low": [c * 0.99 for c in closes],
        "Close": closes,
        "Adj Close": closes,
        "Volume": [1000000] * n_days,
        "daily_return": daily_returns,
        "daily_return_pct": [r * 100.0 for r in daily_returns],
        "Ticker": [ticker] * n_days,
        "Sector": [sector] * n_days
    })


def test_sector_performance_summary_calculation():
    """Verify performance metrics calculation for individual sectors."""
    frames = {
        "XLK": _create_mock_etf_frame("XLK", "Technology", start_val=100.0, growth=0.20, n_days=252),
        "XLE": _create_mock_etf_frame("XLE", "Energy", start_val=80.0, growth=-0.10, n_days=252)
    }

    summary_df = calculate_sector_performance_summary(frames, risk_free_rate=0.03)

    assert len(summary_df) == 2
    assert "total_return_pct" in summary_df.columns
    assert "annualized_return_pct" in summary_df.columns
    assert "annualized_volatility_pct" in summary_df.columns
    assert "sharpe_ratio" in summary_df.columns
    assert "max_drawdown_pct" in summary_df.columns

    xlk_row = summary_df[summary_df["ticker"] == "XLK"].iloc[0]
    assert np.isclose(xlk_row["total_return_pct"], 20.0, atol=0.5)
    assert xlk_row["sharpe_ratio"] is not None

    xle_row = summary_df[summary_df["ticker"] == "XLE"].iloc[0]
    assert np.isclose(xle_row["total_return_pct"], -10.0, atol=0.5)


def test_compare_multiple_sector_etfs_normalized():
    """Verify multi-ETF normalized comparison rebases all series to 100.0 on base date."""
    frames = {
        "XLK": _create_mock_etf_frame("XLK", "Technology", start_val=200.0, growth=0.10, n_days=10),
        "XLF": _create_mock_etf_frame("XLF", "Financials", start_val=40.0, growth=0.05, n_days=10),
        "XLU": _create_mock_etf_frame("XLU", "Utilities", start_val=60.0, growth=-0.02, n_days=10),
    }

    normalized = compare_multiple_sector_etfs(frames)

    assert "Date" in normalized.columns
    assert "XLK" in normalized.columns
    assert "XLF" in normalized.columns
    assert "XLU" in normalized.columns

    # Initial row must be normalized to exactly 100.0
    first_row = normalized.iloc[0]
    assert np.isclose(first_row["XLK"], 100.0)
    assert np.isclose(first_row["XLF"], 100.0)
    assert np.isclose(first_row["XLU"], 100.0)

    # Last row reflects expected relative growth
    last_row = normalized.iloc[-1]
    assert last_row["XLK"] > 100.0
    assert last_row["XLF"] > 100.0
    assert last_row["XLU"] < 100.0


def test_compute_sector_correlation_matrix():
    """Verify pairwise return correlation matrix properties."""
    frames = {
        "XLK": _create_mock_etf_frame("XLK", "Technology", start_val=100.0, growth=0.10, n_days=50),
        "XLF": _create_mock_etf_frame("XLF", "Financials", start_val=50.0, growth=0.05, n_days=50),
        "XLE": _create_mock_etf_frame("XLE", "Energy", start_val=80.0, growth=-0.05, n_days=50),
    }

    corr = compute_sector_correlation_matrix(frames)

    assert corr.shape == (3, 3)
    # Diagonal must be 1.0
    for ticker in ["XLK", "XLF", "XLE"]:
        assert np.isclose(corr.loc[ticker, ticker], 1.0)

    # Matrix must be symmetric
    assert np.isclose(corr.loc["XLK", "XLF"], corr.loc["XLF", "XLK"])
    assert np.isclose(corr.loc["XLK", "XLE"], corr.loc["XLE", "XLK"])


def test_rank_sector_performance():
    """Verify sector ranking assigns ranks 1 (Best) to N (Worst) based on return."""
    frames = {
        "XLK": _create_mock_etf_frame("XLK", "Technology", start_val=100.0, growth=0.30, n_days=30),  # +30%
        "XLF": _create_mock_etf_frame("XLF", "Financials", start_val=50.0, growth=0.10, n_days=30),   # +10%
        "XLE": _create_mock_etf_frame("XLE", "Energy", start_val=80.0, growth=-0.15, n_days=30),     # -15%
    }

    ranking = rank_sector_performance(frames)

    assert len(ranking) == 3
    assert list(ranking["rank"]) == [1, 2, 3]
    assert ranking.iloc[0]["ticker"] == "XLK"
    assert ranking.iloc[1]["ticker"] == "XLF"
    assert ranking.iloc[2]["ticker"] == "XLE"


def test_align_sector_etfs_with_market_and_vix():
    """Verify alignment of sector ETFs with equity indices and VIX regimes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        n_days = 80
        dates = pd.date_range("2024-01-02", periods=n_days, freq="B").strftime("%Y-%m-%d").tolist()

        # Create mock market indices
        index_paths = {}
        for idx_name in ["sp500", "nasdaq", "dow_jones", "russell2000"]:
            idx_path = os.path.join(tmpdir, f"{idx_name}.parquet")
            idx_df = pd.DataFrame({
                "Date": dates,
                "Close": [5000.0 * (1.0 + 0.05 * (i / n_days)) for i in range(n_days)],
                "daily_return": [0.0] + [0.0005] * (n_days - 1)
            })
            idx_df.to_parquet(idx_path, index=False)
            index_paths[idx_name] = idx_path

        # Create mock VIX
        vix_path = os.path.join(tmpdir, "vix.parquet")
        vix_df = pd.DataFrame({
            "Date": dates,
            "Close": [14.0 if i < 30 else 24.0 if i < 60 else 32.0 for i in range(n_days)]
        })
        vix_df.to_parquet(vix_path, index=False)

        etf_frames = {
            "XLK": _create_mock_etf_frame("XLK", "Technology", start_val=150.0, growth=0.10, n_days=n_days)
        }

        out_parquet = os.path.join(tmpdir, "aligned.parquet")
        out_csv = os.path.join(tmpdir, "aligned.csv")

        aligned = align_sector_etfs_with_market_and_vix(
            etf_frames=etf_frames,
            market_parquet_paths=index_paths,
            vix_parquet_path=vix_path,
            output_parquet=out_parquet,
            output_csv=out_csv
        )

        assert os.path.exists(out_parquet)
        assert len(aligned) == n_days
        assert "spread_vs_sp500" in aligned.columns
        assert "rolling_beta_60d_sp500" in aligned.columns
        assert "vix_regime" in aligned.columns

        # Verify VIX regimes correctly tagged
        assert "Low Volatility (<20)" in aligned["vix_regime"].values
        assert "Elevated Volatility (20-30)" in aligned["vix_regime"].values
        assert "High Volatility (>30)" in aligned["vix_regime"].values


def test_sector_event_study():
    """Verify sector event study runs and calculates Pre-5D, Post-5D, and Post-30D metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        n_days = 70
        dates = pd.date_range("2024-01-02", periods=n_days, freq="B").strftime("%Y-%m-%d").tolist()

        events_csv = os.path.join(tmpdir, "events.csv")
        pd.DataFrame([{
            "event_id": "TEST-001",
            "event_name": "Test Global Event",
            "event_date": dates[15],
            "event_category": "Geopolitical"
        }]).to_csv(events_csv, index=False)

        vix_path = os.path.join(tmpdir, "vix.parquet")
        pd.DataFrame({
            "Date": dates,
            "Close": [20.0] * n_days
        }).to_parquet(vix_path, index=False)

        etf_frames = {
            "XLK": _create_mock_etf_frame("XLK", "Technology", start_val=100.0, growth=0.10, n_days=n_days),
            "XLE": _create_mock_etf_frame("XLE", "Energy", start_val=80.0, growth=-0.05, n_days=n_days)
        }

        study_parquet = os.path.join(tmpdir, "study.parquet")
        study_csv = os.path.join(tmpdir, "study.csv")

        results = run_sector_event_study(
            events_csv_path=events_csv,
            etf_frames=etf_frames,
            vix_parquet_path=vix_path,
            output_parquet=study_parquet,
            output_csv=study_csv
        )

        assert os.path.exists(study_parquet)
        assert len(results) == 2  # XLK and XLE for the event
        assert set(results["ticker"]) == {"XLK", "XLE"}
        assert "pre_5d_return_pct" in results.columns
        assert "post_5d_return_pct" in results.columns
        assert "post_30d_return_pct" in results.columns
        assert "max_drawdown_30d_pct" in results.columns
