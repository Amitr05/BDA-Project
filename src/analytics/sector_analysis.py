"""
Sector-Level Market Analytics Module
Provides:
1. Sector-level performance analysis (cumulative return, annualized return, volatility, Sharpe ratio, max drawdown).
2. Comparison of multiple sector ETFs (normalized price tracking, performance ranking, correlation matrices).
3. Alignment of sector ETF data with the four US equity indices (S&P 500, NASDAQ, Dow Jones, Russell 2000) and VIX.
4. Sector event-study analysis across curated global events.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATHS = {
    "sp500": "data/processed/sp500_processed.parquet",
    "nasdaq": "data/processed/nasdaq_processed.parquet",
    "dow_jones": "data/processed/dow_jones_processed.parquet",
    "russell2000": "data/processed/russell2000_processed.parquet",
}
VIX_PROCESSED_PATH = "data/processed/vix_processed.parquet"
ALIGNED_SECTOR_MARKET_PARQUET = "data/processed/aligned_sector_market_data.parquet"
ALIGNED_SECTOR_MARKET_CSV = "data/processed/aligned_sector_market_data.csv"
SECTOR_EVENT_STUDY_PARQUET = "data/processed/sector_event_study_results.parquet"
SECTOR_EVENT_STUDY_CSV = "data/processed/sector_event_study_results.csv"

SECTOR_NAME_MAP = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities"
}


def load_processed_etf_frames(
    etfs_dir: str = "data/processed/etfs",
    tickers: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """Load processed Parquet files for each ETF into a dictionary."""
    target_tickers = tickers or list(SECTOR_NAME_MAP.keys())
    frames: Dict[str, pd.DataFrame] = {}

    for ticker in target_tickers:
        file_path = os.path.join(etfs_dir, f"{ticker}_processed.parquet")
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            df = df.sort_values("Date").reset_index(drop=True)
            frames[ticker] = df
        else:
            logger.debug(f"Processed file for {ticker} not found at {file_path}")

    return frames


def calculate_sector_performance_summary(
    etf_frames: Dict[str, pd.DataFrame],
    risk_free_rate: float = 0.02
) -> pd.DataFrame:
    """
    Compute comprehensive quantitative performance metrics for each sector ETF:
    - Total Return (%)
    - Annualized Return (%)
    - Annualized Volatility (%)
    - Sharpe Ratio
    - Maximum Drawdown (%)
    - Best & Worst Single-Day Return (%)
    - Positive Days Ratio (%)
    """
    records = []

    for ticker, df in etf_frames.items():
        if df.empty or len(df) < 5:
            continue

        close = df["Close"].dropna()
        daily_ret = df["daily_return"].dropna() if "daily_return" in df.columns else close.pct_change().dropna()

        n_days = len(daily_ret)
        if n_days == 0 or close.iloc[0] <= 0:
            continue

        p_start = float(close.iloc[0])
        p_end = float(close.iloc[-1])
        total_return = (p_end - p_start) / p_start

        # Annualized return
        years = n_days / 252.0
        ann_return = ((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 and (1.0 + total_return) > 0 else np.nan

        # Annualized volatility
        ann_vol = float(daily_ret.std() * np.sqrt(252.0))

        # Sharpe ratio
        sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 and pd.notna(ann_return) else np.nan

        # Maximum drawdown
        cum_max = close.cummax()
        drawdowns = (close - cum_max) / cum_max
        max_drawdown = float(drawdowns.min())

        # Win rate / positive days
        pos_days = (daily_ret > 0).sum() / float(n_days) * 100.0

        sector = df["Sector"].iloc[0] if "Sector" in df.columns else SECTOR_NAME_MAP.get(ticker, "Unknown")

        records.append({
            "ticker": ticker,
            "sector": sector,
            "start_date": df["Date"].min(),
            "end_date": df["Date"].max(),
            "trading_sessions": n_days,
            "total_return_pct": round(total_return * 100.0, 2),
            "annualized_return_pct": round(ann_return * 100.0, 2) if pd.notna(ann_return) else None,
            "annualized_volatility_pct": round(ann_vol * 100.0, 2),
            "sharpe_ratio": round(sharpe, 2) if pd.notna(sharpe) else None,
            "max_drawdown_pct": round(max_drawdown * 100.0, 2),
            "best_day_pct": round(float(daily_ret.max() * 100.0), 2),
            "worst_day_pct": round(float(daily_ret.min() * 100.0), 2),
            "positive_days_pct": round(pos_days, 2)
        })

    summary_df = pd.DataFrame(records)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("total_return_pct", ascending=False).reset_index(drop=True)

    return summary_df


def compare_multiple_sector_etfs(
    etf_frames: Dict[str, pd.DataFrame],
    base_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate normalized close price series for multiple sector ETFs, rebased to 100.0
    at base_date (or the earliest common trading date).
    Enables direct visual and quantitative comparison across sectors.
    """
    close_series = {}

    for ticker, df in etf_frames.items():
        if not df.empty and "Date" in df.columns and "Close" in df.columns:
            s = df.set_index("Date")["Close"]
            close_series[ticker] = s

    if not close_series:
        return pd.DataFrame()

    merged_df = pd.DataFrame(close_series).sort_index()

    if base_date is not None and base_date in merged_df.index:
        start_row = merged_df.loc[base_date]
    else:
        # Find earliest date where at least one ETF has valid close
        valid_idx = merged_df.dropna(how="all").index
        if len(valid_idx) == 0:
            return pd.DataFrame()
        start_row = merged_df.loc[valid_idx[0]]

    # Rebase each column to 100.0 at starting valid price
    normalized_df = pd.DataFrame(index=merged_df.index)
    for col in merged_df.columns:
        base_val = start_row[col]
        if pd.isna(base_val) or base_val <= 0:
            first_valid = merged_df[col].dropna()
            base_val = first_valid.iloc[0] if len(first_valid) > 0 else np.nan

        if pd.notna(base_val) and base_val > 0:
            normalized_df[col] = (merged_df[col] / base_val) * 100.0
        else:
            normalized_df[col] = np.nan

    normalized_df = normalized_df.round(2).reset_index()
    return normalized_df


def compute_sector_correlation_matrix(
    etf_frames: Dict[str, pd.DataFrame],
    method: str = "pearson"
) -> pd.DataFrame:
    """
    Compute pairwise correlation matrix of daily returns across sector ETFs.
    """
    returns_series = {}

    for ticker, df in etf_frames.items():
        if not df.empty and "Date" in df.columns:
            if "daily_return" in df.columns:
                ret = df.set_index("Date")["daily_return"]
            else:
                ret = df.set_index("Date")["Close"].pct_change()
            returns_series[ticker] = ret

    if not returns_series:
        return pd.DataFrame()

    combined_ret = pd.DataFrame(returns_series).dropna(how="all")
    corr_matrix = combined_ret.corr(method=method).round(4)
    return corr_matrix


def rank_sector_performance(
    etf_frames: Dict[str, pd.DataFrame],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Rank all sector ETFs from 1 (Best) to N (Worst) based on cumulative return
    over a specified historical date window.
    """
    rows = []

    for ticker, df in etf_frames.items():
        if df.empty or "Date" not in df.columns or "Close" not in df.columns:
            continue

        sub = df.copy()
        if start_date:
            sub = sub[sub["Date"] >= start_date]
        if end_date:
            sub = sub[sub["Date"] <= end_date]

        if len(sub) < 2:
            continue

        p_start = float(sub["Close"].iloc[0])
        p_end = float(sub["Close"].iloc[-1])
        if p_start > 0:
            ret = (p_end - p_start) / p_start * 100.0
            sector = sub["Sector"].iloc[0] if "Sector" in sub.columns else SECTOR_NAME_MAP.get(ticker, "Unknown")
            rows.append({
                "ticker": ticker,
                "sector": sector,
                "return_pct": round(ret, 2),
                "start_close": round(p_start, 2),
                "end_close": round(p_end, 2),
                "start_date": sub["Date"].iloc[0],
                "end_date": sub["Date"].iloc[-1]
            })

    ranking_df = pd.DataFrame(rows)
    if not ranking_df.empty:
        ranking_df = ranking_df.sort_values("return_pct", ascending=False).reset_index(drop=True)
        ranking_df["rank"] = range(1, len(ranking_df) + 1)
        # Reorder columns
        cols = ["rank", "ticker", "sector", "return_pct", "start_close", "end_close", "start_date", "end_date"]
        ranking_df = ranking_df[cols]

    return ranking_df


def align_sector_etfs_with_market_and_vix(
    etf_frames: Dict[str, pd.DataFrame],
    market_parquet_paths: Optional[Dict[str, str]] = None,
    vix_parquet_path: Optional[str] = None,
    output_parquet: str = ALIGNED_SECTOR_MARKET_PARQUET,
    output_csv: str = ALIGNED_SECTOR_MARKET_CSV
) -> pd.DataFrame:
    """
    Align sector ETF daily returns with:
    - 4 equity indices: S&P 500 (^GSPC), NASDAQ (^IXIC), Dow Jones (^DJI), Russell 2000 (^RUT)
    - CBOE Volatility Index (VIX)
    Calculates:
    - Beta to S&P 500 and other indices
    - Relative Return / Spread: Return(ETF) - Return(Index)
    - Volatility regime classification based on VIX (<20 Low, 20-30 Elevated, >30 High)
    """
    paths = market_parquet_paths or DEFAULT_INDEX_PATHS
    vix_path = vix_parquet_path or VIX_PROCESSED_PATH

    # Load market indices
    market_returns = {}
    for key, path in paths.items():
        if os.path.exists(path):
            mdf = pd.read_parquet(path)
            mdf["Date"] = pd.to_datetime(mdf["Date"]).dt.strftime("%Y-%m-%d")
            if "daily_return" not in mdf.columns:
                mdf["daily_return"] = mdf["Close"].pct_change()
            market_returns[f"{key}_return"] = mdf.set_index("Date")["daily_return"]
            market_returns[f"{key}_close"] = mdf.set_index("Date")["Close"]

    # Load VIX
    vix_close = None
    if os.path.exists(vix_path):
        vdf = pd.read_parquet(vix_path)
        vdf["Date"] = pd.to_datetime(vdf["Date"]).dt.strftime("%Y-%m-%d")
        vix_close = vdf.set_index("Date")["Close"]

    all_rows = []

    # Calculate beta and spreads
    for ticker, etf_df in etf_frames.items():
        if etf_df.empty:
            continue

        sub = etf_df.copy()
        sub["Date"] = pd.to_datetime(sub["Date"]).dt.strftime("%Y-%m-%d")

        if "daily_return" not in sub.columns:
            sub["daily_return"] = sub["Close"].pct_change()

        etf_ret = sub.set_index("Date")["daily_return"]

        aligned = pd.DataFrame({"etf_return": etf_ret, "etf_close": sub.set_index("Date")["Close"]})

        for col_name, s in market_returns.items():
            aligned[col_name] = s

        if vix_close is not None:
            aligned["vix_level"] = vix_close

        # Calculate relative return spreads
        if "sp500_return" in aligned.columns:
            aligned["spread_vs_sp500"] = (aligned["etf_return"] - aligned["sp500_return"]) * 100.0
        if "nasdaq_return" in aligned.columns:
            aligned["spread_vs_nasdaq"] = (aligned["etf_return"] - aligned["nasdaq_return"]) * 100.0
        if "dow_jones_return" in aligned.columns:
            aligned["spread_vs_dow_jones"] = (aligned["etf_return"] - aligned["dow_jones_return"]) * 100.0
        if "russell2000_return" in aligned.columns:
            aligned["spread_vs_russell2000"] = (aligned["etf_return"] - aligned["russell2000_return"]) * 100.0

        # Calculate rolling 60-day Beta to S&P 500
        if "sp500_return" in aligned.columns:
            cov = aligned["etf_return"].rolling(60).cov(aligned["sp500_return"])
            var = aligned["sp500_return"].rolling(60).var()
            aligned["rolling_beta_60d_sp500"] = (cov / var).round(4)

        # Volatility regime
        if "vix_level" in aligned.columns:
            vix = aligned["vix_level"]
            aligned["vix_regime"] = np.where(
                vix < 20.0, "Low Volatility (<20)",
                np.where(vix <= 30.0, "Elevated Volatility (20-30)", "High Volatility (>30)")
            )

        aligned["ticker"] = ticker
        aligned["sector"] = sub["Sector"].iloc[0] if "Sector" in sub.columns else SECTOR_NAME_MAP.get(ticker, "Unknown")
        aligned = aligned.reset_index().rename(columns={"index": "Date"})
        all_rows.append(aligned)

    if not all_rows:
        return pd.DataFrame()

    aligned_panel = pd.concat(all_rows, ignore_index=True)
    aligned_panel = aligned_panel.sort_values(["Date", "ticker"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    aligned_panel.to_parquet(output_parquet, engine="pyarrow", index=False)
    aligned_panel.to_csv(output_csv, index=False)
    logger.info(f"Saved aligned sector and market dataset to {output_parquet} ({len(aligned_panel)} rows).")

    return aligned_panel


def run_sector_event_study(
    events_csv_path: str = "data/raw/events.csv",
    etf_frames: Optional[Dict[str, pd.DataFrame]] = None,
    vix_parquet_path: str = VIX_PROCESSED_PATH,
    output_parquet: str = SECTOR_EVENT_STUDY_PARQUET,
    output_csv: str = SECTOR_EVENT_STUDY_CSV
) -> pd.DataFrame:
    """
    Perform event-study analysis for each of the 8 sector ETFs across curated global events.
    Computes:
    - Pre-5D Return (%)
    - Post-5D Return (%)
    - Post-30D Return (%)
    - 30D Maximum Drawdown (%)
    Aligned with market VIX levels.
    """
    if not os.path.exists(events_csv_path):
        raise FileNotFoundError(f"Events CSV not found at {events_csv_path}")

    events_df = pd.read_csv(events_csv_path)
    frames = etf_frames or load_processed_etf_frames()

    if not frames:
        logger.warning("No ETF frames provided for sector event study.")
        return pd.DataFrame()

    # Load VIX
    vix_by_date = {}
    if os.path.exists(vix_parquet_path):
        vdf = pd.read_parquet(vix_parquet_path)
        vdf["Date"] = pd.to_datetime(vdf["Date"]).dt.strftime("%Y-%m-%d")
        vix_by_date = vdf.set_index("Date")["Close"].to_dict()

    study_records = []

    for ticker, df in frames.items():
        if df.empty:
            continue

        dates = df["Date"].tolist()
        close_series = df["Close"].tolist()
        n_days = len(dates)
        sector_name = df["Sector"].iloc[0] if "Sector" in df.columns else SECTOR_NAME_MAP.get(ticker, "Unknown")

        for _, ev in events_df.iterrows():
            ev_date = pd.Timestamp(ev["event_date"]).strftime("%Y-%m-%d")

            # Find T0
            t0_idx = None
            for idx, d in enumerate(dates):
                if d >= ev_date:
                    t0_idx = idx
                    break

            if t0_idx is None:
                continue

            t0_date = dates[t0_idx]
            pre_idx = max(0, t0_idx - 5)
            short_idx = t0_idx + 5
            long_idx = t0_idx + 30

            short_available = short_idx < n_days
            long_available = long_idx < n_days

            p0 = float(close_series[t0_idx])
            p_pre = float(close_series[pre_idx])
            p_short = float(close_series[short_idx]) if short_available else np.nan
            p_long = float(close_series[long_idx]) if long_available else np.nan

            pre_ret = ((p0 - p_pre) / p_pre * 100.0) if p_pre > 0 else np.nan
            post_5d_ret = ((p_short - p0) / p0 * 100.0) if short_available and p0 > 0 else np.nan
            post_30d_ret = ((p_long - p0) / p0 * 100.0) if long_available and p0 > 0 else np.nan

            drawdown = np.nan
            if long_available and p0 > 0:
                lowest_price = min(close_series[t0_idx : long_idx + 1])
                drawdown = (lowest_price - p0) / p0 * 100.0

            vix_t0 = vix_by_date.get(t0_date, np.nan)

            study_records.append({
                "event_id": ev.get("event_id"),
                "event_name": ev.get("event_name"),
                "calendar_event_date": ev_date,
                "effective_trading_date": t0_date,
                "event_category": ev.get("event_category"),
                "ticker": ticker,
                "sector": sector_name,
                "price_t0": round(p0, 2),
                "pre_5d_return_pct": round(pre_ret, 2) if pd.notna(pre_ret) else None,
                "post_5d_return_pct": round(post_5d_ret, 2) if pd.notna(post_5d_ret) else None,
                "post_30d_return_pct": round(post_30d_ret, 2) if pd.notna(post_30d_ret) else None,
                "max_drawdown_30d_pct": round(drawdown, 2) if pd.notna(drawdown) else None,
                "vix_at_event": round(vix_t0, 2) if pd.notna(vix_t0) else None
            })

    result_df = pd.DataFrame(study_records)
    if not result_df.empty:
        os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
        result_df.to_parquet(output_parquet, engine="pyarrow", index=False)
        result_df.to_csv(output_csv, index=False)
        logger.info(f"Generated sector event study results: {len(result_df)} records.")

    return result_df
