"""
Event Study Analytics Module
Computes market reactions around major geopolitical and macroeconomic events
using exact trading-day windows (Pre-5D, Post-5D, Post-30D) and volatility metrics.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


def load_processed_data(
    sp500_path: str = "data/processed/sp500_processed.parquet",
    vix_path: str = "data/processed/vix_processed.parquet"
) -> Dict[str, pd.DataFrame]:
    """
    Load preprocessed S&P 500 and VIX datasets from Parquet into Pandas DataFrames.
    """
    if not os.path.exists(sp500_path):
        raise FileNotFoundError(f"Processed S&P 500 file not found at {sp500_path}")
    if not os.path.exists(vix_path):
        raise FileNotFoundError(f"Processed VIX file not found at {vix_path}")

    sp500_df = pd.read_parquet(sp500_path)
    vix_df = pd.read_parquet(vix_path)

    sp500_df["Date"] = pd.to_datetime(sp500_df["Date"]).dt.strftime("%Y-%m-%d")
    vix_df["Date"] = pd.to_datetime(vix_df["Date"]).dt.strftime("%Y-%m-%d")

    sp500_df = sp500_df.sort_values("Date").reset_index(drop=True)
    vix_df = vix_df.sort_values("Date").reset_index(drop=True)

    return {"sp500": sp500_df, "vix": vix_df}


def find_effective_trading_day(target_date: str, trading_dates: List[str]) -> Optional[int]:
    """
    Given a calendar target date (YYYY-MM-DD) and a sorted list of trading day strings:
    Returns the index of the exact trading day if open, or the first subsequent trading day.
    """
    for idx, d in enumerate(trading_dates):
        if d >= target_date:
            return idx
    return None


def calculate_event_metrics(
    event: Dict[str, Any],
    sp500_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    pre_window: int = 5,
    post_window_short: int = 5,
    post_window_long: int = 30
) -> Dict[str, Any]:
    """
    Calculate trading-day event metrics for a single event.

    Window definitions (using trading days, not calendar days):
    - T0: Effective event trading day (date of event or next available trading day)
    - T-5: 5 trading days prior to T0
    - T+5: 5 trading days following T0
    - T+30: 30 trading days following T0

    Returns:
    - Pre-5D Return (%): (Close[T0] - Close[T-5]) / Close[T-5] * 100
    - Post-5D Return (%): (Close[T+5] - Close[T0]) / Close[T0] * 100
    - Post-30D Return (%): (Close[T+30] - Close[T0]) / Close[T0] * 100
    - VIX values at T-5, T0, T+5, T+30, and maximum VIX within [T0, T+30].
    """
    calendar_date = str(event["event_date"])
    trading_dates = sp500_df["Date"].tolist()

    t0_idx = find_effective_trading_day(calendar_date, trading_dates)
    if t0_idx is None:
        logger.warning(f"Event '{event.get('event_name')}' date {calendar_date} is beyond dataset range.")
        return {}

    t0_date = trading_dates[t0_idx]
    n_days = len(trading_dates)

    # Pre-event index
    pre_idx = max(0, t0_idx - pre_window)
    pre_date = trading_dates[pre_idx]

    # Post-event short window index
    post_5_idx = min(n_days - 1, t0_idx + post_window_short)
    post_5_date = trading_dates[post_5_idx]

    # Post-event long window index
    post_30_idx = min(n_days - 1, t0_idx + post_window_long)
    post_30_date = trading_dates[post_30_idx]

    # S&P 500 Prices
    sp_p0 = sp500_df.loc[t0_idx, "Close"]
    sp_pre = sp500_df.loc[pre_idx, "Close"]
    sp_post_5 = sp500_df.loc[post_5_idx, "Close"]
    sp_post_30 = sp500_df.loc[post_30_idx, "Close"]

    # Returns calculation
    pre_5d_ret = ((sp_p0 - sp_pre) / sp_pre) * 100.0 if sp_pre else None
    post_5d_ret = ((sp_post_5 - sp_p0) / sp_p0) * 100.0 if sp_p0 else None
    post_30d_ret = ((sp_post_30 - sp_p0) / sp_p0) * 100.0 if sp_p0 else None

    # Max Drawdown during [T0, T+30]
    post_30_slice = sp500_df.loc[t0_idx:post_30_idx, "Close"]
    min_post_30 = post_30_slice.min()
    max_drawdown_30d = ((min_post_30 - sp_p0) / sp_p0) * 100.0 if sp_p0 else None

    # VIX alignment via Date lookup
    vix_date_map = vix_df.set_index("Date")["Close"].to_dict()
    vix_t0 = vix_date_map.get(t0_date, np.nan)
    vix_pre = vix_date_map.get(pre_date, np.nan)
    vix_post_5 = vix_date_map.get(post_5_date, np.nan)
    vix_post_30 = vix_date_map.get(post_30_date, np.nan)

    vix_slice = [vix_date_map[d] for d in trading_dates[t0_idx:post_30_idx + 1] if d in vix_date_map]
    vix_max_30d = max(vix_slice) if vix_slice else np.nan

    vix_5d_change = (vix_post_5 - vix_t0) if not np.isnan(vix_post_5) and not np.isnan(vix_t0) else np.nan

    return {
        "event_name": event.get("event_name"),
        "calendar_event_date": calendar_date,
        "effective_trading_date": t0_date,
        "event_category": event.get("event_category"),
        "short_description": event.get("short_description"),
        "sp500_price_t0": round(float(sp_p0), 2),
        "sp500_pre_5d_return_pct": round(float(pre_5d_ret), 2) if pre_5d_ret is not None else None,
        "sp500_post_5d_return_pct": round(float(post_5d_ret), 2) if post_5d_ret is not None else None,
        "sp500_post_30d_return_pct": round(float(post_30d_ret), 2) if post_30d_ret is not None else None,
        "sp500_max_drawdown_30d_pct": round(float(max_drawdown_30d), 2) if max_drawdown_30d is not None else None,
        "vix_at_event": round(float(vix_t0), 2) if not np.isnan(vix_t0) else None,
        "vix_pre_5d": round(float(vix_pre), 2) if not np.isnan(vix_pre) else None,
        "vix_post_5d": round(float(vix_post_5), 2) if not np.isnan(vix_post_5) else None,
        "vix_post_30d": round(float(vix_post_30), 2) if not np.isnan(vix_post_30) else None,
        "vix_5d_change": round(float(vix_5d_change), 2) if not np.isnan(vix_5d_change) else None,
        "vix_max_30d": round(float(vix_max_30d), 2) if not np.isnan(vix_max_30d) else None,
    }


def run_event_study_analysis(
    events_csv_path: str = "data/raw/events.csv",
    sp500_parquet_path: str = "data/processed/sp500_processed.parquet",
    vix_parquet_path: str = "data/processed/vix_processed.parquet",
    output_csv_path: str = "data/processed/event_study_results.csv",
    output_parquet_path: str = "data/processed/event_study_results.parquet"
) -> pd.DataFrame:
    """
    Run event-study calculations for all events in the event calendar
    and export the dataset as CSV and Parquet.
    """
    if not os.path.exists(events_csv_path):
        raise FileNotFoundError(f"Events file not found at {events_csv_path}")

    events_df = pd.read_csv(events_csv_path)
    data = load_processed_data(sp500_path=sp500_parquet_path, vix_path=vix_parquet_path)
    sp500_df = data["sp500"]
    vix_df = data["vix"]

    results: List[Dict[str, Any]] = []
    for _, row in events_df.iterrows():
        event_dict = row.to_dict()
        metrics = calculate_event_metrics(event_dict, sp500_df, vix_df)
        if metrics:
            results.append(metrics)

    res_df = pd.DataFrame(results)

    # Save to CSV and Parquet
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    res_df.to_csv(output_csv_path, index=False)
    res_df.to_parquet(output_parquet_path, engine="pyarrow", index=False)

    logger.info(f"Event study analysis completed for {len(res_df)} events. Saved to {output_csv_path} and {output_parquet_path}.")
    return res_df


if __name__ == "__main__":
    run_event_study_analysis()
