"""
Macroeconomic Preprocessing and Trading-Date Alignment Pipeline Module
Processes raw FRED macroeconomic indicators and aligns them with market trading dates:
- Handles missing observations (FRED '.' placeholders)
- Preserves native observation frequencies (Daily, Monthly, Quarterly)
- Computes derived macroeconomic metrics (YoY CPI inflation, YoY GDP growth)
- ALIGNS WITH MARKET TRADING DATES WITH STRICT ZERO LOOK-AHEAD BIAS
  Uses publication lag modeling and backward as-of merges to ensure market participants
  at trading date T only observe data officially released on or before date T.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.ingestion.fred_data import FRED_SERIES_CATALOG, DEFAULT_FRED_RAW_DIR

logger = logging.getLogger(__name__)

DEFAULT_PROCESSED_MACRO_DIR = "data/processed/macro"
NATIVE_MACRO_PARQUET = "data/processed/macro/macro_indicators_native.parquet"
NATIVE_MACRO_CSV = "data/processed/macro/macro_indicators_native.csv"
ALIGNED_MACRO_PARQUET = "data/processed/macro/macro_indicators_aligned.parquet"
ALIGNED_MACRO_CSV = "data/processed/macro/macro_indicators_aligned.csv"
MARKET_MACRO_ALIGNED_PARQUET = "data/processed/macro/market_macro_aligned.parquet"


def calculate_publication_date(
    observation_date_str: str,
    frequency: str,
    publication_lag_days: int
) -> str:
    """
    Compute point-in-time publication/release date from observation reference date
    to eliminate look-ahead bias:
    - Daily: Available on observation date (or same day close).
    - Monthly: Reference date YYYY-MM-01 represents reference month.
      Reference month ends on month end (e.g. Jan 31).
      Release date = Month End + lag_days (e.g. Jan 31 + 15 days = Feb 15).
    - Quarterly: Reference date YYYY-MM-01 represents quarter start.
      Quarter ends on last day of 3rd month (e.g. Q1 ends Mar 31).
      Release date = Quarter End + lag_days (e.g. Mar 31 + 30 days = Apr 30).
    """
    obs_dt = pd.Timestamp(observation_date_str)
    freq = frequency.capitalize()

    if freq == "Daily":
        pub_dt = obs_dt + pd.Timedelta(days=publication_lag_days)
    elif freq == "Monthly":
        # Period end is last day of reference month
        period_end = (obs_dt + pd.offsets.MonthEnd(1)).normalize()
        pub_dt = period_end + pd.Timedelta(days=publication_lag_days)
    elif freq == "Quarterly":
        # Period end is last day of reference quarter
        period_end = (obs_dt + pd.offsets.QuarterEnd(startingMonth=3)).normalize()
        pub_dt = period_end + pd.Timedelta(days=publication_lag_days)
    else:
        pub_dt = obs_dt + pd.Timedelta(days=publication_lag_days)

    return pub_dt.strftime("%Y-%m-%d")


def clean_fred_series_dataframe(
    df: pd.DataFrame,
    series_id: str
) -> pd.DataFrame:
    """
    Clean raw FRED series:
    - Replaces missing values with NaN
    - Ensures sorted date indexing
    - Calculates publication dates for look-ahead bias prevention
    - Computes derived YoY rates for price indices and levels
    """
    cleaned = df.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    cleaned = cleaned.dropna(subset=["date"])
    cleaned["value"] = pd.to_numeric(cleaned["value"], errors="coerce")
    cleaned = cleaned.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)

    meta = FRED_SERIES_CATALOG.get(series_id, {})
    freq = meta.get("frequency", "Unknown")
    lag_days = meta.get("publication_lag_days", 0)

    # Populate metadata columns if missing
    if "series_id" not in cleaned.columns:
        cleaned["series_id"] = series_id
    if "frequency" not in cleaned.columns:
        cleaned["frequency"] = freq
    if "indicator_name" not in cleaned.columns:
        cleaned["indicator_name"] = meta.get("indicator_name", series_id.lower())
    if "category" not in cleaned.columns:
        cleaned["category"] = meta.get("category", "Macroeconomics")
    if "units" not in cleaned.columns and "units" in meta:
        cleaned["units"] = meta["units"]

    # Compute point-in-time publication date
    cleaned["publication_date"] = cleaned["date"].apply(
        lambda d: calculate_publication_date(d, freq, lag_days)
    )

    # Derived metrics
    # YoY Inflation for CPI series
    if series_id in ("CPIAUCSL", "CPILFESL"):
        # Monthly 12-period shift for YoY percentage change
        cleaned["yoy_change_pct"] = (cleaned["value"].pct_change(12) * 100.0).round(2)
        cleaned["mom_change_pct"] = (cleaned["value"].pct_change(1) * 100.0).round(2)
    elif series_id == "GDPC1":
        # Quarterly 4-period shift for YoY GDP growth
        cleaned["yoy_change_pct"] = (cleaned["value"].pct_change(4) * 100.0).round(2)

    return cleaned


def build_native_frequency_macro_dataset(
    raw_dir: str = DEFAULT_FRED_RAW_DIR,
    output_parquet: str = NATIVE_MACRO_PARQUET,
    output_csv: str = NATIVE_MACRO_CSV
) -> pd.DataFrame:
    """
    Build a comprehensive dataset containing all indicators at their native observation
    frequencies, preserving original dates, units, and categories.
    """
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    frames = []

    if not os.path.exists(raw_dir):
        logger.warning(f"Raw FRED directory not found: {raw_dir}")
        return pd.DataFrame()

    csv_files = [f for f in os.listdir(raw_dir) if f.endswith("_raw.csv")]

    for f in sorted(csv_files):
        sid = f.replace("_raw.csv", "")
        raw_path = os.path.join(raw_dir, f)
        try:
            raw_df = pd.read_csv(raw_path)
            cleaned = clean_fred_series_dataframe(raw_df, sid)
            frames.append(cleaned)
        except Exception as e:
            logger.error(f"Error processing native FRED series '{sid}': {e}")

    if not frames:
        return pd.DataFrame()

    native_df = pd.concat(frames, ignore_index=True)
    native_df = native_df.sort_values(["date", "series_id"]).reset_index(drop=True)

    native_df.to_parquet(output_parquet, engine="pyarrow", index=False)
    native_df.to_csv(output_csv, index=False)
    logger.info(f"Saved native frequency macro dataset to {output_parquet} ({len(native_df)} rows).")

    return native_df


def align_macro_with_trading_dates(
    raw_fred_dir: str = DEFAULT_FRED_RAW_DIR,
    trading_dates: Optional[List[str]] = None,
    sp500_parquet_path: str = "data/processed/sp500_processed.parquet",
    output_parquet: str = ALIGNED_MACRO_PARQUET,
    output_csv: str = ALIGNED_MACRO_CSV
) -> pd.DataFrame:
    """
    Align macroeconomic indicators with NYSE market trading dates with ZERO look-ahead bias.

    Algorithm:
    1. Extracts the active NYSE trading calendar from processed S&P 500 data.
    2. For each economic indicator, assigns its official publication date (when the market learned the number).
    3. Performs a backward as-of merge:
       For every trading day T:
       Indicator(T) = Observation with MAX(publication_date) where publication_date <= T.
    4. Forward-fills values between release dates (the macro state of the world as known on trading day T).
    """
    # Obtain trading calendar
    if trading_dates is None:
        if os.path.exists(sp500_parquet_path):
            sp_df = pd.read_parquet(sp500_parquet_path)
            dates = pd.to_datetime(sp_df["Date"]).dt.strftime("%Y-%m-%d").drop_duplicates().sort_values().tolist()
        else:
            # Fallback to standard business days if sp500 not yet available
            dates = pd.date_range("2015-01-01", "2026-02-01", freq="B").strftime("%Y-%m-%d").tolist()
    else:
        dates = sorted(list(set(trading_dates)))

    calendar_df = pd.DataFrame({"Date": dates}).sort_values("Date").reset_index(drop=True)

    # Process each FRED indicator into point-in-time series
    macro_aligned = calendar_df.copy()

    if not os.path.exists(raw_fred_dir):
        logger.warning(f"Raw FRED directory missing: {raw_fred_dir}")
        return macro_aligned

    csv_files = [f for f in os.listdir(raw_fred_dir) if f.endswith("_raw.csv")]

    for f in sorted(csv_files):
        sid = f.replace("_raw.csv", "")
        meta = FRED_SERIES_CATALOG.get(sid, {})
        ind_name = meta.get("indicator_name", sid.lower())
        raw_path = os.path.join(raw_fred_dir, f)

        try:
            raw_df = pd.read_csv(raw_path)
            cleaned = clean_fred_series_dataframe(raw_df, sid)

            # Drop rows with NaN values for alignment
            valid_obs = cleaned.dropna(subset=["value", "publication_date"]).copy()
            if valid_obs.empty:
                continue

            # Sort by publication_date for merge_asof
            valid_obs = valid_obs.sort_values("publication_date").reset_index(drop=True)

            # Look-ahead bias prevention:
            # Match trading date 'Date' to 'publication_date' with direction='backward'
            pit_df = valid_obs[["publication_date", "value"]].rename(
                columns={"publication_date": "Date", "value": ind_name}
            )

            # If CPI or GDP has YoY change, also align that
            if "yoy_change_pct" in valid_obs.columns:
                pit_df[f"{ind_name}_yoy_pct"] = valid_obs["yoy_change_pct"]

            # Merge onto trading calendar using backward asof
            # Convert to Timestamp for merge_asof
            cal_temp = calendar_df.copy()
            cal_temp["_merge_dt"] = pd.to_datetime(cal_temp["Date"])
            pit_temp = pit_df.copy()
            pit_temp["_merge_dt"] = pd.to_datetime(pit_temp["Date"])
            pit_temp = pit_temp.drop(columns=["Date"])

            merged = pd.merge_asof(
                cal_temp,
                pit_temp,
                on="_merge_dt",
                direction="backward"
            )
            merged = merged.drop(columns=["_merge_dt"])

            # Attach indicator column(s) to macro_aligned
            for c in merged.columns:
                if c != "Date":
                    macro_aligned[c] = merged[c]

        except Exception as e:
            logger.error(f"Error aligning FRED series '{sid}': {e}")

    # Additional economic derived indicators
    if "treasury_yield_10y" in macro_aligned.columns and "treasury_yield_2y" in macro_aligned.columns:
        macro_aligned["yield_curve_spread_calculated"] = (
            macro_aligned["treasury_yield_10y"] - macro_aligned["treasury_yield_2y"]
        ).round(2)
        macro_aligned["yield_curve_inverted"] = (macro_aligned["yield_curve_spread_calculated"] < 0)

    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    macro_aligned.to_parquet(output_parquet, engine="pyarrow", index=False)
    macro_aligned.to_csv(output_csv, index=False)
    logger.info(f"Saved aligned macroeconomic dataset to {output_parquet} ({len(macro_aligned)} rows).")

    return macro_aligned


def align_macro_with_market_indices(
    macro_aligned_df: pd.DataFrame,
    sp500_parquet_path: str = "data/processed/sp500_processed.parquet",
    vix_parquet_path: str = "data/processed/vix_processed.parquet",
    output_parquet: str = MARKET_MACRO_ALIGNED_PARQUET
) -> pd.DataFrame:
    """
    Combine market benchmark data (S&P 500, VIX) with aligned macroeconomic indicators
    into a unified dataset for econometric regression and event analysis.
    """
    if macro_aligned_df.empty or "Date" not in macro_aligned_df.columns:
        return pd.DataFrame()

    aligned = macro_aligned_df.copy()
    aligned["Date"] = pd.to_datetime(aligned["Date"]).dt.strftime("%Y-%m-%d")

    # Load S&P 500
    if os.path.exists(sp500_parquet_path):
        sp_df = pd.read_parquet(sp500_parquet_path)
        sp_df["Date"] = pd.to_datetime(sp_df["Date"]).dt.strftime("%Y-%m-%d")
        sp_sub = sp_df[["Date", "Close", "daily_return_pct", "rolling_volatility_21d"]].rename(
            columns={
                "Close": "sp500_close",
                "daily_return_pct": "sp500_daily_return_pct",
                "rolling_volatility_21d": "sp500_rolling_vol_21d"
            }
        )
        aligned = pd.merge(aligned, sp_sub, on="Date", how="inner")

    # Load VIX
    if os.path.exists(vix_parquet_path):
        vix_df = pd.read_parquet(vix_parquet_path)
        vix_df["Date"] = pd.to_datetime(vix_df["Date"]).dt.strftime("%Y-%m-%d")
        vix_sub = vix_df[["Date", "Close"]].rename(columns={"Close": "vix_level"})
        aligned = pd.merge(aligned, vix_sub, on="Date", how="left")

    aligned = aligned.sort_values("Date").reset_index(drop=True)

    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    aligned.to_parquet(output_parquet, engine="pyarrow", index=False)
    logger.info(f"Saved master market-macro aligned dataset to {output_parquet} ({len(aligned)} rows).")

    return aligned


def run_macro_pipeline(
    raw_fred_dir: str = DEFAULT_FRED_RAW_DIR,
    output_dir: str = DEFAULT_PROCESSED_MACRO_DIR
) -> Dict[str, pd.DataFrame]:
    """
    Execute full macroeconomic preprocessing and market alignment pipeline:
    1. Build native frequency dataset
    2. Build trading-date aligned dataset (zero look-ahead bias)
    3. Build market-macro aligned dataset
    """
    os.makedirs(output_dir, exist_ok=True)

    native_df = build_native_frequency_macro_dataset(raw_dir=raw_fred_dir)
    aligned_df = align_macro_with_trading_dates(raw_fred_dir=raw_fred_dir)
    master_df = align_macro_with_market_indices(aligned_df)

    return {
        "native": native_df,
        "aligned": aligned_df,
        "market_macro": master_df
    }


if __name__ == "__main__":
    run_macro_pipeline()
