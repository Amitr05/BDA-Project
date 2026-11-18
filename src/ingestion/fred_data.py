"""
FRED (Federal Reserve Economic Data) Ingestion Module
Retrieves macroeconomic indicators from the Federal Reserve Bank of St. Louis.

Selected Historical Indicators:
1. Inflation / CPI:
   - CPIAUCSL: Consumer Price Index for All Urban Consumers (Monthly)
   - CPILFESL: Core CPI excluding Food and Energy (Monthly)
2. Federal Funds Rate:
   - FEDFUNDS: Federal Funds Effective Rate (Monthly)
   - DFF: Federal Funds Effective Rate (Daily)
3. Treasury Yields:
   - DGS10: 10-Year Treasury Constant Maturity Rate (Daily)
   - DGS2: 2-Year Treasury Constant Maturity Rate (Daily)
   - T10Y2Y: 10-Year Minus 2-Year Treasury Spread (Daily)
   - DGS3MO: 3-Month Treasury Bill Secondary Market Rate (Daily)
4. Unemployment:
   - UNRATE: Civilian Unemployment Rate (Monthly)
5. GDP Growth:
   - A191RL1Q225SBEA: Real GDP Percent Change from Preceding Period Annualized (Quarterly)
   - GDPC1: Real Gross Domestic Product Level (Quarterly)
6. Consumer Confidence:
   - UMCSENT: University of Michigan Consumer Sentiment (Monthly)

Implementation Features:
- Official FRED REST API integration with environment variable key management (FRED_API_KEY).
- Graceful public endpoint fallback when no API key is configured.
- Preservation of original observation dates and frequencies in raw datasets.
- Handling of missing observations (FRED '.' placeholders).
- Incremental updates avoiding duplicate records.
"""

import os
import io
import sys
import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

DEFAULT_START_DATE = "2015-01-01"
DEFAULT_END_DATE = "2026-02-01"
DEFAULT_FRED_RAW_DIR = "data/raw/fred"
FRED_LOG_JSON = "data/raw/fred_ingestion_log.json"
FRED_MANIFEST_JSON = "data/raw/fred/fred_manifest.json"

OFFICIAL_FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
PUBLIC_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# Comprehensive Catalog of Selected FRED Macroeconomic Indicators
FRED_SERIES_CATALOG: Dict[str, Dict[str, Any]] = {
    "CPIAUCSL": {
        "indicator_name": "cpi_headline",
        "title": "Consumer Price Index for All Urban Consumers: All Items",
        "frequency": "Monthly",
        "units": "Index 1982-1984=100",
        "seasonal_adjustment": "Seasonally Adjusted",
        "category": "Inflation",
        # Monthly CPI is released approx. 14-16 days after reference month ends
        "publication_lag_days": 15
    },
    "CPILFESL": {
        "indicator_name": "cpi_core",
        "title": "Consumer Price Index for All Urban Consumers: All Items Less Food and Energy",
        "frequency": "Monthly",
        "units": "Index 1982-1984=100",
        "seasonal_adjustment": "Seasonally Adjusted",
        "category": "Inflation",
        "publication_lag_days": 15
    },
    "FEDFUNDS": {
        "indicator_name": "fed_funds_rate_monthly",
        "title": "Federal Funds Effective Rate (Monthly Average)",
        "frequency": "Monthly",
        "units": "Percent",
        "seasonal_adjustment": "Not Seasonally Adjusted",
        "category": "Monetary Policy",
        "publication_lag_days": 1
    },
    "DFF": {
        "indicator_name": "fed_funds_rate_daily",
        "title": "Federal Funds Effective Rate (Daily)",
        "frequency": "Daily",
        "units": "Percent",
        "seasonal_adjustment": "Not Seasonally Adjusted",
        "category": "Monetary Policy",
        "publication_lag_days": 1
    },
    "DGS10": {
        "indicator_name": "treasury_yield_10y",
        "title": "Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity",
        "frequency": "Daily",
        "units": "Percent",
        "seasonal_adjustment": "Not Seasonally Adjusted",
        "category": "Treasury Yields",
        "publication_lag_days": 0
    },
    "DGS2": {
        "indicator_name": "treasury_yield_2y",
        "title": "Market Yield on U.S. Treasury Securities at 2-Year Constant Maturity",
        "frequency": "Daily",
        "units": "Percent",
        "seasonal_adjustment": "Not Seasonally Adjusted",
        "category": "Treasury Yields",
        "publication_lag_days": 0
    },
    "T10Y2Y": {
        "indicator_name": "yield_curve_spread_10y2y",
        "title": "10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity Spread",
        "frequency": "Daily",
        "units": "Percent",
        "seasonal_adjustment": "Not Seasonally Adjusted",
        "category": "Treasury Yields",
        "publication_lag_days": 0
    },
    "DGS3MO": {
        "indicator_name": "treasury_yield_3m",
        "title": "Market Yield on U.S. Treasury Securities at 3-Month Constant Maturity",
        "frequency": "Daily",
        "units": "Percent",
        "seasonal_adjustment": "Not Seasonally Adjusted",
        "category": "Treasury Yields",
        "publication_lag_days": 0
    },
    "UNRATE": {
        "indicator_name": "unemployment_rate",
        "title": "Civilian Unemployment Rate",
        "frequency": "Monthly",
        "units": "Percent",
        "seasonal_adjustment": "Seasonally Adjusted",
        "category": "Labor Market",
        # First Friday of following month (~5 to 8 days)
        "publication_lag_days": 7
    },
    "A191RL1Q225SBEA": {
        "indicator_name": "gdp_growth_annualized",
        "title": "Real Gross Domestic Product: Percent Change from Preceding Period",
        "frequency": "Quarterly",
        "units": "Percent Change at Annual Rate",
        "seasonal_adjustment": "Seasonally Adjusted Annual Rate",
        "category": "Economic Output",
        # Advance estimate released ~28-30 days after quarter ends
        "publication_lag_days": 30
    },
    "GDPC1": {
        "indicator_name": "real_gdp",
        "title": "Real Gross Domestic Product",
        "frequency": "Quarterly",
        "units": "Billions of Chained 2017 Dollars",
        "seasonal_adjustment": "Seasonally Adjusted Annual Rate",
        "category": "Economic Output",
        "publication_lag_days": 30
    },
    "UMCSENT": {
        "indicator_name": "consumer_sentiment",
        "title": "University of Michigan: Consumer Sentiment",
        "frequency": "Monthly",
        "units": "Index 1966:Q1=100",
        "seasonal_adjustment": "Not Seasonally Adjusted",
        "category": "Consumer Confidence",
        # Preliminary release around mid-month (~15th), final near end (~28th)
        "publication_lag_days": 15
    }
}


def get_fred_api_key() -> Optional[str]:
    """
    Retrieve the official FRED API key from environment variables.
    Checks 'FRED_API_KEY' in os.environ.
    """
    key = os.environ.get("FRED_API_KEY", "").strip()
    return key if key else None


def get_fred_catalog() -> pd.DataFrame:
    """Return a DataFrame describing all configured FRED macroeconomic series."""
    rows = []
    for sid, info in FRED_SERIES_CATALOG.items():
        rows.append({
            "series_id": sid,
            "indicator_name": info["indicator_name"],
            "title": info["title"],
            "frequency": info["frequency"],
            "units": info["units"],
            "seasonal_adjustment": info["seasonal_adjustment"],
            "category": info["category"],
            "publication_lag_days": info["publication_lag_days"]
        })
    return pd.DataFrame(rows)


def parse_fred_observations(
    raw_data: List[Dict[str, Any]],
    series_id: str
) -> pd.DataFrame:
    """
    Parse observation records from FRED into a standardized DataFrame.
    Replaces '.' and empty values with NaN.
    Preserves original observation dates and native frequency.
    """
    meta = FRED_SERIES_CATALOG.get(series_id, {})
    records = []

    for item in raw_data:
        dt = item.get("date")
        val = item.get("value")

        if not dt:
            continue

        # Handle FRED missing values marked as '.' or empty
        num_val = np.nan
        if val is not None and str(val).strip() not in (".", "", "ND", "null", "None"):
            try:
                num_val = float(val)
            except (ValueError, TypeError):
                num_val = np.nan

        records.append({
            "date": str(dt).strip(),
            "value": num_val,
            "series_id": series_id,
            "indicator_name": meta.get("indicator_name", series_id.lower()),
            "frequency": meta.get("frequency", "Unknown"),
            "category": meta.get("category", "Macroeconomics")
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df = df.dropna(subset=["date"])
        df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)

    return df


def fetch_fred_via_official_api(
    series_id: str,
    api_key: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    timeout: int = 15
) -> pd.DataFrame:
    """
    Fetch historical observations using the official St. Louis Fed FRED REST API.
    GET https://api.stlouisfed.org/fred/series/observations
    """
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
    }

    logger.info(f"Querying official FRED API for series '{series_id}' ({start_date} to {end_date})...")
    resp = requests.get(OFFICIAL_FRED_API_URL, params=params, timeout=timeout)

    if resp.status_code != 200:
        raise RuntimeError(
            f"FRED API returned HTTP {resp.status_code} for series '{series_id}': {resp.text[:300]}"
        )

    data = resp.json()
    observations = data.get("observations", [])
    logger.info(f"Official FRED API returned {len(observations)} raw observations for '{series_id}'.")
    return parse_fred_observations(observations, series_id)


def fetch_fred_via_public_csv(
    series_id: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    timeout: int = 15
) -> pd.DataFrame:
    """
    Fetch observations using the public FRED CSV download endpoint.
    Used when no API key is configured or as fallback.
    GET https://fred.stlouisfed.org/graph/fredgraph.csv?id=<series_id>
    """
    params = {"id": series_id}
    logger.info(f"Querying public FRED endpoint for series '{series_id}'...")
    resp = requests.get(PUBLIC_FRED_CSV_URL, params=params, timeout=timeout)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Public FRED endpoint returned HTTP {resp.status_code} for series '{series_id}': {resp.text[:300]}"
        )

    csv_text = resp.text
    raw_df = pd.read_csv(io.StringIO(csv_text))

    if len(raw_df.columns) < 2:
        raise ValueError(f"Unexpected CSV format received from public FRED for '{series_id}'.")

    date_col = raw_df.columns[0]
    val_col = raw_df.columns[1]

    raw_records = []
    for _, row in raw_df.iterrows():
        raw_records.append({
            "date": row[date_col],
            "value": row[val_col]
        })

    df = parse_fred_observations(raw_records, series_id)

    # Filter to requested date range
    if not df.empty:
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].reset_index(drop=True)

    logger.info(f"Public FRED endpoint retrieved {len(df)} observations for '{series_id}' within {start_date} to {end_date}.")
    return df


def fetch_fred_series(
    series_id: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    api_key: Optional[str] = None,
    max_retries: int = 3
) -> pd.DataFrame:
    """
    Fetch historical observations for a given FRED series identifier.
    Uses official FRED API when key is provided or present in environment.
    Falls back to public FRED CSV endpoint seamlessly.
    """
    key = api_key or get_fred_api_key()

    for attempt in range(1, max_retries + 1):
        try:
            if key:
                try:
                    return fetch_fred_via_official_api(series_id, key, start_date=start_date, end_date=end_date)
                except Exception as api_err:
                    logger.warning(f"Official FRED API call failed ({api_err}), attempting public CSV fallback...")
                    return fetch_fred_via_public_csv(series_id, start_date=start_date, end_date=end_date)
            else:
                return fetch_fred_via_public_csv(series_id, start_date=start_date, end_date=end_date)
        except Exception as err:
            wait_sec = 1.5 * attempt
            logger.warning(f"Attempt {attempt}/{max_retries} failed for FRED series '{series_id}': {err}. Retrying in {wait_sec:.1f}s...")
            time.sleep(wait_sec)

    raise RuntimeError(f"All {max_retries} download attempts failed for FRED series '{series_id}'.")


def save_fred_series_incremental(
    series_id: str,
    new_df: pd.DataFrame,
    output_dir: str = DEFAULT_FRED_RAW_DIR
) -> str:
    """
    Save or merge observations incrementally into data/raw/fred/<series_id>_raw.csv.
    Preserves original dates and native frequencies while deduplicating records.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{series_id}_raw.csv")

    if os.path.exists(file_path):
        try:
            existing_df = pd.read_csv(file_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df["date"] = pd.to_datetime(combined_df["date"]).dt.strftime("%Y-%m-%d")
            # Deduplicate keeping latest
            combined_df = combined_df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
            combined_df.to_csv(file_path, index=False)
            logger.info(f"Incrementally updated {file_path} (Total observations: {len(combined_df)}).")
            return file_path
        except Exception as e:
            logger.warning(f"Error merging incremental FRED data for {series_id}: {e}. Overwriting with fresh data.")

    new_df["date"] = pd.to_datetime(new_df["date"]).dt.strftime("%Y-%m-%d")
    new_df = new_df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    new_df.to_csv(file_path, index=False)
    logger.info(f"Saved fresh FRED series {file_path} ({len(new_df)} observations).")
    return file_path


def ingest_all_fred_indicators(
    series_list: Optional[List[str]] = None,
    output_dir: str = DEFAULT_FRED_RAW_DIR,
    force_download: bool = False,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Ingest all selected FRED macroeconomic indicators:
    - Research identifiers from catalog
    - Authenticate via environment or parameter
    - Maintain incremental caching
    - Save raw CSV per indicator preserving native dates and frequencies
    - Output manifest and audit log
    """
    targets = series_list or list(FRED_SERIES_CATALOG.keys())
    os.makedirs(output_dir, exist_ok=True)
    key = api_key or get_fred_api_key()

    logger.info("=" * 60)
    logger.info(f"STARTING FRED MACROECONOMIC INGESTION ({len(targets)} series)")
    logger.info(f"API Key configured: {bool(key)} | Output: {output_dir}")
    logger.info(f"Date Range: {start_date} to {end_date}")
    logger.info("=" * 60)

    results: Dict[str, str] = {}
    failures: Dict[str, str] = {}
    skipped_cached: List[str] = []

    for sid in targets:
        raw_path = os.path.join(output_dir, f"{sid}_raw.csv")

        # Caching check
        if not force_download and os.path.exists(raw_path) and os.path.getsize(raw_path) > 100:
            try:
                existing = pd.read_csv(raw_path)
                if len(existing) > 10 and "date" in existing.columns and "value" in existing.columns:
                    # Check date coverage
                    max_d = existing["date"].max()
                    if max_d >= "2024-01-01":
                        logger.info(f"Valid local cache found for FRED series '{sid}' ({len(existing)} rows). Skipping download.")
                        skipped_cached.append(sid)
                        results[sid] = raw_path
                        continue
            except Exception:
                pass

        try:
            df = fetch_fred_series(sid, start_date=start_date, end_date=end_date, api_key=key)
            if not df.empty:
                path = save_fred_series_incremental(sid, df, output_dir=output_dir)
                results[sid] = path
            else:
                failures[sid] = "Empty dataset returned"
        except Exception as e:
            failures[sid] = str(e)
            logger.error(f"Failed to ingest FRED series '{sid}': {e}")

    # Build manifest
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "api_key_used": bool(key),
        "total_series": len(targets),
        "successful_series": len(results),
        "skipped_from_cache": len(skipped_cached),
        "newly_downloaded": len(results) - len(skipped_cached),
        "failed_series": failures,
        "series_details": {
            sid: {
                "path": results.get(sid),
                "metadata": FRED_SERIES_CATALOG.get(sid, {})
            } for sid in results
        }
    }

    with open(FRED_MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(FRED_LOG_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"FRED INGESTION COMPLETE: {len(results)}/{len(targets)} Successful, {len(failures)} Failed")
    logger.info("=" * 60)

    return manifest


if __name__ == "__main__":
    force = "--force-download" in sys.argv
    ingest_all_fred_indicators(force_download=force)
