
"""Event-study analytics for US equity indices and the VIX."""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATHS = {
    "sp500": "data/processed/sp500_processed.parquet",
    "nasdaq": "data/processed/nasdaq_processed.parquet",
    "dow_jones": "data/processed/dow_jones_processed.parquet",
    "russell2000": "data/processed/russell2000_processed.parquet",
}

INDEX_LABELS = {
    "sp500": "S&P 500",
    "nasdaq": "NASDAQ Composite",
    "dow_jones": "Dow Jones Industrial Average",
    "russell2000": "Russell 2000",
}

INDEX_TICKERS = {
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "dow_jones": "^DJI",
    "russell2000": "^RUT",
}


def _load_parquet(path: str, description: str) -> pd.DataFrame:
    """Load and validate a processed market-data Parquet file."""

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed {description} file not found at {path}"
        )

    frame = pd.read_parquet(path)

    required = {"Date", "Close"}
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Processed {description} data is missing columns: "
            f"{sorted(missing)}"
        )

    frame = frame.copy()

    frame["Date"] = (
        pd.to_datetime(frame["Date"], errors="coerce")
        .dt.strftime("%Y-%m-%d")
    )

    frame["Close"] = pd.to_numeric(
        frame["Close"], errors="coerce"
    )

    frame = frame.dropna(
        subset=["Date", "Close"]
    )

    frame = frame.sort_values("Date")

    frame = frame.drop_duplicates(
        subset=["Date"],
        keep="last"
    ).reset_index(drop=True)

    if frame.empty:
        raise ValueError(
            f"Processed {description} dataset contains "
            "no valid Date/Close rows"
        )

    return frame


def load_processed_data(
    sp500_path: str = DEFAULT_INDEX_PATHS["sp500"],
    vix_path: str = "data/processed/vix_processed.parquet",
    index_parquet_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load S&P 500, VIX, and additional equity-index datasets.

    Legacy S&P 500 and VIX arguments remain supported.
    """

    data = {
        "sp500": _load_parquet(
            sp500_path, "S&P 500"
        )
    }

    if index_parquet_paths:
        for key, path in index_parquet_paths.items():

            if key not in INDEX_LABELS:
                raise ValueError(
                    f"Unsupported equity index key: {key}"
                )

            if key == "sp500":
                continue

            data[key] = _load_parquet(
                path, INDEX_LABELS[key]
            )

    data["vix"] = _load_parquet(
        vix_path, "VIX"
    )

    return data


def find_effective_trading_day(
    target_date: str,
    trading_dates: List[str]
) -> Optional[int]:
    """
    Return the index of the first trading day on or after
    the specified event date.
    """

    target = pd.Timestamp(target_date).strftime(
        "%Y-%m-%d"
    )

    for idx, date in enumerate(trading_dates):
        if date >= target:
            return idx

    return None


def calculate_event_metrics(
    event: Dict[str, Any],
    sp500_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    pre_window: int = 5,
    post_window_short: int = 5,
    post_window_long: int = 30,
    index_name: str = "S&P 500",
    index_key: str = "sp500",
) -> Dict[str, Any]:
    """
    Calculate event metrics for one equity index.

    Each index uses its own trading calendar.

    Legacy S&P 500 result columns are retained for
    backward compatibility.

    Incomplete post-event windows are reported as None.
    """

    if index_key not in INDEX_LABELS:
        raise ValueError(
            f"Unsupported equity index key: {index_key}"
        )

    index_df = sp500_df

    dates = index_df["Date"].tolist()

    calendar_date = pd.Timestamp(
        event["event_date"]
    ).strftime("%Y-%m-%d")

    t0_idx = find_effective_trading_day(
        calendar_date, dates
    )

    if t0_idx is None:
        logger.warning(
            "Event '%s' (%s) is beyond %s dataset range",
            event.get("event_name"),
            calendar_date,
            index_name
        )
        return {}

    t0_date = dates[t0_idx]

    n_days = len(dates)

    # Preserve original pre-event behavior.
    pre_idx = max(0, t0_idx - pre_window)

    short_idx = t0_idx + post_window_short
    long_idx = t0_idx + post_window_long

    pre_available = True
    short_available = short_idx < n_days
    long_available = long_idx < n_days

    p0 = float(
        index_df.loc[t0_idx, "Close"]
    )

    pre_price = float(
        index_df.loc[pre_idx, "Close"]
    )

    short_price = (
        float(index_df.loc[short_idx, "Close"])
        if short_available else np.nan
    )

    long_price = (
        float(index_df.loc[long_idx, "Close"])
        if long_available else np.nan
    )

    # Returns
    pre_return = (
        (p0 - pre_price) / pre_price * 100.0
        if pre_available and pre_price != 0
        else np.nan
    )

    short_return = (
        (short_price - p0) / p0 * 100.0
        if short_available and p0 != 0
        else np.nan
    )

    long_return = (
        (long_price - p0) / p0 * 100.0
        if long_available and p0 != 0
        else np.nan
    )

    # Preserve original drawdown definition:
    # lowest close from T0 through T+30,
    # relative to the T0 closing price.

    drawdown = np.nan

    if long_available and p0 != 0:

        lowest_close = float(
            index_df.loc[
                t0_idx:long_idx, "Close"
            ].min()
        )

        drawdown = (
            (lowest_close - p0) / p0 * 100.0
        )

    # Align VIX independently using calendar dates.

    vix_by_date = (
        vix_df.set_index("Date")["Close"].to_dict()
    )

    pre_date = (
        dates[pre_idx] if pre_available else None
    )

    short_date = (
        dates[short_idx] if short_available else None
    )

    long_date = (
        dates[long_idx] if long_available else None
    )

    vix_t0 = vix_by_date.get(
        t0_date, np.nan
    )

    vix_pre = (
        vix_by_date.get(pre_date, np.nan)
        if pre_date else np.nan
    )

    vix_short = (
        vix_by_date.get(short_date, np.nan)
        if short_date else np.nan
    )

    vix_long = (
        vix_by_date.get(long_date, np.nan)
        if long_date else np.nan
    )

    vix_window = [
        vix_by_date[d]
        for d in dates[t0_idx:long_idx + 1]
        if long_available and d in vix_by_date
    ]

    vix_max = (
        max(vix_window)
        if vix_window else np.nan
    )

    vix_change = (
        vix_short - vix_t0
        if pd.notna(vix_short)
        and pd.notna(vix_t0)
        else np.nan
    )

    def rounded(value):
        if pd.notna(value) and np.isfinite(value):
            return round(float(value), 2)
        return None

    result = {
        "event_name": event.get("event_name"),
        "calendar_event_date": calendar_date,
        "effective_trading_date": t0_date,
        "event_category": event.get("event_category"),
        "short_description": event.get("short_description"),

        "index": index_name,
        "index_key": index_key,
        "ticker": INDEX_TICKERS[index_key],

        "price_t0": rounded(p0),

        "pre_5d_date": pre_date,
        "pre_5d_return_pct": rounded(pre_return),

        "post_5d_date": short_date,
        "post_5d_return_pct": rounded(short_return),

        "post_30d_date": long_date,
        "post_30d_return_pct": rounded(long_return),

        "max_drawdown_30d_pct": rounded(drawdown),

        "vix_at_event": rounded(vix_t0),
        "vix_pre_5d": rounded(vix_pre),
        "vix_post_5d": rounded(vix_short),
        "vix_post_30d": rounded(vix_long),
        "vix_5d_change": rounded(vix_change),
        "vix_max_30d": rounded(vix_max),
    }

    # Backward compatibility for existing S&P 500 consumers.

    if index_key == "sp500":

        result.update({
            "sp500_price_t0": result["price_t0"],

            "sp500_pre_5d_return_pct":
                result["pre_5d_return_pct"],

            "sp500_post_5d_return_pct":
                result["post_5d_return_pct"],

            "sp500_post_30d_return_pct":
                result["post_30d_return_pct"],

            "sp500_max_drawdown_30d_pct":
                result["max_drawdown_30d_pct"],
        })

    return result


def run_event_study_analysis(
    events_csv_path: str = "data/raw/events.csv",
    sp500_parquet_path: str = DEFAULT_INDEX_PATHS["sp500"],
    vix_parquet_path: str = "data/processed/vix_processed.parquet",
    output_csv_path: str = "data/processed/event_study_results.csv",
    output_parquet_path: str = "data/processed/event_study_results.parquet",
    index_parquet_paths: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Run event-study analysis.

    If index_parquet_paths is supplied, all four equity
    indices are processed independently.

    If omitted, legacy S&P 500-only behavior is retained.
    """

    if not os.path.exists(events_csv_path):
        raise FileNotFoundError(
            f"Events file not found at {events_csv_path}"
        )

    events_df = pd.read_csv(events_csv_path)

    if "event_date" not in events_df.columns:
        raise ValueError(
            "Events CSV must contain an 'event_date' column"
        )

    paths = None

    if index_parquet_paths is not None:

        paths = dict(index_parquet_paths)

        paths.setdefault(
            "sp500", sp500_parquet_path
        )

        sp500_parquet_path = paths["sp500"]

        missing = set(INDEX_LABELS).difference(paths)

        if missing:
            raise ValueError(
                "Multi-index mode requires paths for all "
                f"four indices; missing: {sorted(missing)}"
            )

    data = load_processed_data(
        sp500_path=sp500_parquet_path,
        vix_path=vix_parquet_path,
        index_parquet_paths=paths
    )

    index_keys = (
        list(INDEX_LABELS)
        if paths is not None
        else ["sp500"]
    )

    results: List[Dict[str, Any]] = []

    for index_key in index_keys:

        index_df = data[index_key]

        for _, row in events_df.iterrows():

            metrics = calculate_event_metrics(
                row.to_dict(),
                index_df,
                data["vix"],
                index_name=INDEX_LABELS[index_key],
                index_key=index_key
            )

            if metrics:
                results.append(metrics)

    result_df = pd.DataFrame(results)

    # Create output directories if necessary.

    for output_path in (
        output_csv_path,
        output_parquet_path
    ):

        directory = os.path.dirname(output_path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

    result_df.to_csv(
        output_csv_path,
        index=False
    )

    result_df.to_parquet(
        output_parquet_path,
        engine="pyarrow",
        index=False
    )

    logger.info(
        "Event study completed for %d event/index records",
        len(result_df)
    )

    return result_df


if __name__ == "__main__":
    run_event_study_analysis()
