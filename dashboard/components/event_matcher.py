"""
Three-tier Event Identifier Matching Logic for Multi-Dataset Cross-Referencing
"""

from typing import Optional
import pandas as pd


def match_event_subset(df: Optional[pd.DataFrame], ev_name: str, ev_date_str: str) -> pd.DataFrame:
    """
    Match an event in a secondary dataframe using a robust 3-tier matching strategy:
    1. Exact match on 'event_name'
    2. Date match on 'effective_trading_date', 'event_date', or 'calendar_event_date'
    3. Canonical keyword/slug substring match (e.g. 'russia-ukraine', 'brexit', 'covid', 'svb')
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # 1. Exact match
    if "event_name" in df.columns:
        exact = df[df["event_name"] == ev_name]
        if not exact.empty:
            return exact

    # 2. Date match
    for d_col in ["effective_trading_date", "event_date", "calendar_event_date"]:
        if d_col in df.columns:
            dt_series = pd.to_datetime(df[d_col], errors="coerce").dt.strftime("%Y-%m-%d")
            d_match = df[dt_series == ev_date_str]
            if not d_match.empty:
                return d_match

    # 3. Canonical keyword / slug match
    norm_target = ev_name.lower().replace("\u2013", "-").replace("\u2014", "-")
    key_patterns = [
        ("russia", "ukraine"),
        ("brexit",),
        ("covid",),
        ("banking", "crisis"),
        ("svb",),
        ("trade", "war"),
        ("gaza",),
        ("israel", "hamas"),
        ("yen", "carry"),
    ]
    target_keys = [kw for p in key_patterns if all(w in norm_target for w in p) for kw in p]
    if target_keys and "event_name" in df.columns:
        for _, row in df.iterrows():
            cand = str(row["event_name"]).lower().replace("\u2013", "-").replace("\u2014", "-")
            if any(k in cand for k in target_keys):
                return df[df["event_name"] == row["event_name"]]

    return pd.DataFrame()


# Backward-compatible alias
_match_event_subset = match_event_subset
