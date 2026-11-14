"""
Curated Event Dataset Module
Provides structured records for major historical events from 2015 to 2026.
"""

import os
import pandas as pd
from typing import List, Dict

CURATED_EVENTS: List[Dict[str, str]] = [
    {
        "event_name": "UK Brexit Referendum",
        "event_date": "2016-06-23",
        "event_category": "Geopolitical",
        "short_description": "UK citizens vote to leave the European Union triggering immediate global market uncertainty."
    },
    {
        "event_name": "US-China Trade War Escalation",
        "event_date": "2018-03-22",
        "event_category": "Trade Conflict",
        "short_description": "US announces Section 301 tariffs on $50B+ Chinese imports initiating multi-year trade tensions."
    },
    {
        "event_name": "COVID-19 Pandemic Declaration",
        "event_date": "2020-03-11",
        "event_category": "Global Health",
        "short_description": "WHO officially declares COVID-19 a global pandemic following rapid worldwide transmission and market turmoil."
    },
    {
        "event_name": "Russia-Ukraine Invasion",
        "event_date": "2022-02-24",
        "event_category": "Armed Conflict",
        "short_description": "Russian military forces launch full-scale invasion of Ukraine triggering energy and security shocks."
    },
    {
        "event_name": "US Regional Banking Crisis (SVB)",
        "event_date": "2023-03-10",
        "event_category": "Financial Shock",
        "short_description": "Regulators shut down Silicon Valley Bank following a sudden run on deposits sparking contagion fears."
    },
    {
        "event_name": "Israel-Hamas War Outbreak",
        "event_date": "2023-10-07",
        "event_category": "Armed Conflict",
        "short_description": "Hamas launches large-scale offensive against Israel escalating Middle East geopolitical conflict."
    },
    {
        "event_name": "Global Market Selloff (Yen Carry Trade)",
        "event_date": "2024-08-05",
        "event_category": "Financial Shock",
        "short_description": "Global market volatility spike and equity drop driven by Bank of Japan rate hike and carry trade unwind."
    }
]


def get_curated_events() -> pd.DataFrame:
    """Return the curated list of events as a pandas DataFrame."""
    df = pd.DataFrame(CURATED_EVENTS)
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.strftime("%Y-%m-%d")
    return df


def save_curated_events(output_path: str = "data/raw/events.csv", overwrite: bool = False) -> str:
    """
    Save curated events to CSV if not already present or if overwrite is requested.
    """
    if os.path.exists(output_path) and not overwrite:
        print(f"[INFO] Event dataset already exists at {output_path}.")
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = get_curated_events()
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Curated {len(df)} events saved to {output_path}.")
    return output_path


def load_event_dataset(file_path: str = "data/raw/events.csv") -> pd.DataFrame:
    """
    Load and validate event dataset from CSV.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Event file not found at {file_path}. Run save_curated_events() first.")

    df = pd.read_csv(file_path)
    required_cols = {"event_name", "event_date", "event_category", "short_description"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Event file is missing required columns: {missing}")

    df["event_date"] = pd.to_datetime(df["event_date"]).dt.strftime("%Y-%m-%d")
    return df
