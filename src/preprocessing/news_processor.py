"""
News Processing & Sentiment Standardization Pipeline
=====================================================
Transforms raw GDELT news records into a standardized financial news dataset.

Features:
  1. Standardized schema with publication timestamps, URLs, domains, and geography
  2. Entity extraction: organization names, company mentions, and ticker symbol alignment
  3. Distinction between publication timestamp and actual event occurrence date
  4. Temporal timing classification: 'pre-event', 'day-of', and 'post-event'
  5. Calibrated financial sentiment scoring from GDELT tone metrics
  6. Financial relevance scoring based on dictionary and thematic density
  7. Deduplication by unique article hash / URL
  8. Clean dual persistence to Parquet and CSV in data/processed/
"""

import hashlib
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_RAW_NEWS_PATH = "data/raw/gdelt/raw_news_records.jsonl"
DEFAULT_PROCESSED_NEWS_PARQUET = "data/processed/news_articles.parquet"
DEFAULT_PROCESSED_NEWS_CSV = "data/processed/news_articles.csv"
COMPANIES_CSV = "data/raw/companies.csv"
EVENTS_CSV = "data/raw/events_expanded.csv"

# Pre-compiled regex patterns for company and financial entity detection
COMPANY_TICKER_MAP = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "jpmorgan": "JPM",
    "goldman sachs": "GS",
    "bank of america": "BAC",
    "wells fargo": "WFC",
    "exxon": "XOM",
    "chevron": "CVX",
    "lockheed": "LMT",
    "northrop": "NOC",
    "rtx": "RTX",
    "raytheon": "RTX",
    "boeing": "BA",
    "caterpillar": "CAT",
    "deere": "DE",
    "intel": "INTC",
    "amd": "AMD",
    "qualcomm": "QCOM",
    "pfizer": "PFE",
    "biontech": "BNTX",
    "moderna": "MRNA",
    "take-two": "TTWO",
    "rockstar": "TTWO",
    "disney": "DIS",
    "netflix": "NFLX",
    "coinbase": "COIN",
    "robinhood": "HOOD",
    "gamestop": "GME",
    "silicon valley bank": "SIVB",
    "svb": "SIVB",
    "first republic": "FRC",
    "allstate": "ALL",
    "travelers": "TRV",
    "chubb": "CB",
}

FINANCIAL_TERMS = {
    "market", "stock", "stocks", "share", "shares", "equity", "equities",
    "investor", "investors", "trading", "trader", "traders", "wall street",
    "dow", "nasdaq", "s&p", "yield", "yields", "bond", "bonds", "treasury",
    "inflation", "rate", "rates", "fed", "federal reserve", "central bank",
    "recession", "gdp", "earnings", "revenue", "profit", "dividend", "dividends",
    "crude", "oil", "commodity", "volatility", "vix", "drawdown", "rally", "selloff"
}


# -----------------------------------------------------------------------------
# 1. Parsing & Normalization Helpers
# -----------------------------------------------------------------------------

def parse_gdelt_timestamp(seendate_str: str) -> Tuple[str, str]:
    """
    Parses GDELT seendate format (e.g. '20231008T101500Z' or '20231008101500')
    Returns (YYYY-MM-DD, ISO8601 string)
    """
    if not seendate_str:
        return ("", "")
    clean = re.sub(r"[^\d]", "", str(seendate_str))
    if len(clean) >= 8:
        date_part = f"{clean[0:4]}-{clean[4:6]}-{clean[6:8]}"
        if len(clean) >= 14:
            iso_str = f"{clean[0:4]}-{clean[4:6]}-{clean[6:8]}T{clean[8:10]}:{clean[10:12]}:{clean[12:14]}Z"
        else:
            iso_str = f"{date_part}T00:00:00Z"
        return (date_part, iso_str)
    return ("", "")


def generate_article_id(url: str, title: str, published_date: str) -> str:
    """Generate deterministic MD5 hash as article_id."""
    raw = f"{url.strip()}_{title.strip()}_{published_date}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def extract_matched_tickers(title: str, text_content: str, existing_tickers: Any = "") -> str:
    """
    Extracts relevant stock tickers from title and text using the company ticker map
    and existing assigned tickers.
    """
    found_tickers: Set[str] = set()

    if existing_tickers and pd.notna(existing_tickers):
        str_tickers = str(existing_tickers).strip()
        if str_tickers:
            for t in str_tickers.split("|"):
                if t.strip():
                    found_tickers.add(t.strip())

    combined_text = f"{str(title or '')} {str(text_content or '')}".lower()

    for name, ticker in COMPANY_TICKER_MAP.items():
        # Match whole word
        pattern = r"\b" + re.escape(name) + r"\b"
        if re.search(pattern, combined_text):
            found_tickers.add(ticker)

    return "|".join(sorted(found_tickers))


def compute_financial_relevance_score(title: str, organizations: Any) -> float:
    """
    Computes a 0.0 to 1.0 financial relevance score based on keyword density
    and presence of financial institutions.
    """
    org_str = str(organizations or "") if pd.notna(organizations) else ""
    words = re.findall(r"\w+", f"{str(title or '')} {org_str}".lower())
    if not words:
        return 0.1

    fin_matches = sum(1 for w in words if w in FINANCIAL_TERMS)
    ratio = fin_matches / max(len(words), 5)
    return min(1.0, round(ratio * 3.0, 2))


def classify_sentiment(tone: float) -> str:
    """
    Classifies GDELT tone score into Positive, Neutral, Negative.
    GDELT tone typically ranges between -10 and +10.
    """
    if tone >= 1.0:
        return "Positive"
    elif tone <= -1.0:
        return "Negative"
    else:
        return "Neutral"


def compute_event_timing_phase(
    published_date_str: str,
    event_date_str: str
) -> Tuple[Optional[int], str]:
    """
    Distinguishes news publication date from actual event date:
      - days_from_event: T_pub - T_event (in calendar days)
      - timing_phase: 'pre-event', 'day-of', or 'post-event'
    """
    if not published_date_str or not event_date_str:
        return (None, "unknown")

    try:
        pub_dt = datetime.strptime(published_date_str, "%Y-%m-%d").date()
        ev_dt = datetime.strptime(event_date_str, "%Y-%m-%d").date()
        delta = (pub_dt - ev_dt).days

        if delta < 0:
            phase = "pre-event"
        elif delta == 0:
            phase = "day-of"
        else:
            phase = "post-event"

        return (delta, phase)
    except Exception:
        return (None, "unknown")


# -----------------------------------------------------------------------------
# 2. Main Processing Pipeline
# -----------------------------------------------------------------------------

def process_gdelt_news(
    raw_news_path: str = DEFAULT_RAW_NEWS_PATH,
    events_csv_path: str = EVENTS_CSV,
    output_parquet_path: str = DEFAULT_PROCESSED_NEWS_PARQUET,
    output_csv_path: str = DEFAULT_PROCESSED_NEWS_CSV,
    min_financial_relevance: float = 0.0
) -> pd.DataFrame:
    """
    Loads raw GDELT records from jsonl, normalizes schemas, maps events and company tickers,
    calculates timing phases, classifies sentiment, and saves to Parquet and CSV.
    """
    if not os.path.exists(raw_news_path):
        raise FileNotFoundError(f"Raw news file not found at {raw_news_path}. Run ingestion first.")

    # Load events map for date alignment
    events_date_map: Dict[str, str] = {}
    if os.path.exists(events_csv_path):
        try:
            ev_df = pd.read_csv(events_csv_path)
            for _, r in ev_df.iterrows():
                ev_id = r.get("event_id")
                ev_date = r.get("event_date")
                if ev_id and ev_date:
                    events_date_map[str(ev_id)] = str(ev_date)
        except Exception as e:
            logger.warning(f"Could not load event dates from {events_csv_path}: {e}")

    raw_records: List[Dict[str, Any]] = []
    with open(raw_news_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    raw_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    logger.info(f"Loaded {len(raw_records)} raw records from {raw_news_path}")

    processed_rows: List[Dict[str, Any]] = []
    seen_article_ids: Set[str] = set()

    for r in raw_records:
        url = r.get("url", "").strip()
        title = r.get("title", "").strip()
        seendate = r.get("seendate", "")

        pub_date, pub_datetime = parse_gdelt_timestamp(seendate)
        if not pub_date:
            continue

        article_id = generate_article_id(url, title, pub_date)
        if article_id in seen_article_ids:
            continue
        seen_article_ids.add(article_id)

        ev_id = r.get("matched_event_id", "")
        ev_name = r.get("matched_event_name", "")
        ev_cat = r.get("event_category", "General")
        ev_date = events_date_map.get(ev_id, "")

        days_diff, timing_phase = compute_event_timing_phase(pub_date, ev_date)

        domain = r.get("domain", "")
        if not domain and url:
            domain_match = re.search(r"https?://(?:www\.)?([^/]+)", url)
            domain = domain_match.group(1) if domain_match else ""

        orgs = r.get("organizations", "")
        tickers = extract_matched_tickers(title, orgs, existing_tickers=r.get("related_tickers", ""))
        fin_relevance = compute_financial_relevance_score(title, orgs)

        if fin_relevance < min_financial_relevance:
            continue

        tone = float(r.get("tone", 0.0) or 0.0)
        pos = float(r.get("positive_score", 0.0) or 0.0)
        neg = float(r.get("negative_score", 0.0) or 0.0)
        pol = float(r.get("polarity", 0.0) or 0.0)
        sentiment_label = classify_sentiment(tone)

        processed_rows.append({
            "article_id": article_id,
            "published_date": pub_date,
            "published_datetime": pub_datetime,
            "event_date": ev_date,
            "days_from_event": days_diff,
            "timing_phase": timing_phase,
            "title": title,
            "url": url,
            "domain": domain,
            "source_country": r.get("sourcecountry", "Global"),
            "language": r.get("language", "English"),
            "event_id": ev_id,
            "event_name": ev_name,
            "event_category": ev_cat,
            "organizations": orgs,
            "matched_tickers": tickers,
            "related_sectors": r.get("related_sectors", ""),
            "gdelt_tone": tone,
            "positive_score": pos,
            "negative_score": neg,
            "polarity": pol,
            "sentiment_label": sentiment_label,
            "financial_relevance_score": fin_relevance
        })

    df = pd.DataFrame(processed_rows)

    if df.empty:
        logger.warning("Processed news dataset is empty.")
        return df

    # Sort chronologically by published date
    df = df.sort_values(["published_date", "published_datetime"]).reset_index(drop=True)

    # Save to CSV and Parquet
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    df.to_parquet(output_parquet_path, engine="pyarrow", index=False)
    df.to_csv(output_csv_path, index=False)

    logger.info(f"Processed news saved to {output_parquet_path} and {output_csv_path} ({len(df)} articles).")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df = process_gdelt_news()
    print(f"\nProcessed News DataFrame: {len(df)} rows, {len(df.columns)} columns.")
    if not df.empty:
        print(df[["published_date", "timing_phase", "event_category", "sentiment_label", "title"]].head(5))
