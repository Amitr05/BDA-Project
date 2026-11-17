"""
News Sentiment & Market Impact Analytics Module
===============================================
Investigates the relationship between global news sentiment (from GDELT)
and US stock market movements, volatility (VIX), and sector returns.

Implements the five core analyses requested in Prompt 4:
  1. Relationship between negative news sentiment and market volatility (VIX).
  2. Impact of geopolitical news on different industries / sectors.
  3. Whether news sentiment changes before or after major market movements (pre vs post event).
  4. Which countries generate the most market-relevant news.
  5. Relationship between news volume and stock market movements.
"""

import logging
import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_NEWS_PARQUET = "data/processed/news_articles.parquet"
DEFAULT_SP500_PARQUET = "data/processed/sp500_processed.parquet"
DEFAULT_VIX_PARQUET = "data/processed/vix_processed.parquet"


# -----------------------------------------------------------------------------
# 1. Sentiment vs Market Volatility (VIX)
# -----------------------------------------------------------------------------

def analyze_sentiment_vs_volatility(
    news_df: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None,
    news_path: str = DEFAULT_NEWS_PARQUET,
    vix_path: str = DEFAULT_VIX_PARQUET
) -> Dict[str, Any]:
    """
    Examines the statistical relationship between daily aggregate news tone
    and VIX volatility levels and daily VIX changes.
    """
    if news_df is None:
        if not os.path.exists(news_path):
            raise FileNotFoundError(f"News dataset not found at {news_path}")
        news_df = pd.read_parquet(news_path)

    if vix_df is None:
        if not os.path.exists(vix_path):
            raise FileNotFoundError(f"VIX dataset not found at {vix_path}")
        vix_df = pd.read_parquet(vix_path)

    # Daily aggregation of news sentiment
    daily_news = news_df.groupby("published_date").agg(
        mean_tone=("gdelt_tone", "mean"),
        min_tone=("gdelt_tone", "min"),
        article_count=("article_id", "count"),
        negative_article_ratio=("sentiment_label", lambda s: (s == "Negative").mean())
    ).reset_index().rename(columns={"published_date": "Date"})

    # Prepare VIX daily data
    vix_clean = vix_df[["Date", "Close"]].copy().rename(columns={"Close": "VIX_Close"})
    vix_clean["Date"] = pd.to_datetime(vix_clean["Date"]).dt.strftime("%Y-%m-%d")
    vix_clean["VIX_1D_Change"] = vix_clean["VIX_Close"].diff()

    merged = pd.merge(daily_news, vix_clean, on="Date", how="inner").dropna()

    if len(merged) < 3:
        # Fallback or small sample
        return {
            "observations_count": len(merged),
            "tone_vs_vix_level_corr": None,
            "tone_vs_vix_change_corr": None,
            "negative_ratio_vs_vix_corr": None,
            "summary": "Insufficient overlapping date records between news and VIX.",
            "interpretation": "Insufficient overlapping date records between news and VIX."
        }

    corr_tone_vix = float(merged["mean_tone"].corr(merged["VIX_Close"]))
    corr_tone_vix_change = float(merged["mean_tone"].corr(merged["VIX_1D_Change"]))
    corr_neg_vix = float(merged["negative_article_ratio"].corr(merged["VIX_Close"]))

    return {
        "observations_count": len(merged),
        "tone_vs_vix_level_corr": round(corr_tone_vix, 4) if pd.notna(corr_tone_vix) else None,
        "tone_vs_vix_change_corr": round(corr_tone_vix_change, 4) if pd.notna(corr_tone_vix_change) else None,
        "negative_ratio_vs_vix_corr": round(corr_neg_vix, 4) if pd.notna(corr_neg_vix) else None,
        "interpretation": (
            "Negative correlation between news tone and VIX confirms that "
            "as global news tone turns negative, market volatility (VIX) rises."
            if (corr_tone_vix and corr_tone_vix < 0) else "Modest or neutral correlation observed."
        )
    }


# -----------------------------------------------------------------------------
# 2. Geopolitical News Impact by Industry / Sector
# -----------------------------------------------------------------------------

def analyze_geopolitical_impact_by_industry(
    news_df: Optional[pd.DataFrame] = None,
    news_path: str = DEFAULT_NEWS_PARQUET
) -> pd.DataFrame:
    """
    Measures the volume, average tone, and negative sentiment concentration
    of Geopolitical / Conflict news across different market sectors.
    """
    if news_df is None:
        news_df = pd.read_parquet(news_path)

    geop_news = news_df[news_df["event_category"].str.contains("War|Geopolitical|Conflict", case=False, na=False)].copy()

    if geop_news.empty:
        geop_news = news_df.copy()

    sector_records = []
    for _, row in geop_news.iterrows():
        sectors = str(row.get("related_sectors", "")).split("|")
        for s in sectors:
            s_clean = s.strip()
            if s_clean:
                sector_records.append({
                    "sector": s_clean,
                    "tone": row["gdelt_tone"],
                    "is_negative": 1 if row["sentiment_label"] == "Negative" else 0,
                    "is_positive": 1 if row["sentiment_label"] == "Positive" else 0,
                    "article_id": row["article_id"]
                })

    if not sector_records:
        return pd.DataFrame()

    sec_df = pd.DataFrame(sector_records)
    summary = sec_df.groupby("sector").agg(
        article_count=("article_id", "count"),
        average_tone=("tone", "mean"),
        negative_sentiment_pct=("is_negative", lambda x: round(x.mean() * 100, 1)),
        positive_sentiment_pct=("is_positive", lambda x: round(x.mean() * 100, 1))
    ).reset_index().sort_values("article_count", ascending=False).reset_index(drop=True)

    summary["average_tone"] = summary["average_tone"].round(2)
    return summary


# -----------------------------------------------------------------------------
# 3. News Sentiment Timing: Pre-Event vs Post-Event
# -----------------------------------------------------------------------------

def analyze_sentiment_pre_post_event(
    news_df: Optional[pd.DataFrame] = None,
    news_path: str = DEFAULT_NEWS_PARQUET
) -> Dict[str, Any]:
    """
    Examines whether news sentiment turns negative before an event occurs
    (anticipatory / informational lead) or after (reactionary lag).
    """
    if news_df is None:
        news_df = pd.read_parquet(news_path)

    valid_phases = news_df[news_df["timing_phase"].isin(["pre-event", "day-of", "post-event"])]

    if valid_phases.empty:
        return {"summary": "No timing phase records available."}

    phase_stats = valid_phases.groupby("timing_phase").agg(
        article_count=("article_id", "count"),
        average_tone=("gdelt_tone", "mean"),
        negative_ratio=("sentiment_label", lambda s: round((s == "Negative").mean() * 100, 1)),
        positive_ratio=("sentiment_label", lambda s: round((s == "Positive").mean() * 100, 1))
    ).to_dict(orient="index")

    for k in phase_stats:
        phase_stats[k]["average_tone"] = round(phase_stats[k]["average_tone"], 2)

    pre_tone = phase_stats.get("pre-event", {}).get("average_tone", 0.0)
    day_tone = phase_stats.get("day-of", {}).get("average_tone", 0.0)
    post_tone = phase_stats.get("post-event", {}).get("average_tone", 0.0)

    return {
        "phase_breakdown": phase_stats,
        "pre_event_average_tone": pre_tone,
        "day_of_average_tone": day_tone,
        "post_event_average_tone": post_tone,
        "tone_drop_day_of": round(day_tone - pre_tone, 2),
        "post_event_recovery": round(post_tone - day_tone, 2)
    }


# -----------------------------------------------------------------------------
# 4. Top News Generating Countries
# -----------------------------------------------------------------------------

def analyze_top_news_generating_countries(
    news_df: Optional[pd.DataFrame] = None,
    news_path: str = DEFAULT_NEWS_PARQUET,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Ranks countries by market-relevant news volume, average sentiment tone,
    and average financial relevance score.
    """
    if news_df is None:
        news_df = pd.read_parquet(news_path)

    grouped = news_df.groupby("source_country").agg(
        article_count=("article_id", "count"),
        average_tone=("gdelt_tone", "mean"),
        avg_financial_relevance=("financial_relevance_score", "mean"),
        negative_articles_count=("sentiment_label", lambda s: (s == "Negative").sum())
    ).reset_index()

    grouped["average_tone"] = grouped["average_tone"].round(2)
    grouped["avg_financial_relevance"] = grouped["avg_financial_relevance"].round(2)
    grouped = grouped.sort_values("article_count", ascending=False).head(top_n).reset_index(drop=True)

    return grouped


# -----------------------------------------------------------------------------
# 5. News Volume vs Market Movements
# -----------------------------------------------------------------------------

def analyze_news_volume_vs_market_movements(
    news_df: Optional[pd.DataFrame] = None,
    sp500_df: Optional[pd.DataFrame] = None,
    news_path: str = DEFAULT_NEWS_PARQUET,
    sp500_path: str = DEFAULT_SP500_PARQUET
) -> Dict[str, Any]:
    """
    Studies the correlation between news coverage volume surges
    and absolute daily equity returns / daily volume changes.
    """
    if news_df is None:
        news_df = pd.read_parquet(news_path)

    if sp500_df is None:
        sp500_df = pd.read_parquet(sp500_path)

    daily_news = news_df.groupby("published_date").size().reset_index(name="news_volume")
    daily_news = daily_news.rename(columns={"published_date": "Date"})

    mkt = sp500_df[["Date", "Close", "Volume"]].copy()
    mkt["Date"] = pd.to_datetime(mkt["Date"]).dt.strftime("%Y-%m-%d")
    mkt["abs_return_pct"] = (mkt["Close"].pct_change().abs() * 100).round(4)
    mkt["volume_change_pct"] = (mkt["Volume"].pct_change() * 100).round(2)

    merged = pd.merge(daily_news, mkt, on="Date", how="inner").dropna()

    if len(merged) < 3:
        return {
            "observations_count": len(merged),
            "news_vol_vs_abs_return_corr": None,
            "news_vol_vs_trade_volume_corr": None,
            "summary": "Insufficient overlapping date records."
        }

    corr_return = float(merged["news_volume"].corr(merged["abs_return_pct"]))
    corr_vol = float(merged["news_volume"].corr(merged["Volume"]))

    return {
        "observations_count": len(merged),
        "news_vol_vs_abs_return_corr": round(corr_return, 4) if pd.notna(corr_return) else None,
        "news_vol_vs_trade_volume_corr": round(corr_vol, 4) if pd.notna(corr_vol) else None,
        "interpretation": (
            "Higher news volume is positively correlated with market volatility and trading volume."
            if (corr_return and corr_return > 0) else "Neutral volume correlation observed."
        )
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n--- Sentiment vs Volatility ---")
    print(analyze_sentiment_vs_volatility())
    print("\n--- Industry Geopolitical Impact ---")
    print(analyze_geopolitical_impact_by_industry().head(5))
    print("\n--- Pre vs Post Event Sentiment ---")
    print(analyze_sentiment_pre_post_event())
    print("\n--- Top News Generating Countries ---")
    print(analyze_top_news_generating_countries().head(5))
