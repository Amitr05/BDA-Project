"""
Analytics Package
Provides event study methodology and quantitative analysis modules.
"""

from src.analytics.event_study import (
    load_processed_data,
    calculate_event_metrics,
    run_event_study_analysis
)
from src.analytics.news_impact import (
    analyze_sentiment_vs_volatility,
    analyze_geopolitical_impact_by_industry,
    analyze_sentiment_pre_post_event,
    analyze_top_news_generating_countries,
    analyze_news_volume_vs_market_movements
)
