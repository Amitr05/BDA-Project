"""
Analytics Package
Provides event study methodology, news impact analysis, sector ETF analytics,
market-model abnormal returns, statistical significance testing, and benchmarking.
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
from src.analytics.sector_analysis import (
    SECTOR_NAME_MAP,
    load_processed_etf_frames,
    calculate_sector_performance_summary,
    compare_multiple_sector_etfs,
    compute_sector_correlation_matrix,
    rank_sector_performance,
    align_sector_etfs_with_market_and_vix,
    run_sector_event_study
)
from src.analytics.advanced_event_study import (
    find_effective_trading_day_index,
    estimate_market_model_parameters,
    calculate_abnormal_returns,
    compute_statistical_tests,
    run_company_event_studies,
    analyze_cross_company_dispersion,
    compare_event_categories,
    analyze_macroeconomic_context,
    run_master_analytics_pipeline
)
from src.analytics.benchmark_validation import (
    independent_daily_return_and_volatility,
    independent_ols_market_model,
    run_benchmark_and_verification
)
