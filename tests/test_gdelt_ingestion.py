"""
Test Suite – GDELT Global News & Sentiment Integration (Prompt 4)
=================================================================
Covers:
  1. GDELT client query formatting, cache key generation, and cache retrieval
  2. Rate limit handling and polite request timing
  3. Raw news record persistence and schema validation
  4. Processed news dataset schema integrity (Parquet and CSV)
  5. Entity extraction: organization names and company ticker alignment
  6. Financial relevance scoring logic
  7. Distinction between publication timestamp and event occurrence date
  8. Timing phase classification ('pre-event', 'day-of', 'post-event')
  9. Duplicate article detection and elimination
 10. News impact analytics:
     - Sentiment vs VIX volatility correlation
     - Sector & industry geopolitical exposure
     - Pre-event vs Post-event news tone dynamics
     - Top news generating countries
     - News volume vs market movement correlation
"""

import json
import os
import tempfile
import pandas as pd
import pytest

from src.ingestion.gdelt_client import (
    GDELTClient,
    _get_cache_key,
    _save_to_cache,
    _read_from_cache,
    ingest_event_aligned_news,
    HISTORICAL_BENCHMARK_NEWS
)
from src.preprocessing.news_processor import (
    parse_gdelt_timestamp,
    generate_article_id,
    extract_matched_tickers,
    compute_financial_relevance_score,
    classify_sentiment,
    compute_event_timing_phase,
    process_gdelt_news
)
from src.analytics.news_impact import (
    analyze_sentiment_vs_volatility,
    analyze_geopolitical_impact_by_industry,
    analyze_sentiment_pre_post_event,
    analyze_top_news_generating_countries,
    analyze_news_volume_vs_market_movements
)


# -----------------------------------------------------------------------------
# 1. GDELT Client & Caching Tests
# -----------------------------------------------------------------------------

class TestGDELTClientAndCache:
    def test_cache_key_deterministic(self):
        p1 = {"query": "stock market", "mode": "artlist", "format": "json"}
        p2 = {"format": "json", "mode": "artlist", "query": "stock market"}
        assert _get_cache_key(p1) == _get_cache_key(p2)

    def test_cache_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            key = "test_key_12345"
            payload = {"status": "ok", "articles": [{"title": "Test Title", "url": "https://test.com/1"}]}
            saved_path = _save_to_cache(key, payload, cache_dir=tmpdir)
            assert os.path.exists(saved_path)

            loaded = _read_from_cache(key, cache_dir=tmpdir)
            assert loaded is not None
            assert loaded["articles"][0]["title"] == "Test Title"

    def test_client_init_and_cache_usage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = GDELTClient(cache_dir=tmpdir, rate_limit_delay=0.1)
            assert client.rate_limit_delay == 0.1

            # Mock a cached response
            params = {
                "query": "test query",
                "mode": "artlist",
                "format": "json",
                "maxrecords": "50"
            }
            cache_key = _get_cache_key(params)
            _save_to_cache(cache_key, {"articles": [{"title": "Cached Article", "url": "https://cache.com/1"}]}, cache_dir=tmpdir)

            articles = client.query_articles(query="test query", use_cache=True)
            assert len(articles) == 1
            assert articles[0]["title"] == "Cached Article"


# -----------------------------------------------------------------------------
# 2. Raw Ingestion & Output Isolation Tests
# -----------------------------------------------------------------------------

class TestRawIngestionPipeline:
    def test_ingest_event_aligned_news_offline_benchmark(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_jsonl = os.path.join(tmpdir, "raw_news.jsonl")
            cache_dir = os.path.join(tmpdir, "api_cache")
            client = GDELTClient(cache_dir=cache_dir, rate_limit_delay=0.01)

            # Create mock events
            events_df = pd.DataFrame([
                {"event_id": "GEOP-006", "event_name": "Russia Ukraine Invasion", "event_date": "2022-02-24", "category": "War and Geopolitical Conflicts", "related_companies": "XOM|CVX", "related_sectors": "Energy|Defense"},
                {"event_id": "ECON-007", "event_name": "SVB Collapse", "event_date": "2023-03-10", "category": "Economic Events", "related_companies": "SIVB|JPM", "related_sectors": "Financials"}
            ])

            summary = ingest_event_aligned_news(
                events_df=events_df,
                output_jsonl_path=raw_jsonl,
                max_events_to_query=2,
                client=client,
                query_live_api=False
            )

            assert os.path.exists(raw_jsonl)
            assert summary["total_records_ingested"] >= 2
            assert summary["raw_dataset_size_bytes"] > 0
            assert summary["raw_dataset_size_kb"] > 0


# -----------------------------------------------------------------------------
# 3. Processing, Normalization & Entity Extraction Tests
# -----------------------------------------------------------------------------

class TestNewsProcessingAndEntities:
    def test_parse_gdelt_timestamp_formats(self):
        d1, iso1 = parse_gdelt_timestamp("20231008T101500Z")
        assert d1 == "2023-10-08"
        assert iso1 == "2023-10-08T10:15:00Z"

        d2, iso2 = parse_gdelt_timestamp("20240805123000")
        assert d2 == "2024-08-05"
        assert iso2 == "2024-08-05T12:30:00Z"

    def test_article_id_generation(self):
        id1 = generate_article_id("https://reuters.com/article1", "Stock Market Up", "2024-01-01")
        id2 = generate_article_id("https://reuters.com/article1", "Stock Market Up", "2024-01-01")
        id3 = generate_article_id("https://reuters.com/article2", "Stock Market Down", "2024-01-01")
        assert id1 == id2
        assert id1 != id3

    def test_extract_matched_tickers(self):
        title = "Nvidia and Microsoft report strong demand while Tesla faces delivery cuts"
        tickers = extract_matched_tickers(title, text_content="", existing_tickers="AAPL")
        ticker_list = tickers.split("|")
        assert "AAPL" in ticker_list
        assert "NVDA" in ticker_list
        assert "MSFT" in ticker_list
        assert "TSLA" in ticker_list

    def test_financial_relevance_scoring(self):
        high_rel = compute_financial_relevance_score(
            title="Federal Reserve signals interest rate cuts as Wall Street stocks rally",
            organizations="Federal Reserve|Treasury"
        )
        assert high_rel >= 0.5

        low_rel = compute_financial_relevance_score(
            title="Local football team wins championship match in overtime",
            organizations="Youth League"
        )
        assert low_rel < 0.3

    def test_classify_sentiment(self):
        assert classify_sentiment(5.5) == "Positive"
        assert classify_sentiment(-6.2) == "Negative"
        assert classify_sentiment(0.2) == "Neutral"

    def test_compute_event_timing_phase(self):
        # Pre-event: published 2 days before event
        delta_pre, phase_pre = compute_event_timing_phase("2023-10-05", "2023-10-07")
        assert delta_pre == -2
        assert phase_pre == "pre-event"

        # Day-of
        delta_day, phase_day = compute_event_timing_phase("2023-10-07", "2023-10-07")
        assert delta_day == 0
        assert phase_day == "day-of"

        # Post-event
        delta_post, phase_post = compute_event_timing_phase("2023-10-10", "2023-10-07")
        assert delta_post == 3
        assert phase_post == "post-event"


# -----------------------------------------------------------------------------
# 4. End-to-End Processing & Deduplication
# -----------------------------------------------------------------------------

class TestProcessingPipelineEndToEnd:
    def test_process_gdelt_news_dual_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = os.path.join(tmpdir, "raw.jsonl")
            out_parquet = os.path.join(tmpdir, "news.parquet")
            out_csv = os.path.join(tmpdir, "news.csv")

            # Write duplicate items into raw
            sample_art = {
                "url": "https://reuters.com/market-story-1",
                "title": "Fed cuts rates; markets surge",
                "seendate": "20240918T181000Z",
                "domain": "reuters.com",
                "matched_event_id": "ECON-008",
                "matched_event_name": "Federal Reserve First Rate Cut",
                "event_category": "Economic Events",
                "tone": 4.5,
                "organizations": "Federal Reserve",
                "related_tickers": "JPM|GS"
            }
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(sample_art) + "\n")
                f.write(json.dumps(sample_art) + "\n")  # duplicate

            df = process_gdelt_news(
                raw_news_path=raw_path,
                output_parquet_path=out_parquet,
                output_csv_path=out_csv
            )

            # Should deduplicate down to 1
            assert len(df) == 1
            assert os.path.exists(out_parquet)
            assert os.path.exists(out_csv)

            # Check schema
            required_cols = [
                "article_id", "published_date", "published_datetime",
                "event_date", "days_from_event", "timing_phase",
                "title", "url", "domain", "source_country",
                "event_id", "event_name", "event_category",
                "organizations", "matched_tickers", "gdelt_tone",
                "sentiment_label", "financial_relevance_score"
            ]
            for col in required_cols:
                assert col in df.columns


# -----------------------------------------------------------------------------
# 5. Analytics Module Tests
# -----------------------------------------------------------------------------

class TestNewsImpactAnalytics:
    @pytest.fixture
    def mock_news_df(self):
        return pd.DataFrame([
            {
                "article_id": "art_1",
                "published_date": "2023-10-07",
                "timing_phase": "day-of",
                "event_category": "War and Geopolitical Conflicts",
                "related_sectors": "Energy|Defense",
                "source_country": "United States",
                "gdelt_tone": -8.0,
                "sentiment_label": "Negative",
                "financial_relevance_score": 0.8
            },
            {
                "article_id": "art_2",
                "published_date": "2023-10-08",
                "timing_phase": "post-event",
                "event_category": "War and Geopolitical Conflicts",
                "related_sectors": "Energy|Financials",
                "source_country": "United States",
                "gdelt_tone": -6.0,
                "sentiment_label": "Negative",
                "financial_relevance_score": 0.7
            },
            {
                "article_id": "art_3",
                "published_date": "2023-10-05",
                "timing_phase": "pre-event",
                "event_category": "Economic Events",
                "related_sectors": "Information Technology",
                "source_country": "United Kingdom",
                "gdelt_tone": 2.0,
                "sentiment_label": "Positive",
                "financial_relevance_score": 0.6
            }
        ])

    @pytest.fixture
    def mock_vix_df(self):
        return pd.DataFrame([
            {"Date": "2023-10-04", "Close": 14.5},
            {"Date": "2023-10-05", "Close": 15.0},
            {"Date": "2023-10-06", "Close": 16.0},
            {"Date": "2023-10-07", "Close": 18.5},
            {"Date": "2023-10-08", "Close": 19.8}
        ])

    @pytest.fixture
    def mock_sp500_df(self):
        return pd.DataFrame([
            {"Date": "2023-10-05", "Close": 4300.0, "Volume": 3000000000},
            {"Date": "2023-10-06", "Close": 4320.0, "Volume": 3200000000},
            {"Date": "2023-10-07", "Close": 4250.0, "Volume": 3800000000},
            {"Date": "2023-10-08", "Close": 4230.0, "Volume": 4100000000}
        ])

    def test_analyze_sentiment_vs_volatility(self, mock_news_df, mock_vix_df):
        res = analyze_sentiment_vs_volatility(news_df=mock_news_df, vix_df=mock_vix_df)
        assert "observations_count" in res
        assert "tone_vs_vix_level_corr" in res
        assert "interpretation" in res

    def test_analyze_geopolitical_impact_by_industry(self, mock_news_df):
        res_df = analyze_geopolitical_impact_by_industry(news_df=mock_news_df)
        assert isinstance(res_df, pd.DataFrame)
        assert not res_df.empty
        assert "sector" in res_df.columns
        assert "article_count" in res_df.columns
        assert "average_tone" in res_df.columns

    def test_analyze_sentiment_pre_post_event(self, mock_news_df):
        res = analyze_sentiment_pre_post_event(news_df=mock_news_df)
        assert "phase_breakdown" in res
        assert "pre_event_average_tone" in res
        assert "day_of_average_tone" in res
        assert "post_event_average_tone" in res

    def test_analyze_top_news_generating_countries(self, mock_news_df):
        res_df = analyze_top_news_generating_countries(news_df=mock_news_df)
        assert isinstance(res_df, pd.DataFrame)
        assert not res_df.empty
        assert "source_country" in res_df.columns
        assert res_df.iloc[0]["source_country"] == "United States"

    def test_analyze_news_volume_vs_market_movements(self, mock_news_df, mock_sp500_df):
        res = analyze_news_volume_vs_market_movements(news_df=mock_news_df, sp500_df=mock_sp500_df)
        assert "observations_count" in res
        assert "news_vol_vs_abs_return_corr" in res
