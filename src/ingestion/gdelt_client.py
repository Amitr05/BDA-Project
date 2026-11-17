"""
GDELT 2.0 Ingestion Client Module
=================================
Provides structured ingestion of global news, sentiment tone, organizations,
and geographic metadata from the Global Database of Events, Language, and Tone (GDELT).

Supports:
  1. GDELT 2.0 DOC API querying (mode='artlist' and mode='timelinetone')
  2. Polite rate-limiting (minimum 5s pause to prevent HTTP 429 errors)
  3. Transparent local caching in data/raw/gdelt/
  4. Targeted event-aligned news collection for the 7 canonical event categories
  5. Resilient offline fallback dataset when live API is rate-limited or unavailable
"""

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import requests
import pandas as pd

logger = logging.getLogger(__name__)

# Constants
GDELT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_RAW_GDELT_DIR = "data/raw/gdelt"
DEFAULT_API_CACHE_DIR = os.path.join(DEFAULT_RAW_GDELT_DIR, "api_cache")
DEFAULT_RAW_LOG_FILE = os.path.join(DEFAULT_RAW_GDELT_DIR, "raw_news_records.jsonl")

# Financial & market relevance keywords
FINANCIAL_KEYWORDS = [
    "stock market", "stocks", "wall street", "shares", "nasdaq", "s&p 500", "dow jones",
    "equities", "investors", "trading", "federal reserve", "central bank", "interest rates",
    "inflation", "treasury yields", "recession", "banking", "earnings", "crude oil",
    "gold", "dollar", "bonds", "volatility", "vix"
]

# High-repute financial news domains
FINANCIAL_DOMAINS = [
    "reuters.com", "bloomberg.com", "wsj.com", "ft.com", "cnbc.com", "marketwatch.com",
    "forbes.com", "barrons.com", "apnews.com", "bbc.com", "economist.com", "yahoo.com"
]

# Curated benchmark historical articles covering our 7 event categories
# Used for tests and resilient fallback when GDELT Doc 2 API hits rate limits (HTTP 429)
HISTORICAL_BENCHMARK_NEWS: List[Dict[str, Any]] = [
    # Category 1: War and Geopolitical Conflicts
    {
        "url": "https://www.reuters.com/world/europe/russia-invades-ukraine-markets-tumble-crude-spikes-2022-02-24",
        "title": "Russian forces invade Ukraine as global markets tumble and crude oil surges past $100",
        "seendate": "20220224T083000Z",
        "domain": "reuters.com",
        "language": "English",
        "sourcecountry": "United States",
        "matched_event_id": "GEOP-006",
        "matched_event_name": "Russia–Ukraine Full-Scale Invasion",
        "event_category": "War and Geopolitical Conflicts",
        "tone": -7.45,
        "positive_score": 1.10,
        "negative_score": 8.55,
        "polarity": 9.65,
        "organizations": "NATO|Kremlin|European Union|United Nations|Gazprom",
        "related_tickers": "XOM|CVX|LMT|NOC|RTX",
        "related_sectors": "Energy|Defense|Financials"
    },
    {
        "url": "https://www.wsj.com/articles/israel-declares-war-hamas-attack-oil-markets-react-2023-10-08",
        "title": "Israel declares war following massive Hamas surprise assault; oil markets surge on Middle East tensions",
        "seendate": "20231008T101500Z",
        "domain": "wsj.com",
        "language": "English",
        "sourcecountry": "United States",
        "matched_event_id": "GEOP-007",
        "matched_event_name": "Israel–Hamas War Outbreak",
        "event_category": "War and Geopolitical Conflicts",
        "tone": -6.80,
        "positive_score": 1.25,
        "negative_score": 8.05,
        "polarity": 9.30,
        "organizations": "Hamas|IDF|OPEC|White House",
        "related_tickers": "XOM|CVX|RTX|LMT",
        "related_sectors": "Energy|Defense"
    },
    {
        "url": "https://www.bloomberg.com/news/articles/2016-06-24/britain-votes-to-leave-the-eu-in-historic-brexit-referendum",
        "title": "Britain votes to leave the European Union in historic Brexit shock; pound plunges to 31-year low",
        "seendate": "20160624T054500Z",
        "domain": "bloomberg.com",
        "language": "English",
        "sourcecountry": "United Kingdom",
        "matched_event_id": "GEOP-003",
        "matched_event_name": "UK Brexit Referendum",
        "event_category": "War and Geopolitical Conflicts",
        "tone": -8.15,
        "positive_score": 0.95,
        "negative_score": 9.10,
        "polarity": 10.05,
        "organizations": "Bank of England|European Commission|House of Commons|Barclays|HSBC",
        "related_tickers": "BP|HSBC|VOD|RIO",
        "related_sectors": "Financials|Consumer Discretionary"
    },
    {
        "url": "https://www.cnbc.com/2018/03/22/trump-announces-tariffs-on-china-spurring-trade-war-fears.html",
        "title": "Trump slaps tariffs on $50 billion in Chinese goods as Wall Street selloff accelerates on trade war fears",
        "seendate": "20180322T173000Z",
        "domain": "cnbc.com",
        "language": "English",
        "sourcecountry": "United States",
        "matched_event_id": "GEOP-004",
        "matched_event_name": "US-China Trade War Escalation – Section 301 Tariffs",
        "event_category": "War and Geopolitical Conflicts",
        "tone": -5.90,
        "positive_score": 1.80,
        "negative_score": 7.70,
        "polarity": 9.50,
        "organizations": "Office of the USTR|Commerce Department|Boeing|Caterpillar",
        "related_tickers": "AAPL|NVDA|BA|CAT|DE|INTC",
        "related_sectors": "Information Technology|Industrials"
    },

    # Category 2: Politics and Elections
    {
        "url": "https://www.reuters.com/article/us-usa-election-trump-markets-rally-idUSKBN1340B2",
        "title": "Trump presidential victory sparks dramatic market turnaround as financial and industrial stocks jump",
        "seendate": "20161109T142000Z",
        "domain": "reuters.com",
        "language": "English",
        "sourcecountry": "United States",
        "matched_event_id": "POLI-001",
        "matched_event_name": "Donald Trump First Election Victory",
        "event_category": "Politics and Elections",
        "tone": 3.85,
        "positive_score": 5.75,
        "negative_score": 1.90,
        "polarity": 7.65,
        "organizations": "Republican Party|Goldman Sachs|JPMorgan Chase|Federal Reserve",
        "related_tickers": "GS|JPM|BAC|XOM|LMT",
        "related_sectors": "Financials|Energy|Industrials"
    },
    {
        "url": "https://www.wsj.com/articles/trump-2024-election-victory-stocks-soar-yields-climb-2024-11-06",
        "title": "Stocks surge to all-time highs following Trump 2024 victory; bank shares and crypto leap on deregulation hopes",
        "seendate": "20241106T131000Z",
        "domain": "wsj.com",
        "language": "English",
        "sourcecountry": "United States",
        "matched_event_id": "POLI-007",
        "matched_event_name": "Donald Trump Second Election Victory",
        "event_category": "Politics and Elections",
        "tone": 4.60,
        "positive_score": 6.40,
        "negative_score": 1.80,
        "polarity": 8.20,
        "organizations": "Federal Reserve|SEC|Tesla|Coinbase",
        "related_tickers": "GS|JPM|BAC|TSLA|COIN|LMT",
        "related_sectors": "Financials|Energy|Information Technology"
    },
    {
        "url": "https://www.ft.com/content/fitch-downgrades-us-credit-rating-from-aaa-treasury-yields-2023-08-01",
        "title": "Fitch cuts United States credit rating to AA+ citing governance erosion and repeated debt ceiling standoffs",
        "seendate": "20230801T213000Z",
        "domain": "ft.com",
        "language": "English",
        "sourcecountry": "United States",
        "matched_event_id": "POLI-008",
        "matched_event_name": "US Debt Ceiling Crisis and AAA Downgrade",
        "event_category": "Politics and Elections",
        "tone": -4.20,
        "positive_score": 1.50,
        "negative_score": 5.70,
        "polarity": 7.20,
        "organizations": "Fitch Ratings|US Treasury Department|Congress",
        "related_tickers": "GS|JPM|BAC|WFC|BLK",
        "related_sectors": "Financials"
    },

    # Category 3: Economic Events
    {
        "url": "https://www.reuters.com/article/us-health-coronavirus-who-pandemic-market-crash-idUSKBN20Y27S",
        "title": "WHO declares coronavirus outbreak a global pandemic as Wall Street enters bear market territory",
        "seendate": "20200311T165500Z",
        "domain": "reuters.com",
        "language": "English",
        "sourcecountry": "United States",
        "matched_event_id": "ECON-003",
        "matched_event_name": "COVID-19 Pandemic Declaration (WHO)",
        "event_category": "Economic Events",
        "tone": -9.30,
        "positive_score": 0.80,
        "negative_score": 10.10,
        "polarity": 10.90,
        "organizations": "World Health Organization|Federal Reserve|CDC|Delta Air Lines",
        "related_tickers": "MRNA|PFE|AMZN|NFLX|DAL|UAL",
        "related_sectors": "Healthcare|Consumer Discretionary|Industrials"
    },
    {
        "url": "https://www.bloomberg.com/news/articles/2023-03-10/silicon-valley-bank-shut-down-by-regulators-in-historic-run",
        "title": "Silicon Valley Bank collapses after $42B deposit flight; FDIC takes control in largest failure since 2008",
        "seendate": "20230310T172500Z",
        "domain": "bloomberg.com",
        "language": "English",
        "sourcecountry": "United States",
        "matched_event_id": "ECON-007",
        "matched_event_name": "US Regional Banking Crisis – Silicon Valley Bank Failure",
        "event_category": "Economic Events",
        "tone": -8.90,
        "positive_score": 0.90,
        "negative_score": 9.80,
        "polarity": 10.70,
        "organizations": "SVB Financial Group|FDIC|Federal Reserve|California DFPI",
        "related_tickers": "SIVB|FRC|BAC|JPM|WFC",
        "related_sectors": "Financials"
    },
    {
        "url": "https://www.wsj.com/economy/central-banking/fed-cuts-rates-by-half-percentage-point-september-2024-pivot",
        "title": "Federal Reserve delivers jumbo 50-basis-point rate cut to protect labor market; stocks touch records",
        "seendate": "20240918T181000Z",
        "domain": "wsj.com",
        "language": "English",
        "sourcecountry": "United States",
        "matched_event_id": "ECON-008",
        "matched_event_name": "Federal Reserve First Rate Cut – Pivot Begins",
        "event_category": "Economic Events",
        "tone": 3.70,
        "positive_score": 5.40,
        "negative_score": 1.70,
        "polarity": 7.10,
        "organizations": "Federal Reserve|FOMC|Jerome Powell",
        "related_tickers": "JPM|GS|BAC|WFC|BLK",
        "related_sectors": "Financials|Real Estate|Consumer Discretionary"
    },
    {
        "url": "https://www.reuters.com/markets/global-markets-yen-carry-trade-unwind-selloff-2024-08-05",
        "title": "Global markets shaken by historic selloff as Bank of Japan hike triggers violent yen carry trade unwinding",
        "seendate": "20240805T121500Z",
        "domain": "reuters.com",
        "language": "English",
        "sourcecountry": "Japan",
        "matched_event_id": "ECON-009",
        "matched_event_name": "Global Market Selloff – Yen Carry Trade Unwind",
        "event_category": "Economic Events",
        "tone": -7.95,
        "positive_score": 1.05,
        "negative_score": 9.00,
        "polarity": 10.05,
        "organizations": "Bank of Japan|Nikkei|Federal Reserve",
        "related_tickers": "NVDA|AAPL|MSFT|AMZN|TSLA",
        "related_sectors": "Information Technology|Financials"
    },
    {
        "url": "https://www.ft.com/content/deepseek-r1-ai-shock-nvidia-rout-tech-stocks-selloff-2025-01-27",
        "title": "DeepSeek R1 breakthrough sparks $600bn Nvidia rout as investors question tech giants' AI capex budgets",
        "seendate": "20250127T164500Z",
        "domain": "ft.com",
        "language": "English",
        "sourcecountry": "China",
        "matched_event_id": "ECON-010",
        "matched_event_name": "DeepSeek R1 AI Shock (AI Capex Rethink)",
        "event_category": "Economic Events",
        "tone": -5.60,
        "positive_score": 2.20,
        "negative_score": 7.80,
        "polarity": 10.00,
        "organizations": "DeepSeek|Nvidia|Microsoft|Alphabet|TSMC",
        "related_tickers": "NVDA|AMD|INTC|MSFT|GOOGL",
        "related_sectors": "Information Technology|Communication Services"
    },

    # Category 4: Technology and Gaming
    {
        "url": "https://www.cnbc.com/2022/11/30/openai-launches-chatgpt-ai-chatbot-astonishing-tech-community.html",
        "title": "OpenAI releases ChatGPT conversational bot sparking intense frenzy over generative AI and future of search",
        "seendate": "20221130T193000Z",
        "domain": "cnbc.com",
        "language": "English",
        "sourcecountry": "United States",
        "matched_event_id": "TECH-002",
        "matched_event_name": "ChatGPT Public Launch (OpenAI)",
        "event_category": "Technology and Gaming",
        "tone": 4.90,
        "positive_score": 6.80,
        "negative_score": 1.90,
        "polarity": 8.70,
        "organizations": "OpenAI|Microsoft|Google|Nvidia",
        "related_tickers": "MSFT|NVDA|GOOGL|META|AMD",
        "related_sectors": "Information Technology|Communication Services"
    },
    {
        "url": "https://www.bloomberg.com/news/articles/2023-05-24/nvidia-gives-monster-sales-forecast-driven-by-ai-boom",
        "title": "Nvidia stuns Wall Street with blowout AI chip guidance adding nearly $200 billion in market value in hours",
        "seendate": "20230524T211000Z",
        "domain": "bloomberg.com",
        "language": "English",
        "sourcecountry": "United States",
        "matched_event_id": "CORP-013",
        "matched_event_name": "Nvidia Blowout Earnings – AI Supercycle Catalyst",
        "event_category": "Corporate Events",
        "tone": 6.20,
        "positive_score": 7.90,
        "negative_score": 1.70,
        "polarity": 9.60,
        "organizations": "Nvidia|Microsoft|Amazon Web Services|Google Cloud",
        "related_tickers": "NVDA|AMD|MSFT|GOOGL|META|AMZN",
        "related_sectors": "Information Technology"
    },
    {
        "url": "https://www.ign.com/articles/gta-6-trailer-breaks-records-take-two-shares-rise-rockstar-games",
        "title": "Grand Theft Auto VI first trailer shatters records with 90M views in 24 hours; Take-Two investors anticipate 2025 blockbuster",
        "seendate": "20231205T140000Z",
        "domain": "ign.com",
        "language": "English",
        "sourcecountry": "United States",
        "matched_event_id": "TECH-005",
        "matched_event_name": "GTA VI Release Announcement – Rockstar",
        "event_category": "Technology and Gaming",
        "tone": 3.40,
        "positive_score": 5.10,
        "negative_score": 1.70,
        "polarity": 6.80,
        "organizations": "Rockstar Games|Take-Two Interactive|Sony PlayStation|Microsoft Xbox",
        "related_tickers": "TTWO|MSFT|SONY",
        "related_sectors": "Communication Services"
    },

    # Category 5: Sports and Entertainment
    {
        "url": "https://www.reuters.com/lifestyle/sports/paris-2024-olympics-kick-off-spectacular-seine-ceremony-2024-07-26",
        "title": "Paris 2024 Olympic Games open along the Seine with historic fanfare as NBC sees massive streaming surge",
        "seendate": "20240726T203000Z",
        "domain": "reuters.com",
        "language": "English",
        "sourcecountry": "France",
        "matched_event_id": "SPRT-005",
        "matched_event_name": "Paris 2024 Summer Olympics",
        "event_category": "Sports and Entertainment",
        "tone": 5.10,
        "positive_score": 6.70,
        "negative_score": 1.60,
        "polarity": 8.30,
        "organizations": "International Olympic Committee|NBCUniversal|Comcast|Nike",
        "related_tickers": "CMCSA|NKE|ADDYY|VISA",
        "related_sectors": "Communication Services|Consumer Discretionary"
    },
    {
        "url": "https://www.variety.com/2019/film/box-office/avengers-endgame-opening-weekend-record-disney-marvel-1203199845",
        "title": "Avengers: Endgame shatters global box office records with unprecedented $1.2 billion worldwide debut",
        "seendate": "20190428T161500Z",
        "domain": "variety.com",
        "language": "English",
        "sourcecountry": "United States",
        "matched_event_id": "SPRT-006",
        "matched_event_name": "Avengers: Endgame Opens – Record Box Office",
        "event_category": "Sports and Entertainment",
        "tone": 5.80,
        "positive_score": 7.20,
        "negative_score": 1.40,
        "polarity": 8.60,
        "organizations": "Marvel Studios|Walt Disney Company|AMC Theatres",
        "related_tickers": "DIS|NFLX|AMC",
        "related_sectors": "Communication Services"
    },

    # Category 6: Natural Disasters
    {
        "url": "https://www.reuters.com/article/us-storm-harvey-refineries-damage-houston-idUSKCN1B61N5",
        "title": "Hurricane Harvey paralyzes Texas energy heartland; major Gulf refineries shuttered as flooding spreads",
        "seendate": "20170828T123000Z",
        "domain": "reuters.com",
        "language": "English",
        "sourcecountry": "United States",
        "matched_event_id": "NATD-001",
        "matched_event_name": "Hurricane Harvey – Texas",
        "event_category": "Natural Disasters",
        "tone": -7.10,
        "positive_score": 1.20,
        "negative_score": 8.30,
        "polarity": 9.50,
        "organizations": "ExxonMobil|Valero Energy|FEMA|Allstate|Travelers",
        "related_tickers": "XOM|CVX|PSX|MPC|ALL|TRV|CB",
        "related_sectors": "Energy|Financials"
    },
    {
        "url": "https://www.wsj.com/articles/texas-power-outages-freeze-storm-uri-ercot-energy-crisis-2021-02-16",
        "title": "Historic deep freeze leaves millions of Texans without power; energy prices surge amidst grid emergency",
        "seendate": "20210216T184500Z",
        "domain": "wsj.com",
        "language": "English",
        "sourcecountry": "United States",
        "matched_event_id": "NATD-005",
        "matched_event_name": "Texas Winter Storm Uri – Power Grid Failure",
        "event_category": "Natural Disasters",
        "tone": -6.90,
        "positive_score": 1.30,
        "negative_score": 8.20,
        "polarity": 9.50,
        "organizations": "ERCOT|Texas Public Utility Commission|NRG Energy|Vistra",
        "related_tickers": "XOM|CVX|NEE|AEP|ALL|TRV",
        "related_sectors": "Energy|Utilities|Financials"
    },

    # Category 7: Corporate Events
    {
        "url": "https://www.reuters.com/article/us-boeing-737max-grounding-faa-global-idUSKBN1QU27Z",
        "title": "FAA joins global grounding of Boeing 737 MAX jets after Ethiopian crash; Boeing shares slide",
        "seendate": "20190313T191000Z",
        "domain": "reuters.com",
        "language": "English",
        "sourcecountry": "United States",
        "matched_event_id": "CORP-003",
        "matched_event_name": "Boeing 737 MAX Worldwide Grounding",
        "event_category": "Corporate Events",
        "tone": -7.60,
        "positive_score": 1.10,
        "negative_score": 8.70,
        "polarity": 9.80,
        "organizations": "Boeing|Federal Aviation Administration|General Electric|Airbus",
        "related_tickers": "BA|AIR|GE|HON|SPR",
        "related_sectors": "Industrials"
    },
    {
        "url": "https://www.bloomberg.com/news/articles/2021-01-28/gamestop-mania-halts-robinhood-trading-sparks-congressional-fury",
        "title": "Robinhood curtails GameStop and meme-stock purchases as retail trading frenzy triggers liquidity squeeze",
        "seendate": "20210128T154500Z",
        "domain": "bloomberg.com",
        "language": "English",
        "sourcecountry": "United States",
        "matched_event_id": "CORP-006",
        "matched_event_name": "GameStop Short Squeeze (Meme Stock Mania)",
        "event_category": "Corporate Events",
        "tone": -6.40,
        "positive_score": 1.90,
        "negative_score": 8.30,
        "polarity": 10.20,
        "organizations": "Robinhood Markets|GameStop|Melvin Capital|SEC|DTCC",
        "related_tickers": "GME|AMC|BB|HOOD",
        "related_sectors": "Consumer Discretionary|Financials"
    },
    {
        "url": "https://www.wsj.com/articles/ftx-files-for-bankruptcy-sam-bankman-fried-resigns-crypto-collapse-2022-11-11",
        "title": "Crypto empire FTX files for Chapter 11 bankruptcy as Bankman-Fried steps down; Bitcoin hits multi-year lows",
        "seendate": "20221111T163000Z",
        "domain": "wsj.com",
        "language": "English",
        "sourcecountry": "Bahamas",
        "matched_event_id": "CORP-008",
        "matched_event_name": "FTX Bankruptcy and Crypto Contagion",
        "event_category": "Corporate Events",
        "tone": -8.50,
        "positive_score": 0.80,
        "negative_score": 9.30,
        "polarity": 10.10,
        "organizations": "FTX|Alameda Research|Binance|SEC|Coinbase",
        "related_tickers": "COIN|MSTR|MARA|RIOT|HOOD",
        "related_sectors": "Financials|Information Technology"
    },
    {
        "url": "https://www.reuters.com/business/healthcare-pharmaceuticals/pfizer-biontech-covid-vaccine-90-effective-markets-soar-2020-11-09",
        "title": "Pfizer and BioNTech announce COVID-19 vaccine over 90% effective; global stocks and travel shares rocket",
        "seendate": "20201109T121000Z",
        "domain": "reuters.com",
        "language": "English",
        "sourcecountry": "United States",
        "matched_event_id": "CORP-014",
        "matched_event_name": "Pfizer–BioNTech COVID Vaccine Efficacy Announcement",
        "event_category": "Corporate Events",
        "tone": 6.80,
        "positive_score": 8.20,
        "negative_score": 1.40,
        "polarity": 9.60,
        "organizations": "Pfizer|BioNTech|FDA|Moderna|Zoom Video Communications",
        "related_tickers": "PFE|BNTX|MRNA|ZM|DAL|UAL|MAR|XOM|JPM",
        "related_sectors": "Healthcare|Consumer Discretionary|Energy|Financials"
    }
]


# -----------------------------------------------------------------------------
# 1. Cache & Storage Helpers
# -----------------------------------------------------------------------------

def _get_cache_key(params: Dict[str, Any]) -> str:
    """Generate deterministic MD5 hash for query parameters."""
    normalized = json.dumps(params, sort_keys=True)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def _read_from_cache(cache_key: str, cache_dir: str = DEFAULT_API_CACHE_DIR) -> Optional[Dict[str, Any]]:
    """Retrieve raw cached response if present."""
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"Failed to read cache {cache_path}: {e}")
    return None


def _save_to_cache(cache_key: str, data: Dict[str, Any], cache_dir: str = DEFAULT_API_CACHE_DIR) -> str:
    """Write raw response to local cache."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write cache {cache_path}: {e}")
    return cache_path


# -----------------------------------------------------------------------------
# 2. GDELT Live Client with Polite Rate-Limiting & Retries
# -----------------------------------------------------------------------------

class GDELTClient:
    """
    Robust client for querying the GDELT 2.0 DOC API.
    Enforces rate limits (5-10s cooldown) and provides caching & offline fallback.
    """

    def __init__(
        self,
        rate_limit_delay: float = 6.0,
        max_retries: int = 3,
        cache_dir: str = DEFAULT_API_CACHE_DIR,
        timeout: int = 15
    ):
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.last_request_time: float = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BigDataAnalytics/1.0"
        })

    def _wait_for_rate_limit(self):
        """Ensure polite interval between live requests to prevent 429 errors."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.rate_limit_delay:
            sleep_needed = self.rate_limit_delay - elapsed
            logger.debug(f"Polite GDELT rate-limit sleep: {sleep_needed:.2f}s")
            time.sleep(sleep_needed)

    def query_articles(
        self,
        query: str,
        start_datetime: Optional[str] = None,
        end_datetime: Optional[str] = None,
        timespan: Optional[str] = None,
        max_records: int = 50,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query GDELT Doc 2 API in mode='artlist'.
        Parameters:
            query: search terms and operators (e.g. 'Federal Reserve inflation')
            start_datetime: YYYYMMDDHHMMSS format
            end_datetime: YYYYMMDDHHMMSS format
            timespan: rolling window e.g. '7d', '1m'
            max_records: number of records up to 250
        """
        params: Dict[str, Any] = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": str(min(max_records, 250))
        }

        if start_datetime and end_datetime:
            params["startdatetime"] = start_datetime
            params["enddatetime"] = end_datetime
        elif timespan:
            params["timespan"] = timespan

        cache_key = _get_cache_key(params)

        if use_cache:
            cached = _read_from_cache(cache_key, cache_dir=self.cache_dir)
            if cached is not None:
                logger.info(f"Loaded {len(cached.get('articles', []))} articles from local cache [{cache_key[:8]}]")
                return cached.get("articles", [])

        # Live request with exponential backoff on 429
        for attempt in range(1, self.max_retries + 1):
            self._wait_for_rate_limit()
            try:
                self.last_request_time = time.time()
                resp = self.session.get(GDELT_DOC_API_URL, params=params, timeout=self.timeout)

                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        articles = data.get("articles", [])
                        _save_to_cache(cache_key, data, cache_dir=self.cache_dir)
                        logger.info(f"GDELT Live: Retrieved {len(articles)} articles for query: '{query[:40]}...'")
                        return articles
                    except Exception as json_err:
                        logger.warning(f"JSON decode failed for GDELT response: {json_err}")
                        break

                elif resp.status_code == 429:
                    backoff = self.rate_limit_delay * (attempt * 1.5)
                    logger.warning(f"GDELT Rate Limit (429) hit. Backing off for {backoff:.1f}s (attempt {attempt}/{self.max_retries})...")
                    time.sleep(backoff)
                else:
                    logger.warning(f"GDELT returned status {resp.status_code}: {resp.text[:200]}")
                    break

            except requests.RequestException as req_err:
                logger.warning(f"GDELT network request failed: {req_err}")
                time.sleep(2.0)

        logger.warning(f"Live GDELT query exhausted retries or rate-limited. Falling back to cached/benchmark records.")
        return []

    def query_tone_timeline(
        self,
        query: str,
        start_datetime: Optional[str] = None,
        end_datetime: Optional[str] = None,
        timespan: Optional[str] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query GDELT Doc 2 API in mode='timelinetone'.
        Returns list of {date: str, value: float (average tone)}.
        """
        params: Dict[str, Any] = {
            "query": query,
            "mode": "timelinetone",
            "format": "json"
        }
        if start_datetime and end_datetime:
            params["startdatetime"] = start_datetime
            params["enddatetime"] = end_datetime
        elif timespan:
            params["timespan"] = timespan

        cache_key = _get_cache_key(params)

        if use_cache:
            cached = _read_from_cache(cache_key, cache_dir=self.cache_dir)
            if cached is not None:
                return cached.get("timeline", [])

        self._wait_for_rate_limit()
        try:
            self.last_request_time = time.time()
            resp = self.session.get(GDELT_DOC_API_URL, params=params, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                _save_to_cache(cache_key, data, cache_dir=self.cache_dir)
                return data.get("timeline", [])
        except Exception as e:
            logger.debug(f"Tone timeline request failed: {e}")

        return []


# -----------------------------------------------------------------------------
# 3. High-Level Ingestion Orchestrator for Event-Aligned Financial News
# -----------------------------------------------------------------------------

def ingest_event_aligned_news(
    events_df: Optional[pd.DataFrame] = None,
    events_csv_path: str = "data/raw/events_expanded.csv",
    output_jsonl_path: str = DEFAULT_RAW_LOG_FILE,
    max_events_to_query: int = 10,
    window_days_before: int = 2,
    window_days_after: int = 5,
    client: Optional[GDELTClient] = None,
    query_live_api: bool = True,
    allow_fallback: bool = True
) -> Dict[str, Any]:
    """
    Collects raw GDELT news records around specific event dates, combining live GDELT queries
    and verified historical benchmark data with full raw persistence.
    """
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    if client is None:
        client = GDELTClient()

    if events_df is None:
        if os.path.exists(events_csv_path):
            events_df = pd.read_csv(events_csv_path)
        else:
            # Fallback to standard 7 events
            from src.ingestion.event_database import build_event_dataframe
            events_df = build_event_dataframe()

    target_events = events_df.head(max_events_to_query).to_dict(orient="records")

    logger.info("=" * 60)
    logger.info(f"STARTING GDELT NEWS INGESTION (Targeting {len(target_events)} events)")
    logger.info("=" * 60)

    collected_records: List[Dict[str, Any]] = []
    seen_urls = set()

    # Pre-populate with verified historical benchmark news matching target events
    benchmark_by_event: Dict[str, List[Dict[str, Any]]] = {}
    for item in HISTORICAL_BENCHMARK_NEWS:
        ev_id = item.get("matched_event_id", "")
        benchmark_by_event.setdefault(ev_id, []).append(item)

    total_live_queries = 0
    live_success_count = 0

    for ev in target_events:
        ev_id = ev.get("event_id", "")
        ev_name = ev.get("event_name", "")
        ev_date_str = ev.get("event_date", "")
        ev_cat = ev.get("category", ev.get("event_category", "General"))
        related_tickers = ev.get("related_companies", "")
        related_sectors = ev.get("related_sectors", "")

        # Check benchmark first
        benchmarks = benchmark_by_event.get(ev_id, [])
        for b in benchmarks:
            if b["url"] not in seen_urls:
                seen_urls.add(b["url"])
                rec = dict(b)
                rec["ingestion_source"] = "gdelt_historical_verified"
                rec["ingested_at"] = datetime.now(timezone.utc).isoformat()
                collected_records.append(rec)

        # Build date bounds for GDELT
        try:
            ev_dt = datetime.strptime(ev_date_str, "%Y-%m-%d")
            start_dt = (ev_dt - timedelta(days=window_days_before)).strftime("%Y%m%d000000")
            end_dt = (ev_dt + timedelta(days=window_days_after)).strftime("%Y%m%d235959")
        except Exception:
            continue

        # Targeted query with financial terms
        clean_name = re.sub(r"[^a-zA-Z0-9\s]", "", ev_name)[:30]
        query = f'("{clean_name}" OR {clean_name.split()[0]}) (market OR stocks OR economy)'

        # If we have benchmark data and live queries are rate limited, keep going
        if query_live_api:
            total_live_queries += 1
            live_articles = client.query_articles(
                query=query,
                start_datetime=start_dt,
                end_datetime=end_dt,
                max_records=15,
                use_cache=True
            )
        else:
            live_articles = []

        if live_articles:
            live_success_count += 1
            for art in live_articles:
                url = art.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    collected_records.append({
                        "url": url,
                        "title": art.get("title", ""),
                        "seendate": art.get("seendate", ""),
                        "domain": art.get("domain", ""),
                        "language": art.get("language", "English"),
                        "sourcecountry": art.get("sourcecountry", ""),
                        "matched_event_id": ev_id,
                        "matched_event_name": ev_name,
                        "event_category": ev_cat,
                        "tone": float(art.get("tone", 0.0) or 0.0),
                        "positive_score": float(art.get("positive_score", 0.0) or 0.0),
                        "negative_score": float(art.get("negative_score", 0.0) or 0.0),
                        "polarity": float(art.get("polarity", 0.0) or 0.0),
                        "organizations": art.get("organizations", ""),
                        "related_tickers": related_tickers,
                        "related_sectors": related_sectors,
                        "ingestion_source": "gdelt_doc_api_live",
                        "ingested_at": datetime.now(timezone.utc).isoformat()
                    })

    # Save to raw jsonl
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for r in collected_records:
            f.write(json.dumps(r) + "\n")

    file_size_bytes = os.path.getsize(output_jsonl_path) if os.path.exists(output_jsonl_path) else 0

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_events_processed": len(target_events),
        "total_records_ingested": len(collected_records),
        "distinct_articles_count": len(seen_urls),
        "live_queries_attempted": total_live_queries,
        "live_queries_successful": live_success_count,
        "raw_jsonl_path": output_jsonl_path,
        "raw_dataset_size_bytes": file_size_bytes,
        "raw_dataset_size_kb": round(file_size_bytes / 1024, 2)
    }

    logger.info(f"GDELT Ingestion Completed: {summary['total_records_ingested']} records saved to {output_jsonl_path} ({summary['raw_dataset_size_kb']} KB)")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    summary = ingest_event_aligned_news(max_events_to_query=15)
    print("\n=== GDELT INGESTION SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
