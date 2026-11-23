# Impact of Geopolitical Conflicts and Major Global Events on the US Stock Market: A Big Data Analytics Approach

## 1. Project Overview
This project investigates the empirical impact of major geopolitical conflicts and macroeconomic global events on the United States financial markets. It implements an end-to-end Big Data Analytics architecture processing seven heterogeneous datasets across **three logical storage layers (Raw, Processed, Analytics)** using **Apache Spark (PySpark SQL)**, **Pandas/PyArrow**, and an interactive **Streamlit & Plotly** analytical platform.

### Integrated Datasets:
1. **Four US Benchmark Equity Indices**: S&P 500 (`^GSPC`), NASDAQ Composite (`^IXIC`), Dow Jones Industrial Average (`^DJI`), Russell 2000 (`^RUT`).
2. **CBOE Volatility Index (`^VIX`)**: Market sentiment and volatility expectations.
3. **Individual Company Equities**: 300+ constituent stocks spanning all 11 GICS sectors.
4. **Eight Select Sector SPDR ETFs**: Technology (`XLK`), Financials (`XLF`), Energy (`XLE`), Healthcare (`XLV`), Industrials (`XLI`), Consumer Discretionary (`XLY`), Consumer Staples (`XLP`), Utilities (`XLU`).
5. **Curated & Historical Event Databases**: Curated calendar (2015–2026) and 50+ event comprehensive database with severity scoring and tags.
6. **GDELT Global News Records**: Global news coverage, article volumes, and sentiment polarity scores.
7. **FRED Point-in-Time Macroeconomic Indicators**: CPI inflation (headline & core), Effective Federal Funds rate, Treasury yields (10Y, 2Y, 3M, 10Y-2Y spread), Unemployment rate, Real GDP growth, and UMich Consumer Sentiment.

---

## 2. Three-Layer Storage Architecture

The platform enforces clean separation of concerns across three logical storage layers:

```
data/
├── raw/                                 # LAYER 1: Immutable raw data ingested directly from sources
│   ├── sp500_raw.csv                    # Historical S&P 500 daily OHLCV
│   ├── nasdaq_raw.csv                   # Historical NASDAQ daily OHLCV
│   ├── dow_jones_raw.csv                # Historical Dow Jones daily OHLCV
│   ├── russell2000_raw.csv              # Historical Russell 2000 daily OHLCV
│   ├── vix_raw.csv                      # Historical CBOE VIX daily OHLCV
│   ├── events.csv                       # Curated global event records
│   ├── events_expanded.csv              # 50+ comprehensive global event database
│   ├── companies.csv                    # 500 S&P constituent companies metadata
│   ├── index_constituents.csv           # Historical index membership (survivorship bias prevention)
│   ├── stocks/                          # 298 individual equity raw CSV files
│   ├── etfs/                            # 8 Sector ETF raw CSV files
│   ├── fred/                            # 12 FRED macroeconomic indicator raw CSV files
│   └── gdelt/                           # Raw GDELT JSON Lines news records
│
├── processed/                           # LAYER 2: Cleaned, schema-standardised, PySpark-transformed data
│   ├── sp500_processed.parquet          # Processed S&P 500 with daily returns & rolling 21d vol
│   ├── nasdaq_processed.parquet         # Processed NASDAQ Composite
│   ├── dow_jones_processed.parquet      # Processed Dow Jones Industrial Average
│   ├── russell2000_processed.parquet    # Processed Russell 2000
│   ├── vix_processed.parquet            # Processed VIX with volatility regime tags
│   ├── event_study_results.parquet      # Multi-index event study metrics
│   ├── stocks/                          # Processed company equities (PARTITIONED BY SECTOR)
│   │   ├── sector=Technology/           # AAPL, MSFT, NVDA, etc.
│   │   ├── sector=Financials/           # JPM, BAC, V, etc.
│   │   ├── sector=Energy/               # XOM, CVX, COP, etc.
│   │   └── ...                          # 11 GICS sector partition directories
│   ├── etfs/
│   │   └── sector_etfs_processed.parquet# Processed sector ETFs with returns and rolling vol
│   ├── macro/
│   │   ├── macro_indicators_native.parquet  # Native frequencies and observation dates
│   │   └── macro_indicators_aligned.parquet # Zero look-ahead bias point-in-time aligned series
│   ├── daily_news_sentiment.parquet     # Daily aggregated news volume and sentiment ratio
│   └── news_events_cleaned.parquet      # Joined news records with timing phase tags
│
└── analytics/                           # LAYER 3: Econometric data products & event studies
    ├── event_study_companies.parquet    # Company event studies: Pre/Post returns, Max DD, CAR
    ├── abnormal_returns_detail.parquet  # Daily AR & CAR time series (T-5 to T+30)
    ├── abnormal_returns_significance.parquet # Parametric t-tests, p-values, sample sizes, assumptions
    ├── cross_company_dispersion.parquet # Intra-sector return dispersion, top resilient vs vulnerable
    ├── event_category_comparisons.parquet # Market and sector returns aggregated by shock type
    ├── macro_context_analysis.parquet   # Yield curve inversion & inflation regime conditioning
    ├── event_study_indices.parquet      # Equity benchmark event studies
    ├── event_study_sectors.parquet      # Sector ETF event studies
    └── benchmark_verification_summary.json # PySpark vs independent NumPy precision benchmarks
```

---

## 3. PySpark Preprocessing Engine

The PySpark processing engine (`src/preprocessing/unified_pipeline.py`) performs distributed transformations in Apache Spark:

1. **Multi-Source Ingestion & Schema Standardisation**:
   - Explicit standard schemas (`MARKET_OHLCV_SCHEMA`) enforcing `DateType`, `DoubleType`, `LongType`, and `StringType`.
   - Handles heterogeneous formats, missing values, placeholder tokens (`.`, `ND`), and deduplication.
2. **Multi-Dataset Joins**:
   - Equities joined with company metadata on `Ticker` (Sector, Industry, Company Name).
   - Equities joined with S&P 500 parent benchmark on `Date` to compute relative excess returns:
     $$\text{Spread}_{i,t} = R_{i,t} - R_{\text{SP500},t}$$
   - GDELT news records joined with historical events to compute `days_from_event` and `timing_phase` (`pre-event`, `day-of`, `post-event`).
   - FRED macroeconomic indicators aligned with trading calendar.
3. **Distributed Window Functions**:
   - Partitioned by security: `Window.partitionBy("Ticker").orderBy("Date")`
   - Lagged prices: $R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$
   - Rolling 21-day standard deviation (monthly volatility): `Window.rowsBetween(-20, 0)`
   - Annualized volatility: $\sigma_{\text{ann}} = \sigma_{21d} \times \sqrt{252}$
   - Cumulative compound growth tracking: $\prod (1 + R_t)$
4. **Date-Based Aggregations**:
   - Aggregation of global news sentiment by publication date, computing daily volume, average tone, and negative sentiment ratio.
5. **Partitioned Parquet Storage**:
   - Company equities are stored in partitioned Parquet directories partitioned by `sector` (`data/processed/stocks/sector=.../`).
   - Automatic partition coalescing prevents small-file proliferation while optimizing query pruning.
   - Resilient write fallback handles Windows environments without native Hadoop `winutils.exe`.

---

## 4. Advanced Financial Analytics & Econometric Methodology

### 4.1 Benchmark-Based Market Model (Abnormal Returns)
To isolate idiosyncratic event impact from broader market trends, the platform implements the classic econometric **Market Model** using Ordinary Least Squares (OLS):

$$R_{i,t} = \alpha_i + \beta_i R_{m,t} + \varepsilon_{i,t}$$

- **Estimation Window**: $[T_0 - 120, T_0 - 11]$ (110 trading days ending 10 days prior to the event, preventing event-contamination of beta estimates).
- **Event Window**: $[T_0 - 5, T_0 + 30]$ (36 active trading days spanning pre-event anticipation and post-event reaction).
- **Abnormal Return ($AR_{i,t}$)**:
  $$AR_{i,t} = R_{i,t} - (\hat{\alpha}_i + \hat{\beta}_i R_{m,t})$$
- **Cumulative Abnormal Return ($CAR_i$)**:
  $$CAR_i(t_1, t_2) = \sum_{t=t_1}^{t_2} AR_{i,t}$$
- **Cumulative Average Abnormal Return ($CAAR$)**:
  $$CAAR(t_1, t_2) = \frac{1}{N} \sum_{i=1}^{N} CAR_i(t_1, t_2)$$

### 4.2 Statistical Significance Testing Framework
To assess whether abnormal returns represent genuine market shocks rather than random noise, the engine conducts parametric cross-sectional two-tailed hypothesis tests:

- **Null Hypothesis ($H_0$)**: $\overline{CAR} = 0$ (no abnormal market impact).
- **Cross-Sectional Standard Error**:
  $$SE = \frac{s_{CAR}}{\sqrt{N}} = \frac{1}{\sqrt{N}} \sqrt{\frac{1}{N - 1} \sum_{i=1}^{N} (CAR_i - \overline{CAR})^2}$$
- **t-Statistic**:
  $$t = \frac{\overline{CAR} - 0}{SE}$$
- **p-Value**: Evaluated against the Student's $t$-distribution with degrees of freedom $df = N - 1$:
  $$p = 2 \times P(T > |t|)$$

### 4.3 Econometric Assumptions & Limitations
The platform explicitly documents and reports all statistical assumptions and boundary limitations:
1. **Normality Assumption**: Abnormal returns are assumed to be approximately normally distributed.
2. **Cross-Sectional Independence**: Assumes security returns are cross-sectionally uncorrelated; synchronous market-wide events may introduce covariance.
3. **Parameter Stationarity**: Assumes market beta ($\beta_i$) and intercept ($\alpha_i$) remain stable from estimation to event windows.
4. **Correlation $\neq$ Causation Warning**: Statistically significant abnormal returns reflect strong empirical market associations, not definitive causal proof. Confounding factors (e.g. concurrent earnings releases or fiscal announcements) cannot be fully eliminated.

---

## 5. Benchmarking & Precision Verification

The platform includes an automated mathematical verification suite (`src/analytics/benchmark_validation.py`) comparing PySpark transformations against independent pure NumPy/SciPy formulas on controlled samples:
- **Return Calculation**: Asserted numerical agreement to $\text{diff} < 10^{-5}$ (achieved $\text{diff} = 0.00$).
- **Rolling Volatility**: Asserted numerical agreement to $\text{diff} < 10^{-5}$ (achieved $\text{diff} = 2.08 \times 10^{-17}$).
- **OLS Market Model Parameters**: Verified $\hat{\alpha}$, $\hat{\beta}$, and residual standard error $\hat{\sigma}_{\varepsilon}$ match standard linear algebra formulas to within machine precision.
- **Timing Benchmarks**: Execution times logged and persisted in `data/analytics/benchmark_verification_summary.json`.

---

## 6. Execution Guide

### Run Full Multi-Layer Pipeline
To execute ingestion, PySpark preprocessing, and financial analytics across all three layers:
```bash
python main.py
```

*Execution Options:*
- `--force-download`: Re-downloads market, ETF, and FRED datasets.
- `--skip-pyspark`: Skips Layer 2 PySpark preprocessing.
- `--skip-analytics`: Skips Layer 3 econometric event study calculations.
- `--skip-etfs`: Skips sector ETF ingestion.
- `--skip-macro`: Skips FRED macroeconomic integration.

### Launch Interactive Streamlit Dashboard
```bash
python -m streamlit run dashboard/app.py
```

### Run Comprehensive Test Suite
```bash
pytest tests/ -v
```

---

## 7. Interactive Dashboard Sections

The Streamlit dashboard (`dashboard/app.py`) provides 14 interactive analytical modules:
1. **Equity Index Daily Closing Level**: Historical index trends with annotated event dates.
2. **Daily Returns and Volatility Distribution**: Interactive return distributions and histograms.
3. **CBOE Volatility Index (VIX)**: Regime threshold lines at 20 (Elevated) and 30 (High).
4. **Event-Study Returns by Event**: Pre-5D, Post-5D, and Post-30D returns for selected index.
5. **Cross-Index Comparison**: Multi-index response side-by-side comparison for any event.
6. **Quantitative Event Summary Table**: Filterable and downloadable metrics panel.
7. **Event Detail Drill-Down**: Normalized price ($T_0 = 100$) and VIX trajectory over $\pm 30$ trading days.
8. **US Sector ETF Performance & Multi-Sector Comparison**: Normalized rebased price comparison and correlation heatmap.
9. **FRED Macroeconomic Indicators & Yield Curve (Zero Look-Ahead Bias)**: Point-in-time Treasury yields, 10Y-2Y spread inversions, CPI inflation, and consumer sentiment.
10. **Sector ETF Performance During Major Global Events**: Defensive vs cyclical sector returns around external shocks.
11. **Company-Level Event Studies & Abnormal Returns (Market Model AR & CAR)**:
    - Drill-down across 300+ equities with sector filtering.
    - Post-5D and Post-30D Cumulative Abnormal Returns ($CAR$).
    - Relative excess return spread vs S&P 500 benchmark.
    - Interactive $CAR$ trajectory plot over $[-5, +30]$ trading days.
12. **Cross-Company Intra-Sector Dispersion & Resilience Rankings**:
    - Intra-sector return spread ranges, standard deviations, and identifying top resilient vs most vulnerable equities.
13. **Statistical Significance Testing & Econometric Hypothesis Validation**:
    - Parametric $t$-statistics, $p$-values, standard errors, sample sizes ($N$), and rejection of $H_0$.
    - Econometric assumptions, limitations, and correlation-vs-causation callout.
14. **Macroeconomic Regime Conditioning & Equity Volatility Dynamics**:
    - Conditioned market performance across historical yield curve inversion regimes and inflation regimes.

---

## 8. Machine Learning Extension: Geopolitical Conflict & Armed Escalation Market Impact Prediction

### 8.1 Objective & Fixed Category Hierarchy
The machine learning extension forecasts the forward financial return impact of geopolitical conflict shocks on major US equity benchmark indices:
- **Primary Event Category**: Strictly **War and Armed Conflicts**.
- **Five Canonical Geographical Subcategories**:
  1. `Middle East / MENA` (121 historical training episodes)
  2. `Eastern Europe` (56 historical training episodes)
  3. `South Asia` (114 historical training episodes)
  4. `East Asia` (3 historical training episodes; Taiwan Strait & Korean Peninsula crises)
  5. `North America` (16 historical training episodes; domestic violent extremist incidents)
- **Four Benchmark Prediction Targets**: S&P 500 (`^GSPC`), NASDAQ Composite (`^IXIC`), Dow Jones Industrial Average (`^DJI`), Russell 2000 (`^RUT`).
- **Two Prediction Horizons**: **5 trading days** ($R_{+5}$) and **20 trading days** ($R_{+20}$).
- **Total Simultaneous Outputs**: Exactly **eight continuous numerical percentage returns** generated per scenario.

### 8.2 Datasets & PySpark Preprocessing
- **UCDP GED v26.1**: Uppsala Conflict Data Program Georeferenced Event Dataset (417,968 raw incident records across 126 countries, 1989–2025).
- **PySpark Conflict Pipeline (`src/preprocessing/conflict_pipeline.py`)**:
  - Filters high temporal precision (`date_prec <= 2`, 2015–2025 window: 210,249 incidents).
  - Geographic mapping of all 126 countries to the 5 canonical regions or documented excluded regions.
  - Spatial-temporal episode clustering: Groups incidents by `(canonical_region, conflict_new_id, event_date)` with $\text{fatalities} \ge 25$.
  - Applies a 20-trading-day (28-calendar-day) regional washout window to prevent multi-counting single protracted conflicts.
  - Curated augmentation for East Asia historical escalation milestones (Pelosi Taiwan visit, 2017 North Korea ICBM tests, 2020 Galwan Valley).
  - Produces `data/processed/conflict_episodes/conflict_episodes.parquet` (**314 independent episodes**).

### 8.3 Feature Engineering & Strict Zero Look-Ahead Protection (`src/ml/feature_engineering.py`)
- Maps conflict calendar start dates to effective US trading day $T_0$ via `find_effective_trading_day_index`.
- Strictly excludes ex-post conflict duration from prediction time inputs; duration is discretized into ex-ante operational regimes:
  - `Short / Tactical (< 30 days)`
  - `Medium / Campaign (30–180 days)`
  - `Protracted / War (> 180 days)`
- Input features evaluated strictly **at or before** $T_0$:
  - Pre-event market momentum: 5-day return ($R_{-5}$) and 20-day return ($R_{-20}$).
  - Realized volatility: 21-day annualized volatility ($\sigma_{21\text{D}}$).
  - Market trend: Price deviation from 50-day Simple Moving Average ($P_{T_0} / \text{SMA}_{50} - 1$).
  - Implied volatility: Spot VIX level and 5-day percentage change ($\Delta \text{VIX}_{-5}$).
  - FRED macro conditioning: Point-in-time Treasury 10Y–2Y yield spread, 10Y yield, and 2Y yield.
  - Conflict intensity: Battle fatalities, log-fatalities, incident count, and violence typology.
- Persisted to `data/processed/ml/conflict_market_training.parquet` (**310 usable observations × 38 columns, 0 nulls**; 4 early 2015 episodes excluded due to $T_0 < 20$ lookback boundary constraints).

### 8.4 Model Architecture & Evaluation (`src/ml/train_models.py`)
- **Primary Model**: Unified Multi-Output `RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=5, random_state=42)`.
  - Natively predicts all 8 continuous returns simultaneously in leaf node vectors.
  - Captures piecewise non-linear regime shifts (e.g. market response differences when spot VIX > 25 vs < 13).
  - Constrained tree depth acts as structural regularization against financial time-series noise.
- **Baseline Model**: Multi-Output `Ridge(alpha=10.0)` with `StandardScaler` and `OneHotEncoder`.
- **Validation Framework**:
  - *Chronological Split*: Train on 2015–2022 (216 episodes, 30 conflicts), Test out-of-sample on 2023–2025 (94 episodes, 19 conflicts).
  - *Conflict-Grouped Cross-Validation*: 5-fold `GroupKFold` grouped by `conflict_new_id` within the training set.
- **Out-of-Sample Performance Comparison (2023–2025 Test Set)**:
  - Random Forest beats Ridge regression across **all 8 targets** (Macro Avg MAE: $3.093\%$ vs $3.461\%$; RMSE: $3.755\%$ vs $4.168\%$).
  - Positive out-of-sample $R^2$ achieved on 5-day horizon: S&P 500 ($+0.035$), Dow Jones ($+0.020$), Russell 2000 ($+0.009$).
  - Negative $R^2$ on 20-day horizon reflects the 2023–2025 mega-cap AI secular bull market regime shift compared to the 2015–2022 training regime.
- **Key Feature Importances**:
  - Implied Volatility (`vix_level`): 24.40%
  - Market Trend (`sma_50_ratio_mkt`): 13.70%
  - Short-End Yields (`treasury_yield_2y`): 11.97%
  - Pre-Event Momentum (`pre_20d_return_mkt`): 9.66%
  - Financial/macro conditioning accounts for >78% of predictive power.

### 8.5 Prediction Engine & Dashboard Integration (`src/ml/predictor.py` & `dashboard/app.py`)
- Integrated into Streamlit as **Module 4: AI/ML Conflict Impact Predictor**:
  - Interactive scenario definition (canonical region, expected duration in days, fatalities, violence type).
  - Market condition adjustments with historical training median fallbacks.
  - Exact prediction table with all 8 returns across the four benchmarks.
  - Side-by-side Plotly bar chart comparing 5-day and 20-day predicted returns.
  - Synthesized econometric explanation card based on key model drivers.
  - Empirical data sparsity warnings for East Asia and North America.
  - Diagnostic tabs displaying feature importance, out-of-sample performance scorecards, and model metadata.

### 8.6 Execution Commands
```bash
# 1. Run PySpark Conflict Preprocessing Pipeline:
python -m src.preprocessing.conflict_pipeline

# 2. Generate Supervised Feature & Target Dataset:
python -m src.ml.feature_engineering

# 3. Train Models, Run Cross-Validation, & Save Artifacts:
python -m src.ml.train_models

# 4. Launch Streamlit Dashboard with AI/ML Predictor Module:
python -m streamlit run dashboard/app.py

# 5. Run Complete Test Suite (198 Tests):
pytest tests/ -v
```

### 8.7 Known Limitations & Academic Disclaimers
1. **Severe Data Sparsity in East Asia**: Only 3 historical conflict escalation episodes occurred in East Asia (Taiwan Strait / Korean Peninsula) during 2015–2025. Predictions rely heavily on cross-regional transfer learning.
2. **North America Typology**: The 16 North American incidents reflect domestic violent extremism rather than conventional interstate warfare.
3. **Macroeconomic Non-Stationarity**: Forward return relationships are influenced by monetary regime shifts (e.g. 2022 rate hike cycle vs. 2023–2025 AI bull market).
4. **Correlation $\neq$ Causation**: Model predictions represent econometric statistical estimates conditional on historical patterns, not guaranteed causal forecasts or trading advice.
