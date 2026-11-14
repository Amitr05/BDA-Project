# Impact of Geopolitical Conflicts and Major Global Events on the US Stock Market: A Big Data Analytics Approach

**Phase 1 Base Prototype**

## 1. Project Overview
This project investigates the empirical impact of major geopolitical conflicts and macroeconomic global events on the United States stock market benchmark (**S&P 500 / `^GSPC`**) and volatility regime (**CBOE Volatility Index / `^VIX`**).

The architecture leverages **Apache Spark (PySpark SQL)** for distributed, scalable data preprocessing and rolling window time-series transformations, paired with an interactive **Streamlit & Plotly** analytical dashboard.

---

## 2. Phase 1 Architecture & Structure

```
global-event-market-analysis/
│
├── data/
│   ├── raw/
│   │   ├── sp500_raw.csv                # Raw historical daily S&P 500 (^GSPC)
│   │   ├── vix_raw.csv                  # Raw historical daily CBOE VIX (^VIX)
│   │   └── events.csv                   # Curated global events dataset (2015–2026)
│   └── processed/
│       ├── sp500_processed.parquet      # Preprocessed S&P 500 with returns & rolling volatility
│       ├── vix_processed.parquet        # Preprocessed VIX data
│       ├── event_study_results.csv      # Quantitative event-study metrics (Pre-5D, Post-5D, Post-30D)
│       └── event_study_results.parquet  # Parquet format of event study dataset
│
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── market_data.py               # Ingestion of ^GSPC & ^VIX using yfinance (caching & validation)
│   │   └── event_data.py                # Curated event calendar generation and schema validation
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── spark_session.py             # PySpark session builder with Java 17 & Windows support
│   │   └── market_pipeline.py           # PySpark transformations (returns, volume change, rolling vol)
│   └── analytics/
│       ├── __init__.py
│       └── event_study.py               # Trading-day event-window analysis (T-5 to T+30)
│
├── dashboard/
│   └── app.py                          # Streamlit & Plotly interactive dashboard
│
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py               # Ingestion & event dataset validation tests
│   ├── test_preprocessing.py           # PySpark transformations & window calculations tests
│   ├── test_analytics.py               # Trading-day window mapping & returns tests
│   └── test_dashboard.py               # Dashboard data contract tests
│
├── main.py                             # Master pipeline orchestration script
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation & execution guide
```

---

## 3. Methodology & Event Window Definition

Event-study analysis evaluates price behavior before, during, and after an external shock. Instead of arbitrary calendar intervals, **all windows are evaluated over active trading days**:

1. **Effective Trading Day ($T_0$):**
   - If an event occurs when the NYSE is open, $T_0$ is the event date.
   - If an event occurs on a weekend or public holiday, $T_0$ is automatically mapped to the **first subsequent active trading day**.
2. **Pre-Event Window (5 Trading Days):**
   - Interval: $[T_{-5}, T_0]$
   - Pre-5D Return:
     $$\text{Return}_{\text{Pre-5D}} = \frac{P(T_0) - P(T_{-5})}{P(T_{-5})} \times 100\%$$
3. **Post-Event Short Window (5 Trading Days):**
   - Interval: $[T_0, T_{+5}]$
   - Post-5D Return:
     $$\text{Return}_{\text{Post-5D}} = \frac{P(T_{+5}) - P(T_0)}{P(T_0)} \times 100\%$$
4. **Post-Event Medium Window (30 Trading Days):**
   - Interval: $[T_0, T_{+30}]$
   - Post-30D Return:
     $$\text{Return}_{\text{Post-30D}} = \frac{P(T_{+30}) - P(T_0)}{P(T_0)} \times 100\%$$
5. **Volatility Shock Metrics:**
   - $VIX(T_0)$, $VIX(T_{+5})$, $\Delta VIX_{\text{5D}} = VIX(T_{+5}) - VIX(T_0)$, and $\max_{t \in [T_0, T_{+30}]} VIX(t)$.

---

## 4. Setup & Installation

### Prerequisites
- Python 3.11
- Java 17 (e.g. Eclipse Adoptium Temurin 17)

### 1. Create and Activate Virtual Environment
```bash
py -3.11 -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 5. Execution Guide

### Run Full Pipeline
To run data ingestion, PySpark preprocessing, and event study calculations:
```bash
python main.py
```
*Note: To force re-downloading fresh market data, add the `--force-download` flag:*
```bash
python main.py --force-download
```

### Launch Interactive Dashboard
```bash
python -m streamlit run dashboard/app.py
```

### Run Test Suite
```bash
pytest tests/ -v
```

---

## 6. PySpark on Windows Notes
- **Java 17 Module Options:** PySpark requires JVM `--add-opens` flags on Java 17+, which are automatically configured in `src/preprocessing/spark_session.py`.
- **Hadoop winutils.exe Handling:** On Windows, native Hadoop write calls can fail if `winutils.exe` is absent. `src/preprocessing/market_pipeline.py` implements a resilient export strategy: attempting native Spark parquet writes with automatic graceful fallback to PyArrow without data loss.

---

## 7. Scope Boundaries (Phase 1 vs. Phase 2)
### Completed in Phase 1
- Historical daily data ingestion for S&P 500 and VIX (2015–2026).
- Curated calendar of 7 major global events across geopolitical, financial, and health categories.
- PySpark cleaning, duplicate elimination, daily returns, volume change %, and 21-day rolling volatility.
- Trading-day event study metrics (Pre-5D, Post-5D, Post-30D returns & VIX levels).
- Streamlit interactive dashboard with Plotly charts.
- Modular unit test suite.

### Planned for Phase 2
- GDELT news event sentiment and media tone integration.
- FRED macroeconomic indicators (US Treasury Yields, Inflation, Fed Funds Rate).
- Sector-level ETF analysis (XLE Energy, XLF Financials, XLU Utilities, etc.).
- Statistical significance tests (t-tests, cumulative abnormal return CAR models).
