"""
Streamlit dashboard for multi-index global event and US stock-market analysis.

Compatible with the long-format output produced by src/analytics/event_study.py:
one row per event/index pair, with generic metric columns and separate VIX fields.
Run from the repository root:
    python -m streamlit run dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# Resolve paths from the repository root, not from the caller's working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INDEX_CONFIG = {
    "sp500": {
        "label": "S&P 500",
        "ticker": "^GSPC",
        "path": PROCESSED_DIR / "sp500_processed.parquet",
    },
    "nasdaq": {
        "label": "NASDAQ Composite",
        "ticker": "^IXIC",
        "path": PROCESSED_DIR / "nasdaq_processed.parquet",
    },
    "dow_jones": {
        "label": "Dow Jones Industrial Average",
        "ticker": "^DJI",
        "path": PROCESSED_DIR / "dow_jones_processed.parquet",
    },
    "russell2000": {
        "label": "Russell 2000",
        "ticker": "^RUT",
        "path": PROCESSED_DIR / "russell2000_processed.parquet",
    },
}
VIX_PATH = PROCESSED_DIR / "vix_processed.parquet"
EVENT_STUDY_PATH = PROCESSED_DIR / "event_study_results.csv"

METRIC_COLUMNS = {
    "Pre-5D Return (%)": "pre_5d_return_pct",
    "Post-5D Return (%)": "post_5d_return_pct",
    "Post-30D Return (%)": "post_30d_return_pct",
}
DISPLAY_COLUMNS = [
    "event_name",
    "calendar_event_date",
    "effective_trading_date",
    "event_category",
    "index",
    "ticker",
    "price_t0",
    "pre_5d_return_pct",
    "post_5d_return_pct",
    "post_30d_return_pct",
    "max_drawdown_30d_pct",
    "vix_at_event",
    "vix_5d_change",
    "vix_max_30d",
    "short_description",
]
DISPLAY_NAMES = {
    "event_name": "Event",
    "calendar_event_date": "Calendar Event Date",
    "effective_trading_date": "Effective Trading Date (T0)",
    "event_category": "Category",
    "index": "Index",
    "ticker": "Ticker",
    "price_t0": "Close at T0",
    "pre_5d_return_pct": "Pre-5D Return (%)",
    "post_5d_return_pct": "Post-5D Return (%)",
    "post_30d_return_pct": "Post-30D Return (%)",
    "max_drawdown_30d_pct": "30D Minimum Return / Drawdown (%)",
    "vix_at_event": "VIX at T0",
    "vix_5d_change": "VIX Change over 5D",
    "vix_max_30d": "VIX Maximum over 30D",
    "short_description": "Description",
}


def _read_parquet(path: Path) -> Optional[pd.DataFrame]:
    """Read a processed Parquet dataset and normalise its Date column."""
    if not path.exists():
        return None

    frame = pd.read_parquet(path)
    if "Date" not in frame.columns:
        raise ValueError(f"{path.name} does not contain the required 'Date' column.")

    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).sort_values("Date")
    frame = frame.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return frame


@st.cache_data(show_spinner=False)
def load_index_data(index_key: str) -> Optional[pd.DataFrame]:
    """Load one of the four processed equity-index datasets."""
    if index_key not in INDEX_CONFIG:
        raise ValueError(f"Unsupported index key: {index_key}")
    return _read_parquet(INDEX_CONFIG[index_key]["path"])


@st.cache_data(show_spinner=False)
def load_vix_data() -> Optional[pd.DataFrame]:
    """Load the market-wide VIX dataset independently of the equity indices."""
    return _read_parquet(VIX_PATH)


@st.cache_data(show_spinner=False)
def load_event_study() -> Optional[pd.DataFrame]:
    """Load and validate the new long-format event-study CSV."""
    if not EVENT_STUDY_PATH.exists():
        return None

    frame = pd.read_csv(EVENT_STUDY_PATH)
    required = {
        "event_name",
        "effective_trading_date",
        "index",
        "index_key",
        "pre_5d_return_pct",
        "post_5d_return_pct",
        "post_30d_return_pct",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Event-study CSV is not in the expected multi-index format. "
            f"Missing columns: {', '.join(sorted(missing))}. "
            "Run `python main.py` to regenerate the analytics output."
        )

    frame = frame.copy()
    frame["effective_trading_date"] = pd.to_datetime(
        frame["effective_trading_date"], errors="coerce"
    )
    if "calendar_event_date" in frame.columns:
        frame["calendar_event_date"] = pd.to_datetime(
            frame["calendar_event_date"], errors="coerce"
        )
    return frame


# Backwards-compatible helper for the existing dashboard data-loading tests.
def load_market_data():
    """Return the S&P 500 and VIX frames, as the original dashboard did."""
    return load_index_data("sp500"), load_vix_data()


def _format_number(value, suffix: str = "", digits: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{digits}f}{suffix}"


def _event_lines(fig, event_rows: pd.DataFrame):
    """Add one vertical marker per event, avoiding duplicate index rows."""
    if event_rows.empty:
        return fig
    unique_events = event_rows.drop_duplicates(
        subset=["event_name", "effective_trading_date"]
    )
    for _, event in unique_events.iterrows():
        date = event["effective_trading_date"]
        if pd.notna(date):
            fig.add_vline(
                x=date,
                line_width=1,
                line_dash="dash",
                line_color="rgba(220, 60, 60, 0.65)",
            )
    return fig


def main():
    st.set_page_config(
        page_title="Global Events & US Stock Market Analysis",
        page_icon="📈",
        layout="wide",
    )

    st.title("Impact of Global Events on the US Stock Market")
    st.caption(
        "Multi-index event study | Apache Spark · Pandas · Streamlit · Plotly"
    )

    # Load the new backend outputs.
    with st.spinner("Loading processed market and event-study datasets..."):
        index_frames = {
            key: load_index_data(key) for key in INDEX_CONFIG
        }
        vix_df = load_vix_data()
        event_df = load_event_study()

    missing_files = [
        str(config["path"].relative_to(PROJECT_ROOT))
        for key, config in INDEX_CONFIG.items()
        if index_frames[key] is None
    ]
    if vix_df is None:
        missing_files.append(str(VIX_PATH.relative_to(PROJECT_ROOT)))
    if event_df is None:
        missing_files.append(str(EVENT_STUDY_PATH.relative_to(PROJECT_ROOT)))

    if missing_files:
        st.warning(
            "Some processed datasets are missing. Run the full pipeline from "
            "the project root before opening the dashboard."
        )
        for missing in missing_files:
            st.code(missing)
        st.code("python main.py", language="powershell")
        return

    # Defensive validation: all market datasets need Close for price charts.
    invalid_frames = [
        key for key, frame in index_frames.items()
        if "Close" not in frame.columns
    ]
    if invalid_frames or "Close" not in vix_df.columns:
        st.error(
            "A processed market dataset is missing its 'Close' column. "
            f"Invalid equity datasets: {invalid_frames or 'none'}; "
            f"VIX valid: {'Close' in vix_df.columns}."
        )
        return

    # Sidebar controls.
    st.sidebar.header("Dashboard Controls")
    selected_key = st.sidebar.selectbox(
        "Select equity index",
        options=list(INDEX_CONFIG),
        format_func=lambda key: (
            f"{INDEX_CONFIG[key]['label']} ({INDEX_CONFIG[key]['ticker']})"
        ),
        index=0,
    )
    selected_config = INDEX_CONFIG[selected_key]
    selected_market = index_frames[selected_key]

    st.sidebar.markdown("---")
    st.sidebar.caption("Pipeline: yfinance → PySpark preprocessing → event study")
    show_raw_table = st.sidebar.checkbox(
        "Show recent processed market records", value=False
    )

    # Keep only rows for the selected index. The event study is long-format:
    # each event appears once for each equity index.
    selected_events = event_df[event_df["index_key"] == selected_key].copy()
    selected_events = selected_events.sort_values("effective_trading_date")

    # KPI row.
    k1, k2, k3, k4 = st.columns(4)
    market_start = selected_market["Date"].min().date()
    market_end = selected_market["Date"].max().date()
    avg_post_5d = selected_events["post_5d_return_pct"].mean()

    k1.metric("Trading Sessions", f"{len(selected_market):,}")
    k2.metric("Historical Coverage", f"{market_start} to {market_end}")
    k3.metric("Events Analysed", f"{selected_events['event_name'].nunique():,}")
    k4.metric("Average Post-5D Return", _format_number(avg_post_5d, "%"))

    st.markdown("---")

    # 1. Selected index historical price series.
    st.subheader(f"1. {selected_config['label']} Historical Closing Level")
    st.caption(
        f"Ticker: {selected_config['ticker']} | "
        "Dashed vertical lines indicate effective trading dates of curated events."
    )
    price_fig = px.line(
        selected_market,
        x="Date",
        y="Close",
        title=f"{selected_config['label']} Daily Closing Level",
        labels={"Close": "Index Level", "Date": "Trading Date"},
    )
    _event_lines(price_fig, selected_events)
    price_fig.update_layout(hovermode="x unified", height=440)
    st.plotly_chart(price_fig, use_container_width=True)

    # 2. Daily returns and rolling volatility, when available in preprocessing.
    if "daily_return_pct" in selected_market.columns:
        st.subheader(f"2. {selected_config['label']} Daily Returns")
        returns_df = selected_market.dropna(subset=["daily_return_pct"])
        left, right = st.columns([2, 1])
        with left:
            returns_fig = px.bar(
                returns_df,
                x="Date",
                y="daily_return_pct",
                title="Daily Percentage Returns",
                labels={"daily_return_pct": "Daily Return (%)"},
                color="daily_return_pct",
                color_continuous_scale="RdBu",
            )
            returns_fig.update_layout(height=360)
            st.plotly_chart(returns_fig, use_container_width=True)
        with right:
            hist_fig = px.histogram(
                returns_df,
                x="daily_return_pct",
                nbins=70,
                title="Daily Return Distribution",
                labels={"daily_return_pct": "Daily Return (%)"},
            )
            hist_fig.update_layout(height=360, showlegend=False)
            st.plotly_chart(hist_fig, use_container_width=True)

    # 3. VIX remains a separate market-wide indicator, not a fifth equity index.
    st.subheader("3. CBOE Volatility Index (VIX)")
    st.caption(
        "VIX is shown separately because it represents market-implied volatility, "
        "not an equity index in the four-index comparison."
    )
    vix_fig = px.line(
        vix_df,
        x="Date",
        y="Close",
        title="VIX Daily Closing Level",
        labels={"Close": "VIX Level", "Date": "Trading Date"},
    )
    vix_fig.add_hline(
        y=20, line_dash="dot", annotation_text="Elevated volatility (20)"
    )
    vix_fig.add_hline(
        y=30, line_dash="dash", annotation_text="High volatility (30)"
    )
    vix_fig.update_layout(hovermode="x unified", height=380)
    st.plotly_chart(vix_fig, use_container_width=True)

    st.markdown("---")

    # 4. Event-window comparison for the selected index.
    st.subheader("4. Event-Study Returns by Event")
    st.caption(
        "Pre-5D: T−5 to T0 | Post-5D: T0 to T+5 | "
        "Post-30D: T0 to T+30. Incomplete post-event windows remain unavailable."
    )

    metric_long = selected_events.melt(
        id_vars=["event_name"],
        value_vars=list(METRIC_COLUMNS.values()),
        var_name="metric",
        value_name="return_pct",
    )
    metric_labels = {
        value: label for label, value in METRIC_COLUMNS.items()
    }
    metric_long["Window"] = metric_long["metric"].map(metric_labels)
    metric_fig = px.bar(
        metric_long,
        x="event_name",
        y="return_pct",
        color="Window",
        barmode="group",
        title=f"{selected_config['label']}: Returns Around Major Global Events",
        labels={
            "event_name": "Global Event",
            "return_pct": "Return (%)",
        },
        color_discrete_map={
            "Pre-5D Return (%)": "#636EFA",
            "Post-5D Return (%)": "#EF553B",
            "Post-30D Return (%)": "#00CC96",
        },
    )
    metric_fig.add_hline(y=0, line_color="gray", line_width=1)
    metric_fig.update_layout(height=440, xaxis_tickangle=-20)
    st.plotly_chart(metric_fig, use_container_width=True)

    # 5. Compare the four indices for one selected event.
    st.subheader("5. Cross-Index Comparison")
    available_events = sorted(event_df["event_name"].dropna().unique().tolist())
    if available_events:
        chosen_event = st.selectbox(
            "Choose an event to compare across all four indices",
            options=available_events,
            key="cross_index_event",
        )
        cross_rows = event_df[event_df["event_name"] == chosen_event].copy()
        cross_long = cross_rows.melt(
            id_vars=["index"],
            value_vars=list(METRIC_COLUMNS.values()),
            var_name="metric",
            value_name="return_pct",
        )
        cross_long["Window"] = cross_long["metric"].map(metric_labels)
        cross_fig = px.bar(
            cross_long,
            x="index",
            y="return_pct",
            color="Window",
            barmode="group",
            title=f"Index Performance Around {chosen_event}",
            labels={"index": "Equity Index", "return_pct": "Return (%)"},
        )
        cross_fig.add_hline(y=0, line_color="gray", line_width=1)
        cross_fig.update_layout(height=420, xaxis_tickangle=-15)
        st.plotly_chart(cross_fig, use_container_width=True)

    # 6. Long-format event summary table.
    st.subheader("6. Event Summary & Quantitative Metrics")
    st.caption(
        "The table is filtered to the selected index. Each event-index combination "
        "is one row in the backend output."
    )
    existing_display_cols = [
        col for col in DISPLAY_COLUMNS if col in selected_events.columns
    ]
    summary_table = selected_events[existing_display_cols].rename(
        columns=DISPLAY_NAMES
    )
    st.dataframe(
        summary_table,
        use_container_width=True,
        hide_index=True,
    )

    # 7. Event drill-down: selected index's own trading calendar.
    st.markdown("---")
    st.subheader("7. Event Detail Drill-Down (±30 Trading Sessions)")
    drill_events = selected_events.dropna(
        subset=["event_name", "effective_trading_date"]
    )
    if drill_events.empty:
        st.info("No usable event dates are available for this index.")
    else:
        selected_event_name = st.selectbox(
            "Select an event to inspect its price and volatility path",
            options=drill_events["event_name"].tolist(),
            key="drilldown_event",
        )
        event_row = drill_events[
            drill_events["event_name"] == selected_event_name
        ].iloc[0]
        t0 = pd.Timestamp(event_row["effective_trading_date"])

        description = event_row.get("short_description")
        category = event_row.get("event_category")
        st.info(
            f"**{selected_event_name}**"
            + (f" ({category})" if pd.notna(category) else "")
            + (f": {description}" if pd.notna(description) else "")
            + f" | Effective trading date: {t0.date()}"
        )

        # Use this index's own processed trading calendar, not another index's.
        t0_matches = selected_market.index[
            selected_market["Date"].dt.normalize() == t0.normalize()
        ].tolist()
        if not t0_matches:
            st.warning(
                "The event's effective trading date is not present in this "
                "index's processed price dataset."
            )
        else:
            t0_position = t0_matches[0]
            start_position = max(0, t0_position - 30)
            end_position = min(len(selected_market) - 1, t0_position + 30)
            window = selected_market.iloc[
                start_position : end_position + 1
            ].copy()
            base_price = selected_market.iloc[t0_position]["Close"]

            if pd.isna(base_price) or base_price == 0:
                st.warning("Cannot normalise this event window because T0 close is invalid.")
            else:
                window["normalised_level"] = (
                    window["Close"] / base_price
                ) * 100.0
                drill_fig = make_subplots(specs=[[{"secondary_y": True}]])
                drill_fig.add_trace(
                    go.Scatter(
                        x=window["Date"],
                        y=window["normalised_level"],
                        mode="lines",
                        name=f"{selected_config['label']} (T0 = 100)",
                    ),
                    secondary_y=False,
                )

                # Align VIX to the selected index's event-window dates.
                vix_window = vix_df[
                    vix_df["Date"].isin(window["Date"])
                ][["Date", "Close"]].dropna()
                if not vix_window.empty:
                    drill_fig.add_trace(
                        go.Scatter(
                            x=vix_window["Date"],
                            y=vix_window["Close"],
                            mode="lines",
                            name="VIX",
                            line={"dash": "dot"},
                        ),
                        secondary_y=True,
                    )

                drill_fig.add_vline(
                    x=t0,
                    line_dash="dash",
                    line_color="crimson",
                    annotation_text="T0",
                )
                drill_fig.add_hline(
                    y=100,
                    line_dash="dot",
                    line_color="gray",
                    secondary_y=False,
                )
                drill_fig.update_yaxes(
                    title_text="Equity index (T0 = 100)",
                    secondary_y=False,
                )
                drill_fig.update_yaxes(
                    title_text="VIX level",
                    secondary_y=True,
                )
                drill_fig.update_layout(
                    title=f"{selected_event_name}: {selected_config['label']} and VIX",
                    hovermode="x unified",
                    height=440,
                )
                st.plotly_chart(drill_fig, use_container_width=True)

    if show_raw_table:
        st.markdown("---")
        st.subheader(f"Recent Processed Records: {selected_config['label']}")
        st.dataframe(
            selected_market.tail(20),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
