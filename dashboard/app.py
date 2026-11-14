"""
Streamlit Dashboard: Impact of Global Events on the US Stock Market
Phase 1 Base Prototype
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set Page Config
st.set_page_config(
    page_title="Global Events & US Stock Market Analysis",
    page_icon="📈",
    layout="wide"
)

SP500_PARQUET = "data/processed/sp500_processed.parquet"
VIX_PARQUET = "data/processed/vix_processed.parquet"
EVENT_STUDY_CSV = "data/processed/event_study_results.csv"
EVENTS_CSV = "data/raw/events.csv"


@st.cache_data
def load_market_data():
    """Load preprocessed S&P 500 and VIX data."""
    if not os.path.exists(SP500_PARQUET) or not os.path.exists(VIX_PARQUET):
        return None, None
    sp500_df = pd.read_parquet(SP500_PARQUET)
    vix_df = pd.read_parquet(VIX_PARQUET)
    sp500_df["Date"] = pd.to_datetime(sp500_df["Date"])
    vix_df["Date"] = pd.to_datetime(vix_df["Date"])
    return sp500_df, vix_df


@st.cache_data
def load_event_study():
    """Load event study results."""
    if not os.path.exists(EVENT_STUDY_CSV):
        return None
    df = pd.read_csv(EVENT_STUDY_CSV)
    return df


def main():
    st.title("Impact of Global Events on the US Stock Market")
    st.caption("Phase 1 Base Prototype | Big Data Analytics Approach (PySpark + Streamlit)")

    sp500_df, vix_df = load_market_data()
    event_study_df = load_event_study()

    if sp500_df is None or vix_df is None or event_study_df is None:
        st.warning("Processed datasets not found. Please run the ingestion and PySpark preprocessing pipeline first:")
        st.code("python main.py", language="bash")
        return

    # Sidebar
    st.sidebar.header("Navigation & Options")
    st.sidebar.markdown(
        """
        **Pipeline Components:**
        - **Data Ingestion:** yfinance (^GSPC, ^VIX)
        - **Engine:** PySpark SQL & Window Functions
        - **Event Window:** Trading-Day Offsets (-5D, +5D, +30D)
        """
    )
    show_raw_table = st.sidebar.checkbox("Show Raw Market Data Samples", value=False)

    # Key Metrics Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.metric("Total Trading Days", f"{len(sp500_df):,}")
    with kpi_col2:
        st.metric("S&P 500 Date Range", f"{sp500_df['Date'].dt.date.min()} to {sp500_df['Date'].dt.date.max()}")
    with kpi_col3:
        st.metric("Curated Global Events", f"{len(event_study_df)}")
    with kpi_col4:
        avg_post5d = event_study_df["sp500_post_5d_return_pct"].mean()
        st.metric("Avg Post-5D S&P Return", f"{avg_post5d:.2f}%")

    st.markdown("---")

    # 1. Historical S&P 500 Close Price Chart
    st.subheader("1. S&P 500 Historical Closing Price (^GSPC)")
    fig_sp500 = px.line(
        sp500_df,
        x="Date",
        y="Close",
        title="S&P 500 Daily Closing Price (2015 – 2026)",
        labels={"Close": "Index Value ($)", "Date": "Trading Date"}
    )
    # Add vertical lines for curated events
    for _, ev in event_study_df.iterrows():
        fig_sp500.add_vline(
            x=pd.to_datetime(ev["effective_trading_date"]),
            line_width=1,
            line_dash="dash",
            line_color="crimson",
            annotation_text=ev["event_name"],
            annotation_position="top left",
            annotation_font_size=10
        )
    fig_sp500.update_layout(hovermode="x unified", height=450)
    st.plotly_chart(fig_sp500, use_container_width=True)

    # 2. VIX Historical Trend
    st.subheader("2. CBOE Volatility Index (^VIX) Historical Trend")
    fig_vix = px.line(
        vix_df,
        x="Date",
        y="Close",
        title="CBOE Volatility Index (^VIX)",
        labels={"Close": "VIX Level", "Date": "Trading Date"},
        color_discrete_sequence=["orange"]
    )
    # Volatility threshold markers (20 elevated, 30 crisis)
    fig_vix.add_hline(y=20, line_dash="dot", line_color="goldenrod", annotation_text="Elevated Volatility (20)")
    fig_vix.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Extreme Volatility / Crisis (30)")
    fig_vix.update_layout(hovermode="x unified", height=400)
    st.plotly_chart(fig_vix, use_container_width=True)

    # 3. S&P 500 Daily Percentage Returns
    st.subheader("3. S&P 500 Daily Percentage Returns")
    col_ret1, col_ret2 = st.columns([2, 1])

    with col_ret1:
        fig_returns = px.bar(
            sp500_df.dropna(subset=["daily_return_pct"]),
            x="Date",
            y="daily_return_pct",
            title="S&P 500 Daily Returns (%) Over Time",
            labels={"daily_return_pct": "Daily Return (%)"},
            color="daily_return_pct",
            color_continuous_scale="RdBu"
        )
        fig_returns.update_layout(height=380)
        st.plotly_chart(fig_returns, use_container_width=True)

    with col_ret2:
        fig_hist = px.histogram(
            sp500_df.dropna(subset=["daily_return_pct"]),
            x="daily_return_pct",
            nbins=80,
            title="Return Distribution",
            labels={"daily_return_pct": "Daily Return (%)"},
            color_discrete_sequence=["steelblue"]
        )
        fig_hist.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # 4. Event-Study Comparison Chart
    st.subheader("4. Event Study: S&P 500 Returns Across Event Windows")
    st.caption("Explicit trading-day windows: Pre-5D (T-5 to T0), Post-5D (T0 to T+5), and Post-30D (T0 to T+30).")

    # Prepare data for grouped bar chart
    plot_rows = []
    for _, row in event_study_df.iterrows():
        plot_rows.append({"Event": row["event_name"], "Window": "Pre-5D Return (%)", "Return (%)": row["sp500_pre_5d_return_pct"]})
        plot_rows.append({"Event": row["event_name"], "Window": "Post-5D Return (%)", "Return (%)": row["sp500_post_5d_return_pct"]})
        plot_rows.append({"Event": row["event_name"], "Window": "Post-30D Return (%)", "Return (%)": row["sp500_post_30d_return_pct"]})
    study_plot_df = pd.DataFrame(plot_rows)

    fig_study = px.bar(
        study_plot_df,
        x="Event",
        y="Return (%)",
        color="Window",
        barmode="group",
        title="Comparative Market Returns Across Major Global Events",
        color_discrete_map={
            "Pre-5D Return (%)": "#636EFA",
            "Post-5D Return (%)": "#EF553B",
            "Post-30D Return (%)": "#00CC96"
        }
    )
    fig_study.add_hline(y=0, line_color="black", line_width=1)
    fig_study.update_layout(height=450, xaxis_tickangle=-20)
    st.plotly_chart(fig_study, use_container_width=True)

    # 5. Event Summary Table
    st.subheader("5. Event Summary & Quantitative Metrics Table")
    display_cols = [
        "event_name",
        "effective_trading_date",
        "event_category",
        "sp500_pre_5d_return_pct",
        "sp500_post_5d_return_pct",
        "sp500_post_30d_return_pct",
        "sp500_max_drawdown_30d_pct",
        "vix_at_event",
        "vix_5d_change",
        "vix_max_30d",
        "short_description"
    ]
    renamed_cols = {
        "event_name": "Event",
        "effective_trading_date": "Trading Date (T0)",
        "event_category": "Category",
        "sp500_pre_5d_return_pct": "Pre-5D (%)",
        "sp500_post_5d_return_pct": "Post-5D (%)",
        "sp500_post_30d_return_pct": "Post-30D (%)",
        "sp500_max_drawdown_30d_pct": "Max Drawdown 30D (%)",
        "vix_at_event": "VIX at T0",
        "vix_5d_change": "VIX Δ5D",
        "vix_max_30d": "VIX Max 30D",
        "short_description": "Description"
    }

    styled_table = event_study_df[display_cols].rename(columns=renamed_cols)
    st.dataframe(styled_table, use_container_width=True, hide_index=True)

    # 6. Focused Event Window Drill-Down
    st.markdown("---")
    st.subheader("6. Event Detail Drill-Down (±30 Trading Days)")
    selected_event_name = st.selectbox(
        "Select an event to inspect price & volatility path:",
        options=event_study_df["event_name"].tolist()
    )

    ev_row = event_study_df[event_study_df["event_name"] == selected_event_name].iloc[0]
    t0_date_str = str(ev_row["effective_trading_date"])
    st.info(f"**{ev_row['event_name']}** ({ev_row['event_category']}): {ev_row['short_description']}")

    # Find window around T0
    trading_dates = sp500_df["Date"].dt.strftime("%Y-%m-%d").tolist()
    if t0_date_str in trading_dates:
        t0_idx = trading_dates.index(t0_date_str)
        start_idx = max(0, t0_idx - 30)
        end_idx = min(len(trading_dates) - 1, t0_idx + 30)

        window_sp500 = sp500_df.iloc[start_idx:end_idx + 1].copy()
        # Normalize price to 100 at T0 for standardized comparison
        base_price = sp500_df.loc[t0_idx, "Close"]
        window_sp500["Normalized_Index"] = (window_sp500["Close"] / base_price) * 100.0

        fig_drill = px.line(
            window_sp500,
            x="Date",
            y="Normalized_Index",
            title=f"Normalized S&P 500 Path around {selected_event_name} (T0 = 100)",
            labels={"Normalized_Index": "Indexed (T0 = 100)", "Date": "Date"}
        )
        fig_drill.add_vline(x=pd.to_datetime(t0_date_str), line_dash="dash", line_color="crimson", annotation_text="T0 (Event)")
        fig_drill.add_hline(y=100, line_dash="dot", line_color="gray")
        fig_drill.update_layout(height=400)
        st.plotly_chart(fig_drill, use_container_width=True)

    if show_raw_table:
        st.markdown("---")
        st.subheader("Sample Processed S&P 500 Data (Recent Records)")
        st.dataframe(sp500_df.tail(15), use_container_width=True)


if __name__ == "__main__":
    main()
