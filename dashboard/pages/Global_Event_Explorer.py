"""
Page 3: Global Event Explorer
Browse, filter, and inspect global geopolitical conflicts, elections, macroeconomic shocks,
natural disasters, associated corporate entities, and GDELT news sentiment.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.app import (
    load_events_database,
    load_news_articles,
    _match_event_subset,
)
from dashboard.components.header import render_page_header
from dashboard.components.charts import apply_chart_theme


def render_global_event_explorer_page():
    render_page_header(
        title="Global Event Explorer & News Intelligence",
        subtitle="Historical catalog of geopolitical conflicts, elections, macroeconomic shocks, and natural disasters linked with GDELT news sentiment.",
        badge="MODULE 2",
        badge_color="#10b981"
    )

    events_df = load_events_database()
    if events_df.empty:
        st.warning("No global events found in data/raw/events_expanded.csv or events.csv.")
        return

    # Clean dates and categories
    events_df["event_date_dt"] = pd.to_datetime(events_df["event_date"], errors="coerce")
    if "category" not in events_df.columns and "event_category" in events_df.columns:
        events_df["category"] = events_df["event_category"]

    # 1. Filters Row: Categories, Regions, Date Range, Search
    f1, f2, f3 = st.columns([1.5, 1.2, 1.3])

    with f1:
        # Filter by 7 event categories
        avail_cats = sorted(events_df["category"].dropna().unique().tolist())
        selected_cats = st.multiselect(
            "Filter by Event Category (7 Core Categories):",
            options=avail_cats,
            default=avail_cats,
            key="m2_cat_filter"
        )

    with f2:
        # Filter by geographical region
        regions = ["All Regions"]
        if "region" in events_df.columns:
            regions += sorted(events_df["region"].dropna().unique().tolist())
        selected_region = st.selectbox("Filter by Region:", options=regions, key="m2_reg_filter")

    with f3:
        # Search for specific events
        search_query = st.text_input("Search Events (Name, Keywords, Tickers):", placeholder="e.g. Tariffs, Ukraine, Brexit...", key="m2_search")

    # Date range filter
    min_ev_d = events_df["event_date_dt"].min().date()
    max_ev_d = events_df["event_date_dt"].max().date()
    ev_date_range = st.date_input("Filter by Event Date Range:", value=[min_ev_d, max_ev_d], min_value=min_ev_d, max_value=max_ev_d, key="m2_date_filter")

    # Apply all filters
    filtered_events = events_df.copy()
    if selected_cats:
        filtered_events = filtered_events[filtered_events["category"].isin(selected_cats)]

    if selected_region != "All Regions" and "region" in filtered_events.columns:
        filtered_events = filtered_events[filtered_events["region"] == selected_region]

    if isinstance(ev_date_range, (list, tuple)) and len(ev_date_range) == 2:
        d_start, d_end = pd.to_datetime(ev_date_range[0]), pd.to_datetime(ev_date_range[1])
        filtered_events = filtered_events[(filtered_events["event_date_dt"] >= d_start) & (filtered_events["event_date_dt"] <= d_end)]

    if search_query:
        q = search_query.lower()
        search_mask = (
            filtered_events["event_name"].astype(str).str.lower().str.contains(q) |
            filtered_events.get("description", pd.Series([""] * len(filtered_events))).astype(str).str.lower().str.contains(q) |
            filtered_events.get("tags", pd.Series([""] * len(filtered_events))).astype(str).str.lower().str.contains(q) |
            filtered_events.get("related_companies", pd.Series([""] * len(filtered_events))).astype(str).str.lower().str.contains(q)
        )
        filtered_events = filtered_events[search_mask]

    # Metrics summary
    em1, em2, em3, em4 = st.columns(4)
    em1.metric("Matching Global Events", len(filtered_events))
    em2.metric("Categories Represented", filtered_events["category"].nunique())
    em3.metric("Earliest Event", filtered_events["event_date_dt"].min().strftime("%Y-%m-%d") if not filtered_events.empty else "N/A")
    em4.metric("Latest Event", filtered_events["event_date_dt"].max().strftime("%Y-%m-%d") if not filtered_events.empty else "N/A")

    st.markdown("---")

    # 1. Browse Global Events Table
    st.subheader("1. Global Events Directory")
    if filtered_events.empty:
        st.info("No global events match the current filter criteria. Try expanding the date range or clearing the search query.")
        return

    disp_cols = [c for c in ["event_id", "event_name", "category", "subcategory", "event_date", "region", "severity_score", "related_companies", "related_sectors"] if c in filtered_events.columns]
    st.dataframe(filtered_events[disp_cols].sort_values("event_date", ascending=False), use_container_width=True, hide_index=True)

    # 2. View Event Descriptions & Entity Associations
    st.markdown("---")
    st.subheader("2. Detailed Event Inspector & Associated Entities")

    event_choices = filtered_events["event_name"].tolist()
    sel_event_name = st.selectbox("Select Event to Inspect in Detail:", options=event_choices, key="m2_event_select")
    ev_row = filtered_events[filtered_events["event_name"] == sel_event_name].iloc[0]

    card_col1, card_col2 = st.columns([1.6, 1.4])
    with card_col1:
        st.markdown(f"#### {ev_row['event_name']}")
        st.markdown(f"**Category:** `{ev_row.get('category', 'General')}` | **Subcategory:** `{ev_row.get('subcategory', 'N/A')}`")
        st.markdown(f"**Date:** `{pd.to_datetime(ev_row['event_date']).strftime('%Y-%m-%d')}` | **Region:** `{ev_row.get('region', 'Global')}`")

        sev = ev_row.get("severity_score", np.nan)
        if pd.notna(sev):
            st.markdown(f"**Impact Severity Score:** `{sev} / 5.0`")

        desc = ev_row.get("description", ev_row.get("short_description", "No description available."))
        st.info(f"**Description:**\n\n{desc}")

    with card_col2:
        st.markdown("#### Associated Corporate Entities & Sectors")
        rel_comp = ev_row.get("related_companies", "")
        rel_sec = ev_row.get("related_sectors", "")
        tags = ev_row.get("tags", "")
        source_ref = ev_row.get("source_ref", "")

        st.write(f"**Affected Companies / Tickers:**")
        if pd.notna(rel_comp) and str(rel_comp).strip():
            tickers_list = str(rel_comp).split("|")
            st.markdown(" ".join([f"`{t}`" for t in tickers_list]))
        else:
            st.write("*(Broad market impact / Unspecified)*")

        st.write(f"**Affected Sectors:**")
        if pd.notna(rel_sec) and str(rel_sec).strip():
            sectors_list = str(rel_sec).split("|")
            st.markdown(" ".join([f"`{s}`" for s in sectors_list]))
        else:
            st.write("*(All market sectors)*")

        if pd.notna(tags) and str(tags).strip():
            st.write(f"**Tags:** `{tags}`")
        if pd.notna(source_ref) and str(source_ref).strip():
            st.caption(f"**Source Reference:** {source_ref}")

    # 3. View Related News Articles & Sentiment
    st.markdown("---")
    st.subheader(f"3. GDELT News Coverage & Sentiment Dynamics: {sel_event_name}")
    news_df = load_news_articles()

    if not news_df.empty:
        ev_id = ev_row.get("event_id", "")
        ev_date_str = pd.to_datetime(ev_row["event_date"]).strftime("%Y-%m-%d")
        # Match news by event_id, event_name, or _match_event_subset
        news_match = _match_event_subset(news_df, sel_event_name, ev_date_str)
        if news_match.empty:
            news_match = news_df[
                (news_df["matched_event_id"] == ev_id) |
                (news_df["matched_event_name"].astype(str).str.contains(sel_event_name[:15], case=False, na=False))
            ]

        if not news_match.empty:
            n1, n2, n3, n4 = st.columns(4)
            n1.metric("Articles Analyzed", len(news_match))
            avg_tone = news_match["gdelt_tone"].mean() if "gdelt_tone" in news_match.columns else news_match["tone"].mean()
            n2.metric("Average GDELT Tone", f"{avg_tone:.2f}", delta_color="normal")
            dominant_sent = news_match["sentiment_label"].mode().values[0] if "sentiment_label" in news_match.columns else "Neutral"
            n3.metric("Dominant Sentiment", dominant_sent)
            n4.metric("Sources/Publishers", news_match["domain"].nunique() if "domain" in news_match.columns else "N/A")

            # Timing phase and sentiment breakdown chart
            col_n1, col_n2 = st.columns(2)
            with col_n1:
                if "timing_phase" in news_match.columns:
                    timing_counts = news_match["timing_phase"].value_counts().reset_index()
                    timing_counts.columns = ["Timing Phase", "Articles"]
                    time_fig = px.bar(
                        timing_counts,
                        x="Timing Phase",
                        y="Articles",
                        title="News Distribution Across Event Timing Phases",
                        color="Timing Phase",
                        color_discrete_map={"pre-event": "#636EFA", "day-of": "#EF553B", "post-event": "#00CC96"}
                    )
                    apply_chart_theme(time_fig, height=300)
                    st.plotly_chart(time_fig, use_container_width=True)

            with col_n2:
                if "sentiment_label" in news_match.columns:
                    sent_counts = news_match["sentiment_label"].value_counts().reset_index()
                    sent_counts.columns = ["Sentiment", "Articles"]
                    sent_fig = px.pie(
                        sent_counts,
                        names="Sentiment",
                        values="Articles",
                        title="News Sentiment Label Breakdown",
                        color="Sentiment",
                        color_discrete_map={"Positive": "#2ca02c", "Neutral": "#7f7f7f", "Negative": "#d62728"}
                    )
                    apply_chart_theme(sent_fig, height=300)
                    st.plotly_chart(sent_fig, use_container_width=True)

            # News articles table
            st.write("**Related News Articles Table**")
            news_disp_cols = [c for c in ["published_date", "title", "domain", "sourcecountry", "sentiment_label", "gdelt_tone", "polarity", "timing_phase"] if c in news_match.columns]
            st.dataframe(news_match[news_disp_cols].sort_values("published_date", ascending=True), use_container_width=True, hide_index=True)
        else:
            st.info(f"No specific GDELT news records indexed for '{sel_event_name}' in the sample collection. Showing category news:")
            cat_news = news_df[news_df["event_category"] == ev_row.get("category")].head(5)
            if not cat_news.empty:
                st.dataframe(cat_news[[c for c in ["published_date", "title", "domain", "sentiment_label", "gdelt_tone"] if c in cat_news.columns]], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render_global_event_explorer_page()
