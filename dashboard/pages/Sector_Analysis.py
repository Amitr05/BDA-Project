"""
Page 5: Sector Analysis
In-depth sector ETF benchmark performance, intra-sector peer company comparisons,
cross-sector return correlations, and sector event study impact analysis.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.app import (
    RAW_DIR,
    load_sector_etfs_data,
    load_sector_event_study,
    load_events_database,
    load_stock_data,
    load_companies_registry,
    _match_event_subset,
    _format_num,
)
from dashboard.components.header import render_page_header
from dashboard.components.charts import apply_chart_theme, FINANCIAL_COLORS


def render_sector_analysis_page():
    render_page_header(
        title="Sector Analysis & Benchmark ETFs",
        subtitle="Cross-sector performance dynamics, intra-sector company dispersion, 8 SPDR sector benchmarks, and event-study sector impacts.",
        badge="MODULE 5",
        badge_color="#8b5cf6"
    )

    sector_etfs_df = load_sector_etfs_data()
    sector_events_df = load_sector_event_study()
    comp_registry = load_companies_registry()
    events_df = load_events_database()

    if sector_etfs_df is None or sector_etfs_df.empty:
        st.warning("Sector ETF dataset missing in data/processed/etfs/. Run ingestion pipeline first.")
        return

    # 1. Sector ETF Benchmark Performance (8 Select SPDR ETFs)
    st.subheader("1. Sector ETF Benchmark Performance (8 Select SPDR ETFs)")
    st.caption("Benchmark SPDR ETFs: XLK (Technology), XLF (Financials), XLE (Energy), XLV (Health Care), XLI (Industrials), XLY (Consumer Discretionary), XLP (Consumer Staples), XLU (Utilities)")

    etf_pivot = sector_etfs_df.pivot(index="Date", columns="Ticker", values="Close").dropna()
    if not etf_pivot.empty:
        norm_etf = (etf_pivot / etf_pivot.iloc[0]) * 100.0
        col_e1, col_e2 = st.columns([1.2, 0.8])
        with col_e1:
            e_fig = px.line(
                norm_etf.reset_index(),
                x="Date",
                y=[c for c in norm_etf.columns if c != "Date"],
                title="Comparative Sector ETF Cumulative Performance (Rebased to 100)",
                labels={"value": "Normalized Return", "variable": "Sector ETF"}
            )
            apply_chart_theme(e_fig, height=390)
            st.plotly_chart(e_fig, use_container_width=True)

        with col_e2:
            etf_ret_pivot = sector_etfs_df.pivot(index="Date", columns="Ticker", values="daily_return").dropna()
            etf_corr = etf_ret_pivot.corr().round(2)
            etf_corr_fig = px.imshow(
                etf_corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1.0,
                zmax=1.0,
                title="Sector ETF Daily Return Correlation Matrix"
            )
            apply_chart_theme(etf_corr_fig, height=390)
            st.plotly_chart(etf_corr_fig, use_container_width=True)

        # Performance summary table
        etf_stats = []
        for t in [c for c in etf_pivot.columns if c != "Date"]:
            p0 = etf_pivot[t].iloc[0]
            p1 = etf_pivot[t].iloc[-1]
            tot_ret = ((p1 - p0) / p0) * 100.0 if p0 > 0 else 0.0
            daily_ret = etf_ret_pivot[t].dropna() if t in etf_ret_pivot.columns else pd.Series()
            ann_vol = float(daily_ret.std() * (252 ** 0.5) * 100.0) if not daily_ret.empty else np.nan
            sr = (tot_ret / ann_vol) if (ann_vol and ann_vol > 0) else np.nan
            roll_max = etf_pivot[t].cummax()
            dd = ((etf_pivot[t] - roll_max) / roll_max).min() * 100.0

            sector_names = {
                "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care",
                "XLI": "Industrials", "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLU": "Utilities"
            }
            etf_stats.append({
                "ETF Ticker": t,
                "Sector Name": sector_names.get(t, t),
                "Total Return (%)": round(tot_ret, 2),
                "Annualized Volatility (%)": round(ann_vol, 2) if pd.notna(ann_vol) else None,
                "Sharpe Ratio": round(sr, 2) if pd.notna(sr) else None,
                "Max Drawdown (%)": round(dd, 2)
            })

        st.dataframe(pd.DataFrame(etf_stats).sort_values("Total Return (%)", ascending=False), use_container_width=True, hide_index=True)

    st.markdown("---")

    # 2. Intra-Sector Peer Company Comparison
    st.subheader("2. Intra-Sector Peer Company Comparison")
    st.caption("Compare individual constituent equities within any market sector to evaluate relative valuation, Sharpe ratios, and correlation.")

    if comp_registry is not None and not comp_registry.empty:
        all_sectors = sorted([s for s in comp_registry["sector"].dropna().unique().tolist() if s != "Unknown"])
        default_sec = "Information Technology" if "Information Technology" in all_sectors else all_sectors[0]

        sel_sector_peers = st.selectbox("Select Sector for Peer Comparison:", options=all_sectors, index=all_sectors.index(default_sec))
        sector_stocks = comp_registry[comp_registry["sector"] == sel_sector_peers]["ticker"].tolist()

        stocks_dir = RAW_DIR / "stocks"
        avail_sector_stocks = [t for t in sector_stocks if (stocks_dir / f"{t}_raw.csv").exists()] if stocks_dir.exists() else []

        if len(avail_sector_stocks) >= 2:
            default_peers = avail_sector_stocks[:min(4, len(avail_sector_stocks))]
            chosen_peers = st.multiselect("Select Peer Companies to Compare:", options=avail_sector_stocks, default=default_peers)

            if len(chosen_peers) >= 2:
                peer_dfs = {}
                for t in chosen_peers:
                    df_t = load_stock_data(t)
                    if df_t is not None and not df_t.empty:
                        peer_dfs[t] = df_t

                if len(peer_dfs) >= 2:
                    peer_pivot = pd.concat([df.set_index("Date")["Close"].rename(t) for t, df in peer_dfs.items()], axis=1).dropna()
                    norm_peers = (peer_pivot / peer_pivot.iloc[0]) * 100.0

                    col_p1, col_p2 = st.columns([1.2, 0.8])
                    with col_p1:
                        peer_fig = px.line(
                            norm_peers.reset_index(),
                            x="Date",
                            y=chosen_peers,
                            title=f"Intra-Sector Normalized Performance (Base=100) - {sel_sector_peers}",
                            labels={"value": "Indexed Performance", "variable": "Company"}
                        )
                        apply_chart_theme(peer_fig, height=380)
                        st.plotly_chart(peer_fig, use_container_width=True)

                    with col_p2:
                        ret_pivot = pd.concat([df.set_index("Date")["daily_return"].rename(t) for t, df in peer_dfs.items()], axis=1).dropna()
                        corr_peers = ret_pivot.corr().round(2)
                        p_corr_fig = px.imshow(
                            corr_peers,
                            text_auto=True,
                            aspect="auto",
                            color_continuous_scale="RdBu_r",
                            zmin=-1.0,
                            zmax=1.0,
                            title=f"Daily Return Correlation Matrix ({sel_sector_peers})"
                        )
                        apply_chart_theme(p_corr_fig, height=380)
                        st.plotly_chart(p_corr_fig, use_container_width=True)

                    # Quantitative peer table
                    peer_stats = []
                    for t in chosen_peers:
                        sdf = peer_dfs[t]
                        p_0 = sdf.iloc[0]["Close"]
                        p_1 = sdf.iloc[-1]["Close"]
                        t_ret = ((p_1 - p_0) / p_0) * 100.0 if p_0 > 0 else 0.0
                        v_ann = float(sdf["daily_return"].std() * (252 ** 0.5) * 100.0)
                        sr = (t_ret / v_ann) if v_ann > 0 else 0.0
                        roll_max = sdf["Close"].cummax()
                        dd = ((sdf["Close"] - roll_max) / roll_max).min() * 100.0
                        c_name = comp_registry[comp_registry["ticker"] == t]["company_name"].values[0] if len(comp_registry[comp_registry["ticker"] == t]) > 0 else t
                        peer_stats.append({
                            "Ticker": t,
                            "Company": c_name,
                            "Total Return (%)": round(t_ret, 2),
                            "Annual Volatility (%)": round(v_ann, 2),
                            "Sharpe Ratio": round(sr, 2),
                            "Max Drawdown (%)": round(dd, 2)
                        })
                    st.dataframe(pd.DataFrame(peer_stats), use_container_width=True, hide_index=True)
        else:
            st.info(f"Insufficient local stock records for sector '{sel_sector_peers}' to conduct intra-sector peer comparison.")

    st.markdown("---")

    # 3. Sector Event Study Returns
    st.subheader("3. Sector ETF Response to Global Events")
    st.caption("Empirical event-study performance across 8 major sector ETFs around major historical market shocks.")

    if sector_events_df is not None and not sector_events_df.empty:
        ev_names = sector_events_df["event_name"].dropna().unique().tolist()
        sel_study_ev = st.selectbox("Select Historical Event to Inspect Sector Reactions:", options=ev_names)

        sub_sec_ev = sector_events_df[sector_events_df["event_name"] == sel_study_ev]
        if not sub_sec_ev.empty:
            s_melt = sub_sec_ev.melt(
                id_vars=["sector", "ticker"],
                value_vars=["pre_5d_return_pct", "post_5d_return_pct", "post_30d_return_pct"],
                var_name="window",
                value_name="ret_pct"
            )
            s_melt["Window"] = s_melt["window"].map({
                "pre_5d_return_pct": "Pre-5D %",
                "post_5d_return_pct": "Post-5D %",
                "post_30d_return_pct": "Post-30D %"
            })
            sec_ev_fig = px.bar(
                s_melt,
                x="sector",
                y="ret_pct",
                color="Window",
                barmode="group",
                title=f"8 Sector ETFs Performance Around {sel_study_ev}"
            )
            sec_ev_fig.add_hline(y=0, line_color="gray")
            apply_chart_theme(sec_ev_fig, height=380)
            sec_ev_fig.update_layout(xaxis_tickangle=-20)
            st.plotly_chart(sec_ev_fig, use_container_width=True)

            st.dataframe(
                sub_sec_ev[["sector", "ticker", "pre_5d_return_pct", "post_5d_return_pct", "post_30d_return_pct"]].rename(columns={
                    "sector": "Sector", "ticker": "ETF Ticker", "pre_5d_return_pct": "Pre-5D Return %",
                    "post_5d_return_pct": "Post-5D Return %", "post_30d_return_pct": "Post-30D Return %"
                }),
                use_container_width=True,
                hide_index=True
            )


if __name__ == "__main__":
    render_sector_analysis_page()
