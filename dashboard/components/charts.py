"""
Chart Styling and Layout Components for Financial Visualizations
"""

from typing import Optional, Dict, Any
import plotly.graph_objects as go


FINANCIAL_COLORS = {
    "primary": "#1f77b4",     # Institutional Blue
    "secondary": "#ff7f0e",   # Accent Orange
    "success": "#2ca02c",     # Positive Return Green
    "danger": "#d62728",      # Negative Return Red
    "vix": "#9467bd",         # Volatility Purple
    "benchmark": "#7f7f7f",   # S&P 500 Baseline Gray
    "accent": "#0ea5e9",      # Sky Blue
    "neutral": "#64748b",     # Slate Gray
}


def apply_chart_theme(
    fig: go.Figure,
    height: int = 380,
    title: Optional[str] = None,
    hovermode: str = "x unified",
    show_legend: bool = True,
    margin: Optional[Dict[str, int]] = None
) -> go.Figure:
    """
    Applies unified institutional typography, gridlines, and margins to a Plotly figure.
    """
    layout_update: Dict[str, Any] = {
        "height": height,
        "hovermode": hovermode,
        "showlegend": show_legend,
        "margin": margin or dict(l=40, r=40, t=50, b=40),
        "font": dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"),
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
    }

    if title:
        layout_update["title"] = dict(text=title, font=dict(size=14, weight=600))

    fig.update_layout(**layout_update)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig
