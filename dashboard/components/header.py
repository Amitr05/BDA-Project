"""
Header Component for Multi-Page BDA Financial Analytics Dashboard
"""

from typing import Optional
import streamlit as st


def render_page_header(
    title: str,
    subtitle: str,
    badge: Optional[str] = None,
    badge_color: str = "#2563eb"
) -> None:
    """
    Renders a unified, professional header across dashboard pages.
    
    Args:
        title: Main page heading
        subtitle: Descriptive methodological or functional context
        badge: Optional pill badge text (e.g. 'MODULE 1', 'ECONOMETRIC ENGINE')
        badge_color: Background color for the pill badge
    """
    if badge:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                <span style="
                    background-color: {badge_color}20;
                    color: {badge_color};
                    border: 1px solid {badge_color}40;
                    padding: 3px 10px;
                    border-radius: 9999px;
                    font-size: 0.75rem;
                    font-weight: 700;
                    letter-spacing: 0.05em;
                    text-transform: uppercase;
                ">{badge}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader(title)
    st.caption(subtitle)
    st.markdown("---")
