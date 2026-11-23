"""
Metrics Formatting and KPI Presentation Components
"""

from typing import Any
import numpy as np
import pandas as pd


def format_num(val: Any, suffix: str = "", digits: int = 2) -> str:
    """
    Format numeric values safely for UI presentation.
    Handles None, NaN, strings, and standard floats/ints.
    """
    if val is None or pd.isna(val):
        return "N/A"
    if isinstance(val, (int, float, np.floating, np.integer)):
        if np.isnan(val) or np.isinf(val):
            return "N/A"
        formatted = f"{val:,.{digits}f}"
        return f"{formatted}{suffix}"
    return str(val)


# Backward-compatible alias
_format_num = format_num
