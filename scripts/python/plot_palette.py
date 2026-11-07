"""Shared color palette utilities for competition figures."""
from __future__ import annotations

from typing import Final

from matplotlib.colors import to_rgb, to_hex


RENAISSANCE_BASE: Final[list[str]] = [
    "#3E4A61",  # indigo slate
    "#A77957",  # warm sienna
    "#7A9E7E",  # muted sage
    "#C4A46B",  # aged ochre
    "#8C5A3C",  # burnt umber
    "#B26E63",  # deep rosewood
    "#5E503F",  # walnut brown
    "#6F7D8C",  # steel blue
    "#A1B57D",  # soft moss
    "#9C6F53",  # leather
    "#AF8A6D",  # kiln clay
    "#D4B483",  # wheat gold
]

METRIC_ALIASES: Final[dict[str, str]] = {
    "Net.Benefit": "Net Benefit",
    "Net Benefit (%)": "Net Benefit",
    "Net Benefit %": "Net Benefit",
    "1 - Parsimony": "Parsimony (1 - score)",
    "Parsimony Score": "Parsimony (1 - score)",
    "Parsimony": "Parsimony (1 - score)",
    "Inference Speed": "Inference Time",
    "Inference": "Inference Time",
    "F1 Opt": "F1",
    "F1 Score": "F1",
    "Youden Index": "Youden",
}

METRIC_COLOR_MAP: Final[dict[str, str]] = {
    "F1": "#1F78B4",
    "AUPRC": "#E31A1C",
    "Net Benefit": "#33A02C",
    "ECE": "#FF7F00",
    "AUC": "#6A3D9A",
    "Sensitivity": "#B15928",
    "Specificity": "#A6CEE3",
    "Precision": "#B2DF8A",
    "MCC": "#FB9A99",
    "Parsimony (1 - score)": "#CAB2D6",
    "Inference Time": "#FDBF6F",
    "Youden": "#C49000",
    "Default": "#5F6A7D",
}

PHASE_COLOR_MAP: Final[dict[str, str]] = {
    "Preliminary": "#9CA3AF",
    "Final": "#0F4C81",
}


def canonical_metric_name(name: str) -> str:
    """Return a consistent metric key for color lookup."""
    if not isinstance(name, str):
        return str(name)
    stripped = name.strip()
    return METRIC_ALIASES.get(stripped, stripped)


def get_metric_color(name: str) -> str:
    """Fetch the shared color for a metric or fall back to the default shade."""
    canonical = canonical_metric_name(name)
    return METRIC_COLOR_MAP.get(canonical, METRIC_COLOR_MAP["Default"])


BASE_PALETTE: Final[list[str]] = list(dict.fromkeys(color for color in METRIC_COLOR_MAP.values() if color))


def lighten_color(color: str, amount: float = 0.3) -> str:
    """Blend the given color toward white by ``amount`` (0-1 range)."""
    r, g, b = to_rgb(color)
    r = min(1.0, r + (1.0 - r) * amount)
    g = min(1.0, g + (1.0 - g) * amount)
    b = min(1.0, b + (1.0 - b) * amount)
    return to_hex((r, g, b))


def darken_color(color: str, amount: float = 0.25) -> str:
    """Blend the given color toward black by ``amount`` (0-1 range)."""
    r, g, b = to_rgb(color)
    r = max(0.0, r * (1.0 - amount))
    g = max(0.0, g * (1.0 - amount))
    b = max(0.0, b * (1.0 - amount))
    return to_hex((r, g, b))
