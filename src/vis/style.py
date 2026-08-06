"""Shared matplotlib style for experiment plots.

Palette follows a validated colorblind-safe categorical order (fixed slots,
never cycled beyond 8 -- extra runs fold into gray "Other"). Chrome (grid,
axes, text) is recessive so the data marks carry the figure.
"""

from __future__ import annotations
import matplotlib as mpl

# Categorical palette (light mode), fixed slot order -- do not re-sort.
CATEGORICAL = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]
OTHER_GRAY = "#898781"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def series_color(i: int) -> str:
    """Fixed-order color for the i-th series; series beyond 8 fold into gray."""
    return CATEGORICAL[i] if i < len(CATEGORICAL) else OTHER_GRAY


def apply_style():
    """Apply the shared rcParams. Call once before creating figures."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "figure.facecolor": SURFACE,
        "figure.titlesize": 14,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.titlesize": 11,
        "axes.titlecolor": INK_PRIMARY,
        "axes.labelsize": 9,
        "axes.labelcolor": INK_SECONDARY,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "font.family": "sans-serif",
        "text.color": INK_PRIMARY,
    })


def legend_if_multi(ax, n_series: int, **kwargs):
    """A legend is always present for >= 2 series; a single series needs none
    (the title names it)."""
    if n_series >= 2:
        ax.legend(**kwargs)
