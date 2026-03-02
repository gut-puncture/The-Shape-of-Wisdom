"""Global matplotlib style and colour palette for Paper 1 figures.

Usage::

    from plot_style import apply_style, MODEL_COLORS, REGIME_COLORS, OPTION_COLORS
    apply_style()
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
MODEL_COLORS: dict[str, str] = {
    "Qwen/Qwen2.5-7B-Instruct": "#1f77b4",
    "meta-llama/Llama-3.1-8B-Instruct": "#ff7f0e",
    "mistralai/Mistral-7B-Instruct-v0.3": "#2ca02c",
}

MODEL_SHORT: dict[str, str] = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5-7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1-8B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral 7B-v0.3",
}

REGIME_COLORS: dict[str, str] = {
    "Stable-Correct": "#1f77b4",
    "Stable-Wrong": "#d62728",
    "Unstable": "#7f7f7f",
}

OPTION_COLORS: dict[str, str] = {
    "A": "#1f77b4",
    "B": "#ff7f0e",
    "C": "#2ca02c",
    "D": "#d62728",
}

MODEL_LAYERS: dict[str, int] = {
    "Qwen/Qwen2.5-7B-Instruct": 28,
    "meta-llama/Llama-3.1-8B-Instruct": 32,
    "mistralai/Mistral-7B-Instruct-v0.3": 32,
}


def apply_style() -> None:
    """Apply the Paper 1 matplotlib style globally."""
    plt.rcdefaults()
    params: dict = {
        # Font
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        # Lines
        "lines.linewidth": 1.8,
        "lines.markersize": 4,
        # Axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,
        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # PDF
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Text
        "text.usetex": False,
        "mathtext.fontset": "dejavuserif",
    }
    mpl.rcParams.update(params)
