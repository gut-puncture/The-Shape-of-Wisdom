"""Shared NeurIPS-compatible styling constants for paper figures."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

TRAJ = {
    "stable_correct": "#2ca02c",
    "stable_wrong": "#d62728",
    "unstable_correct": "#4a90d9",
    "unstable_wrong": "#e67e22",
}

TRAJ_LABELS = {
    "stable_correct": "Stable-Correct",
    "stable_wrong": "Stable-Wrong",
    "unstable_correct": "Unstable-Correct",
    "unstable_wrong": "Unstable-Wrong",
}

TRAJ_ORDER = ["stable_correct", "stable_wrong", "unstable_correct", "unstable_wrong"]

COMP = {"attention": "#2563eb", "mlp": "#dc2626"}
COMP_LABELS = {"attention": "Attention", "mlp": "MLP"}

SPAN = {"evidence": "#2ca02c", "distractor": "#d62728", "neutral": "#9ca3af"}
SPAN_LABELS = {"evidence": "Evidence", "distractor": "Distractor", "neutral": "Neutral"}

NEUTRAL = "#4b5563"

# ---------------------------------------------------------------------------
# Layout constants (NeurIPS two-column)
# ---------------------------------------------------------------------------

COL_WIDTH = 3.25       # single-column figure width (inches)
TEXT_WIDTH = 6.75      # full-width figure width (inches)
FONT_TITLE = 9
FONT_AXIS = 8
FONT_TICK = 7
FONT_LEGEND = 6.5
FONT_ANNOT = 6.5
LW_DATA = 1.8
LW_REF = 0.6

# ---------------------------------------------------------------------------
# Matplotlib configuration
# ---------------------------------------------------------------------------

def configure_matplotlib() -> None:
    """Set global rcParams for NeurIPS-quality vector figures."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": FONT_AXIS,
        "axes.titlesize": FONT_TITLE,
        "axes.labelsize": FONT_AXIS,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,       # TrueType fonts in PDF
        "ps.fonttype": 42,
        "lines.linewidth": LW_DATA,
    })


def remove_top_right_spines(ax: plt.Axes) -> None:
    """Remove top and right spines from an axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_zero_line(ax: plt.Axes) -> None:
    """Add a subtle horizontal zero-reference line."""
    ax.axhline(0, color=NEUTRAL, lw=LW_REF, zorder=0)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    """Add (a), (b), etc. label to upper-left corner."""
    ax.text(-0.08, 1.06, f"({label})", transform=ax.transAxes,
            fontsize=FONT_TITLE, fontweight="bold", va="top", ha="right")


def bootstrap_ci(
    values: np.ndarray, *, n_boot: int = 2000, ci: float = 0.95, seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) via bootstrap resampling."""
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0.0, 0.0, 0.0
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        means[i] = sample.mean()
    alpha = (1.0 - ci) / 2.0
    lo, hi = float(np.percentile(means, 100 * alpha)), float(np.percentile(means, 100 * (1 - alpha)))
    return float(vals.mean()), lo, hi


def depth_ticks() -> tuple[list[float], list[str]]:
    """Standard depth axis ticks."""
    vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    labels = ["0", "0.25", "0.5", "0.75", "1"]
    return vals, labels
