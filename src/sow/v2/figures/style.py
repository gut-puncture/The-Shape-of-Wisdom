"""Shared styling and reusable plotting helpers for paper figures.

Premium visual design system for NeurIPS-quality figure output.
"""
from __future__ import annotations

from collections.abc import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# ---------------------------------------------------------------------------
# Color palettes — refined, deeper, publication-grade
# ---------------------------------------------------------------------------

TRAJ = {
    "stable_correct": "#0a6abf",   # deep cerulean – confident knowledge
    "stable_wrong": "#c93545",     # crimson rose – committed error
    "unstable_correct": "#139e8c", # deep teal – recovery / resilience
    "unstable_wrong": "#d4882e",   # burnt amber – volatility / caution
}

TRAJ_ALPHA = {k: v + "22" for k, v in TRAJ.items()}  # 14% alpha hex

TRAJ_LABELS = {
    "stable_correct": "Stable-Correct",
    "stable_wrong": "Stable-Wrong",
    "unstable_correct": "Unstable-Correct",
    "unstable_wrong": "Unstable-Wrong",
}

TRAJ_ORDER = ["stable_correct", "stable_wrong", "unstable_correct", "unstable_wrong"]

COMP = {"attention": "#1a5fa0", "mlp": "#c43a4b"}
COMP_LABELS = {"attention": "Attention", "mlp": "MLP"}

SPAN = {"evidence": "#17876c", "distractor": "#c43a4b", "neutral": "#7b8794"}
SPAN_LABELS = {"evidence": "Evidence", "distractor": "Distractor", "neutral": "Neutral"}

NEUTRAL = "#4a5568"

FILL = {
    "correct_bg": "#ddeaf7",
    "wrong_bg": "#fce4e4",
    "tail_bg": "#fef3cd",
    "card_bg": "#fafbfe",
}

# ---------------------------------------------------------------------------
# Layout constants (NeurIPS two-column)
# ---------------------------------------------------------------------------

COL_WIDTH = 3.25       # single-column figure width (inches)
TEXT_WIDTH = 6.75      # full-width figure width (inches)
FONT_TITLE = 10
FONT_AXIS = 8
FONT_TICK = 7
FONT_LEGEND = 6.8
FONT_ANNOT = 6.6
LW_DATA = 1.8
LW_REF = 0.7

# ---------------------------------------------------------------------------
# Matplotlib configuration — premium typography and styling
# ---------------------------------------------------------------------------

def configure_matplotlib() -> None:
    """Set global rcParams for polished, premium two-column vector figures."""
    # Try CMU Sans Serif for LaTeX-matching typography, fallback to DejaVu Sans
    preferred_fonts = ["CMU Sans Serif", "DejaVu Sans"]
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": preferred_fonts,
        "font.size": FONT_AXIS,
        "axes.titlesize": FONT_TITLE,
        "axes.titleweight": "semibold",
        "axes.labelsize": FONT_AXIS,
        "axes.labelweight": "medium",
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.12,
        "grid.linewidth": 0.35,
        "grid.linestyle": ":",
        "grid.color": "#b0b8c4",
        "axes.axisbelow": True,
        "axes.facecolor": "white",
        "axes.edgecolor": "#5a6577",
        "axes.linewidth": 0.6,
        "figure.facecolor": "white",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": None,
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.linewidth": LW_DATA,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.color": "#5a6577",
        "ytick.color": "#5a6577",
    })


def style_card(ax: plt.Axes) -> None:
    """Apply clean, minimal panel style — pure white, no background fill."""
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.08, linewidth=0.3, linestyle=":")
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#5a6577")
        ax.spines[spine].set_linewidth(0.6)
    remove_top_right_spines(ax)


def shared_legend(
    fig: plt.Figure,
    handles: Iterable,
    labels: Iterable[str],
    *,
    ncol: int = 3,
    loc: str = "upper center",
    bbox_to_anchor: tuple[float, float] = (0.5, 1.01),
) -> None:
    """Place a shared figure-level legend in a stable position."""
    fig.legend(
        list(handles),
        list(labels),
        ncol=ncol,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        frameon=False,
        fontsize=FONT_LEGEND,
        handlelength=2.2,
        columnspacing=1.3,
    )


def shade_threshold_region(
    ax: plt.Axes,
    *,
    x_from: float | None = None,
    x_to: float | None = None,
    y_from: float | None = None,
    y_to: float | None = None,
    color: str = FILL["tail_bg"],
    alpha: float = 0.32,
    zorder: float = 0,
) -> None:
    """Shade threshold regions using axis data coordinates."""
    if x_from is not None and x_to is not None:
        ax.axvspan(x_from, x_to, color=color, alpha=alpha, zorder=zorder)
    if y_from is not None and y_to is not None:
        ax.axhspan(y_from, y_to, color=color, alpha=alpha, zorder=zorder)


def annot_arrow(
    ax: plt.Axes,
    *,
    text: str,
    xy: tuple[float, float],
    xytext: tuple[float, float],
    color: str = NEUTRAL,
    fs: float = FONT_ANNOT,
) -> None:
    """Add a consistently styled annotation arrow."""
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        textcoords="data",
        fontsize=fs,
        color=color,
        ha="left",
        va="center",
        arrowprops={
            "arrowstyle": "->",
            "lw": 0.9,
            "color": color,
            "shrinkA": 0,
            "shrinkB": 0,
            "alpha": 0.95,
        },
        annotation_clip=True,
    )


def cov_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_std: float = 1.8,
    facecolor: str = "none",
    edgecolor: str = NEUTRAL,
    alpha: float = 0.18,
    lw: float = 1.0,
) -> Ellipse | None:
    """Return a covariance ellipse patch for x/y points, or None if ill-defined."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if xx.size < 3:
        return None
    cov = np.cov(np.vstack([xx, yy]))
    if cov.shape != (2, 2):
        return None
    vals, vecs = np.linalg.eigh(cov)
    if np.any(vals <= 0):
        return None
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    width, height = 2.0 * n_std * np.sqrt(vals)
    center = (float(np.mean(xx)), float(np.mean(yy)))
    return Ellipse(
        xy=center,
        width=float(width),
        height=float(height),
        angle=theta,
        facecolor=facecolor,
        edgecolor=edgecolor,
        lw=lw,
        alpha=alpha,
    )


def remove_top_right_spines(ax: plt.Axes) -> None:
    """Remove top and right spines from an axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_zero_line(ax: plt.Axes) -> None:
    """Add a horizontal zero-reference line."""
    ax.axhline(0, color=NEUTRAL, lw=LW_REF, zorder=0, alpha=0.6)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    """Panel labels are encoded in captions; keep plot area clean."""
    _ = (ax, label)


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
    lo = float(np.percentile(means, 100 * alpha))
    hi = float(np.percentile(means, 100 * (1 - alpha)))
    return float(vals.mean()), lo, hi


def depth_ticks() -> tuple[list[float], list[str]]:
    """Standard depth-axis ticks."""
    vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    labels = ["0", "0.25", "0.5", "0.75", "1"]
    return vals, labels


# ---------------------------------------------------------------------------
# Premium figure utilities — glow, gradient, colorbar
# ---------------------------------------------------------------------------

def glow_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    lw: float = 2.0,
    glow_alpha: float = 0.15,
    glow_width: float = 5.0,
    n_glow: int = 3,
    zorder: float = 3,
    **kwargs,
) -> plt.Line2D:
    """Plot a line with a subtle luminous glow halo underneath."""
    for i in range(n_glow, 0, -1):
        frac = i / n_glow
        ax.plot(
            x, y,
            color=color,
            lw=lw + glow_width * frac,
            alpha=glow_alpha * frac * 0.5,
            zorder=zorder - 0.5,
            solid_capstyle="round",
        )
    line, = ax.plot(x, y, color=color, lw=lw, zorder=zorder, solid_capstyle="round", **kwargs)
    return line


def gradient_fill(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    alpha_max: float = 0.25,
    n_steps: int = 6,
    reference: float = 0.0,
    zorder: float = 1,
) -> None:
    """Fill between a line and reference with vertically-graded alpha."""
    y_arr = np.asarray(y, dtype=np.float64)
    for i in range(n_steps):
        frac_lo = i / n_steps
        frac_hi = (i + 1) / n_steps
        y_lo = reference + (y_arr - reference) * frac_lo
        y_hi = reference + (y_arr - reference) * frac_hi
        alpha = alpha_max * (1.0 - frac_hi * 0.7)
        ax.fill_between(
            x, y_lo, y_hi,
            color=color, alpha=alpha, zorder=zorder,
            linewidth=0, edgecolor="none",
        )
