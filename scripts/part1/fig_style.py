"""Consistent matplotlib style for all Paper 1 figures.

Usage::

    from fig_style import apply_style, MODEL_COLORS, REGIME_COLORS, OPTION_COLORS, MODEL_SHORT, save_fig
    apply_style()
    fig, ax = plt.subplots(...)
    ...
    save_fig(fig, "fig1_examples", output_dir)
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Colorblind-safe palette ──
MODEL_COLORS: dict[str, str] = {
    "Qwen/Qwen2.5-7B-Instruct": "#0072B2",       # blue
    "meta-llama/Llama-3.1-8B-Instruct": "#E69F00", # orange
    "mistralai/Mistral-7B-Instruct-v0.3": "#009E73", # green
}
MODEL_SHORT: dict[str, str] = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5-7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1-8B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral 7B-v0.3",
}
REGIME_COLORS: dict[str, str] = {
    "Stable-Correct": "#0072B2",    # blue
    "Stable-Wrong": "#D55E00",      # vermillion
    "Unstable": "#999999",          # grey
}
OPTION_COLORS: dict[str, str] = {
    "A": "#0072B2", "B": "#E69F00", "C": "#009E73", "D": "#D55E00",
}
MODEL_LAYERS: dict[str, int] = {
    "Qwen/Qwen2.5-7B-Instruct": 28,
    "meta-llama/Llama-3.1-8B-Instruct": 32,
    "mistralai/Mistral-7B-Instruct-v0.3": 32,
}
EXPECTED_MODELS = list(MODEL_COLORS.keys())


def apply_style() -> None:
    """Apply the Paper 1 matplotlib style globally."""
    plt.rcdefaults()
    params: dict = {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "legend.framealpha": 0.8,
        "legend.edgecolor": "0.8",
        "lines.linewidth": 1.5,
        "lines.markersize": 3.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,
        "axes.linewidth": 0.6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
        "mathtext.fontset": "cm",
    }
    mpl.rcParams.update(params)


def save_fig(fig: plt.Figure, name: str, output_dir: Path) -> Path:
    """Save figure as PDF and return path."""
    path = output_dir / "figures" / f"{name}.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_hash(path: Path) -> str:
    """SHA256 of a figure file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
