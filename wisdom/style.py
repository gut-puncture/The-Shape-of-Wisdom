"""Figure style: white ground, vivid hues for answers, soft grey floors and shadows under 3-D clouds."""

import colorsys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

INK = "#1d1d1f"
MUTED = "#86868b"
FLOOR = "#ececee"
SHADOW = "#c7c7cc"
ACCENT = "#d0021b"   # a clear red for the headline series
COLW = 3.25          # one column of a two-column page, inches
PAGEW = 6.75         # full page width, inches


def setup():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.4, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def hues(n, s=0.78, v=0.86, start=0.0):
    """n evenly spaced vivid colours around the hue circle."""
    return [colorsys.hsv_to_rgb((start + i / n) % 1.0, s, v) for i in range(n)]


def floor_cloud(ax, P, colors, size=4, alpha=0.9, pad=0.15):
    """A Goodfire-style 3-D cloud: points above a light square floor, each casting a faint shadow."""
    P = np.asarray(P, float)
    lo, hi = P.min(0), P.max(0)
    span = (hi - lo).max() * (1 + pad)
    mid = (hi + lo) / 2
    x0, x1 = mid[0] - span / 2, mid[0] + span / 2
    y0, y1 = mid[1] - span / 2, mid[1] + span / 2
    z0 = lo[2] - 0.08 * span
    ax.plot_surface(np.array([[x0, x1], [x0, x1]]), np.array([[y0, y0], [y1, y1]]), np.full((2, 2), z0),
                    color=FLOOR, alpha=0.9, linewidth=0, antialiased=True, shade=False, zorder=0)
    ax.scatter(P[:, 0], P[:, 1], np.full(len(P), z0), s=size * 0.7, c=SHADOW, alpha=0.5, linewidths=0, depthshade=False)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=size, c=colors, alpha=alpha, linewidths=0, depthshade=False)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z0 + span)
    ax.view_init(elev=24, azim=-48)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 0.8))


def clean(ax, grid=True):
    if grid:
        ax.grid(True, axis="y", color="#e5e5ea", linewidth=0.5)
        ax.set_axisbelow(True)
    return ax


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
