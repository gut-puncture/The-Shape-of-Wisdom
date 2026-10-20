"""Paper figures, from recorded states (Figure 1) and results/summary.json (all others).

    python -m wisdom.figures --run-dir runs/main --results results --out paper/figures

Every figure is drawn at its printed size (one column = 3.3 in, full page = 6.95 in), in the paper's
serif type, with axes positioned explicitly so that titles, labels and legends never collide.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from wisdom import style
from wisdom.analysis import Whitener, split_idx
from wisdom.places import load_stage

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["llama8b", "qwen7b", "gemma9b", "llama1b"]
NAMES = {"llama8b": "Llama-3.1-8B", "qwen7b": "Qwen2.5-7B", "gemma9b": "Gemma-2-9B", "llama1b": "Llama-3.2-1B"}
COLORS = {"llama8b": "#c0392b", "qwen7b": "#2471a3", "gemma9b": "#1e8449", "llama1b": "#d68910"}
DEPTH = "Depth (fraction of layers)"


def lda_3d(Ztr, ytr, k=3):
    """Directions that best separate the training questions' answers (Fisher LDA, shrunk)."""
    classes = sorted(set(ytr))
    mu = Ztr.mean(0)
    Sw = np.zeros((Ztr.shape[1],) * 2)
    Sb = np.zeros_like(Sw)
    for c in classes:
        X = Ztr[[i for i, y in enumerate(ytr) if y == c]]
        m = X.mean(0)
        Sw += (X - m).T @ (X - m)
        Sb += len(X) * np.outer(m - mu, m - mu)
    Sw = Sw / len(Ztr) + 0.1 * np.eye(len(Sw)) * np.trace(Sw) / len(Ztr) / len(Sw)
    vals, vecs = np.linalg.eig(np.linalg.solve(Sw, Sb / len(Ztr)))
    return vecs[:, np.argsort(-vals.real)[:k]].real


def axes_grid(fig, rows, cols, left, right, bottom, top, wgap, hgap, **kw):
    """Axes at exact positions (fractions of the figure), so nothing depends on automatic layout."""
    w = (right - left - wgap * (cols - 1)) / cols
    h = (top - bottom - hgap * (rows - 1)) / rows
    return [[fig.add_axes([left + c * (w + wgap), top - (r + 1) * h - r * hgap, w, h], **kw) for c in range(cols)]
            for r in range(rows)]


def hero(run_dir, model, prompts, answers, out, onset):
    """Figure 1: held-out questions whose answers are countries, at six depths of Llama-3.1-8B."""
    recs, S = load_stage(run_dir / model / "ff", prompts, answers)
    keep = [i for i, r in enumerate(recs) if r["answer_type"] == "country" and r.get("given_qid")]
    recs, S = [recs[i] for i in keep], S[keep]
    tr, te = split_idx(recs)
    counts = {}
    for i in tr:
        counts[recs[i]["given_qid"]] = counts.get(recs[i]["given_qid"], 0) + 1
    top = [q for q, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:12]]
    tr = [i for i in tr if recs[i]["given_qid"] in top]
    te = [i for i in te if recs[i]["given_qid"] in top]
    colors = dict(zip(top, style.DISTINCT))
    L = S.shape[1] - 1
    layers = sorted({1, onset - 1, onset, onset + 1, min(L - 1, onset + 6), L})
    fig = plt.figure(figsize=(style.PAGEW, 4.75))
    grid = axes_grid(fig, 2, 3, 0.0, 1.0, 0.11, 0.99, 0.0, 0.04, projection="3d")
    for ax, layer in zip([a for row in grid for a in row], layers):
        W = Whitener(S[tr, layer])
        P = W(S[te, layer]) @ lda_3d(W(S[tr, layer]), [recs[i]["given_qid"] for i in tr])
        P = P - np.median(P, 0)
        P = np.clip(P / np.percentile(np.abs(P), 95), -1.6, 1.6)  # robust scale: far knots cannot squash the rest
        style.floor_cloud(ax, P, [colors[recs[i]["given_qid"]] for i in te], size=5.5)
        ax.text2D(0.5, 0.93, f"Layer {layer} of {L}", transform=ax.transAxes, ha="center", fontsize=9.5, color=style.INK)
    handles = [Line2D([], [], marker="o", ls="", ms=5, color=colors[q], label=answers[q]["label"]) for q in top]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=8.5,
               handletextpad=0.25, columnspacing=1.6, frameon=False)
    style.save(fig, out / "fig_hero.pdf")


def band(ax, rows, key, color, label, lw=1.7, ls="-"):
    rs = [r for r in rows if r.get(key) and r[key].get("mean") is not None]
    x = [r["depth"] for r in rs]
    ax.fill_between(x, [r[key]["lo"] for r in rs], [r[key]["hi"] for r in rs], color=color, alpha=0.15, linewidth=0)
    ax.plot(x, [r[key]["mean"] for r in rs], color=color, lw=lw, ls=ls, label=label)


def finish_depth_axis(ax, ylabel=None, ylim=(0, 1.0)):
    ax.set_xlim(0, 1.0)
    ax.set_ylim(*ylim)
    ax.set_xlabel(DEPTH)
    if ylabel:
        ax.set_ylabel(ylabel)


def onset(summary, out):
    """Figure 2: readability (with controls) and separation, by depth, for all four models."""
    fig = plt.figure(figsize=(style.PAGEW, 2.45))
    (a, b), = axes_grid(fig, 1, 2, 0.075, 0.99, 0.29, 0.9, 0.09, 0)
    for m in MODELS:
        band(a, summary[m]["depth"], "acc", COLORS[m], NAMES[m])
    a.legend(loc="upper left", handlelength=1.8)
    controls = [Line2D([], [], color="#a6a6a6", lw=1.5, ls=":", label="Shuffled answers (Llama-3.1-8B)"),
                Line2D([], [], color=style.INK, lw=0.9, ls="-.", label="Question text only (Llama-3.1-8B)")]
    rows = summary["llama8b"]["depth"]
    a.plot([r["depth"] for r in rows], [r["acc_shuffled"] for r in rows], color="#a6a6a6", lw=1.5, ls=":")
    a.axhline(summary["llama8b"]["text_baseline"]["mean"], color=style.INK, lw=0.9, ls="-.")
    finish_depth_axis(a, "Readability")
    style.title(a, "A", "Readability")
    fig.legend(handles=controls, loc="lower center", bbox_to_anchor=(0.31, 0.0), ncol=2, columnspacing=1.0,
               handlelength=1.8)
    for m in MODELS:
        rs = [r for r in summary[m]["depth"] if r.get("spread")]
        b.plot([r["depth"] for r in rs], [1 / r["spread"] for r in rs], color=COLORS[m], label=NAMES[m])
    b.set_yscale("log")
    b.set_yticks([0.03, 0.1, 0.3, 1], ["0.03", "0.1", "0.3", "1"])
    b.minorticks_off()
    finish_depth_axis(b, "Separation", ylim=(0.025, 1.3))
    style.title(b, "B", "Separation, without whitening")
    style.save(fig, out / "fig_onset.pdf")


def transfer(summary, out):
    """Figure 3: readability of Llama-3.1-8B under the transfer tests."""
    fig = plt.figure(figsize=(style.COLW, 2.35))
    ax = fig.add_axes([0.16, 0.19, 0.81, 0.78])
    rows = summary["llama8b"]["depth"]
    for key, color, label in [("acc", COLORS["llama8b"], "Standard"), ("acc_cross_relation", "#7d3c98", "Other relations"),
                              ("acc_cross_wording", "#d35400", "Other wordings"),
                              ("acc_no_answer_words", style.INK, "Answer words removed")]:
        band(ax, rows, key, color, label, lw=1.5)
    finish_depth_axis(ax, "Readability")
    ax.legend(loc="center right", bbox_to_anchor=(1.03, 0.45), handlelength=1.5, labelspacing=0.35)
    style.save(fig, out / "fig_transfer.pdf")


def steering(summary, out):
    """Figure 4: moving states toward answer centroids, by the depth of the move."""
    def curve(rows, target, cond, key):
        agg = {}
        for r in rows:
            if r["target"] == target and r["condition"] == cond and r["scale"] == 1.0:
                h, n = agg.get(r["depth"], (0, 0))
                agg[r["depth"]] = (h + r[key] * r["n"], n + r["n"])
        xs = sorted(agg)
        return xs, [agg[x][0] / agg[x][1] for x in xs]

    fig = plt.figure(figsize=(style.PAGEW, 2.4))
    (a, b), = axes_grid(fig, 1, 2, 0.075, 0.99, 0.29, 0.9, 0.09, 0)
    for m in MODELS:
        rows = summary[m]["steering"]
        for ax, target, cond in [(a, "wrong_placed", "gold"), (b, "right", "other")]:
            xs, ys = curve(rows, target, cond, "to_gold" if cond == "gold" else "to_toward")
            ax.plot(xs, ys, color=COLORS[m], marker="o", ms=3.2, label=NAMES[m])
            rx, ry = curve(rows, target, "random", "to_gold" if cond == "gold" else "to_toward")
            ax.plot(rx, ry, color=COLORS[m], lw=1.1, ls=":")
            ax.plot([summary[m]["onset_depth"]] * 2, [-0.08, -0.025], color=COLORS[m], lw=2.0, clip_on=False,
                    solid_capstyle="butt")
    for ax, letter, text in [(a, "A", "Wrong answers moved to the right answer's centroid"),
                             (b, "B", "Right answers moved to another answer's centroid")]:
        finish_depth_axis(ax, None, ylim=(-0.08, 1.02))
        ax.set_xlabel("Depth of the move (fraction of layers)")
        ax.set_xlim(0.15, 0.95)
        style.title(ax, letter, text)
    a.set_ylabel("Gives the target answer")
    handles = [Line2D([], [], color=COLORS[m], marker="o", ms=3.2, label=NAMES[m]) for m in MODELS]
    handles.append(Line2D([], [], color=style.MUTED, lw=1.1, ls=":", label="Random move, same length"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.0), ncol=5, columnspacing=1.4, handlelength=2.2)
    style.save(fig, out / "fig_steering.pdf")


def recognition(summary, out):
    """Figure 5: among wrong answers, how well one number separates near misses from gaps."""
    fig = plt.figure(figsize=(style.COLW, 2.45))
    ax = fig.add_axes([0.16, 0.27, 0.82, 0.7])
    keys = [("auc_rank", style.ACCENT, "Centroid rank"), ("auc_output_rank", "#2c2c2c", "Output rank"),
            ("auc_conf_mean", "#a6a6a6", "Confidence")]
    w = 0.26
    for k, (key, c, lab) in enumerate(keys):
        vals = np.array([summary[m]["errors_beyond_confidence"][key] for m in MODELS])
        ax.bar(np.arange(len(MODELS)) + (k - 1) * w, vals - 0.5, bottom=0.5, width=w, color=c, label=lab)
        for x, v in zip(np.arange(len(MODELS)) + (k - 1) * w, vals):
            if v - 0.5 < 0.02:  # a bar at chance has no height; say its value instead
                ax.text(x, 0.505, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=style.MUTED)
    ax.axhline(0.5, color=style.AXIS, lw=0.8)
    ax.set_xticks(range(len(MODELS)), ["\n".join(NAMES[m].rsplit("-", 1)) for m in MODELS])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0.5, 0.9)
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("AUC")
    ax.legend(loc="upper left", ncol=3, columnspacing=1.0, handlelength=1.2)
    style.save(fig, out / "fig_recognition.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(ROOT / "runs" / "main"))
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "paper" / "figures"))
    args = ap.parse_args()
    style.setup()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((Path(args.results) / "summary.json").read_text())
    prompts = [json.loads(l) for l in open(ROOT / "dataset" / "prompts.jsonl")]
    answers = json.load(open(ROOT / "dataset" / "answers.json"))
    hero(Path(args.run_dir), "llama8b", prompts, answers, out, summary["llama8b"]["onset_layer"])
    onset(summary, out)
    transfer(summary, out)
    steering(summary, out)
    recognition(summary, out)


if __name__ == "__main__":
    main()
