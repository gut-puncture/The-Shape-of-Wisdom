"""Paper figures, from recorded states (hero clouds) and results/summary.json (everything else).

    python -m wisdom.figures --run-dir runs/main --results results --out paper/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wisdom import style
from wisdom.analysis import Whitener, split_idx
from wisdom.places import load_stage

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAMES = {"llama8b": "Llama-3.1-8B", "qwen7b": "Qwen2.5-7B", "gemma9b": "Gemma-2-9B", "llama1b": "Llama-3.2-1B",
               "llama8b_untrained": "Llama-3.1-8B, untrained"}
MODEL_COLORS = {"llama8b": "#d0021b", "qwen7b": "#1f6feb", "gemma9b": "#2da44e", "llama1b": "#bf8700"}


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
    order = np.argsort(-vals.real)[:k]
    return vecs[:, order].real


def hero(run_dir, model, prompts, answers, out, onset=None, answer_type="country", n_panels=6):
    """Test-subject questions at evenly spaced depths, in the three directions that best separate the
    training subjects' answers at that depth (fitted without any test question). Colour = the answer
    the model gave."""
    recs, S = load_stage(run_dir / model / "ff", prompts, answers)
    keep = [i for i, r in enumerate(recs) if r["answer_type"] == answer_type and r.get("given_qid")]
    recs = [recs[i] for i in keep]
    S = S[keep]
    tr, te = split_idx(recs)
    counts = {}
    for i in tr:
        counts[recs[i]["given_qid"]] = counts.get(recs[i]["given_qid"], 0) + 1
    top = [q for q, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:12]]
    tr = [i for i in tr if recs[i]["given_qid"] in top]
    te = [i for i in te if recs[i]["given_qid"] in top]
    colors = dict(zip(top, style.DISTINCT))
    L = S.shape[1] - 1
    o = onset or round(0.5 * L)
    layers = sorted(set([1, max(2, o - 3), o - 1, o + 1, min(L - 1, o + 6), L]))
    fig = plt.figure(figsize=(style.PAGEW, 1.75))
    for j, layer in enumerate(layers):
        W = Whitener(S[tr, layer])
        Ztr, Zte = W(S[tr, layer]), W(S[te, layer])
        V = lda_3d(Ztr, [recs[i]["given_qid"] for i in tr])
        P = Zte @ V
        P = P - np.median(P, 0)
        P = np.clip(P / np.percentile(np.abs(P), 95), -1.6, 1.6)  # robust scale: a few far knots cannot squash the rest
        ax = fig.add_subplot(1, len(layers), j + 1, projection="3d")
        style.floor_cloud(ax, P, [colors[recs[i]["given_qid"]] for i in te], size=2.6)
        ax.set_title(f"layer {layer} of {L}", y=0.9, fontsize=7, color=style.INK)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=3.5, color=colors[q], label=answers[q]["label"]) for q in top]
    fig.legend(handles=handles, loc="lower center", ncol=12, frameon=False, fontsize=6, handletextpad=0.1, columnspacing=0.8)
    plt.subplots_adjust(wspace=-0.12, left=0, right=1, top=1, bottom=0.12)
    style.save(fig, out / "fig_hero.pdf")
    return [answers[q]["label"] for q in top], [colors[q] for q in top]


def band(ax, rows, key, color, label, lw=1.4, ls="-"):
    x = [r["depth"] for r in rows if r.get(key) and r[key].get("mean") is not None]
    m = [r[key]["mean"] for r in rows if r.get(key) and r[key].get("mean") is not None]
    lo = [r[key]["lo"] for r in rows if r.get(key) and r[key].get("mean") is not None]
    hi = [r[key]["hi"] for r in rows if r.get(key) and r[key].get("mean") is not None]
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, m, color=color, lw=lw, ls=ls, label=label)


def decodability(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(style.PAGEW, 2.5), sharey=True)
    ax = axes[0]
    for m, c in MODEL_COLORS.items():
        if m in summary:
            band(ax, summary[m]["depth"], "acc", c, MODEL_NAMES[m])
    if "llama8b_untrained" in summary:
        band(ax, summary["llama8b_untrained"]["depth_trained_labels"], "acc", style.MUTED, "untrained (Llama's answers)", ls="--")
    if "llama8b" in summary:
        rows = summary["llama8b"]["depth"]
        ax.plot([r["depth"] for r in rows], [r["acc_shuffled"] for r in rows], color=style.MUTED, lw=0.9, ls=":", label="shuffled answers")
        tb = summary["llama8b"].get("text_baseline", {})
        if tb.get("mean") is not None:
            ax.axhline(tb["mean"], color=style.INK, lw=0.8, ls="-.", label="question text only")
    ax.set_xlabel("depth (fraction of layers)")
    ax.set_ylabel("held-out answer read from the state")
    ax.set_title("a  The answer becomes readable in one layer", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=6, columnspacing=0.8)
    style.clean(ax)
    ax = axes[1]
    if "llama8b" in summary:
        rows = summary["llama8b"]["depth"]
        for key, c, lab in [("acc", MODEL_COLORS["llama8b"], "same relation"), ("acc_cross_relation", "#8250df", "other relation"),
                            ("acc_cross_wording", "#e16f24", "other wording"), ("acc_ff_to_mcq", "#0a7d8c", "multiple choice"),
                            ("acc_no_answer_words", style.INK, "answer words removed")]:
            if any(r.get(key) for r in rows):
                band(ax, rows, key, c, lab)
    ax.set_xlabel("depth (fraction of layers)")
    ax.set_title("b  A place is shared across questions (Llama-3.1-8B)", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=6, columnspacing=1.0)
    style.clean(ax)
    fig.tight_layout(w_pad=1.2)
    style.save(fig, out / "fig_decodability.pdf")


def contraction(summary, out):
    fig, ax = plt.subplots(figsize=(style.COLW, 2.0))
    for m, c in MODEL_COLORS.items():
        if m in summary:
            rows = [r for r in summary[m]["depth"] if r.get("spread")]
            ax.plot([r["depth"] for r in rows], [r["spread"] for r in rows], color=c, lw=1.4, label=MODEL_NAMES[m])
    ax.set_yscale("log")
    ax.set_xlabel("depth (fraction of layers)")
    ax.set_ylabel("spread within / between places")
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    style.clean(ax)
    fig.tight_layout()
    style.save(fig, out / "fig_contraction.pdf")


def steering(summary, out):
    """(a) Wrong answers pushed toward the right answer's place (solid) or a random direction of the
    same length (dotted). (b) Right answers pushed toward another answer's place. Ticks: each model's
    onset layer (where answers first become readable)."""
    fig, axes = plt.subplots(1, 2, figsize=(style.PAGEW, 2.0), sharey=True)

    def curve(rows, target, cond, key):
        agg = {}
        for r in rows:
            if r["target"] == target and r["condition"] == cond and r["scale"] == 1.0:
                h, n = agg.get(r["depth"], (0, 0))
                agg[r["depth"]] = (h + r[key] * r["n"], n + r["n"])
        xs = sorted(agg)
        return xs, [agg[x][0] / agg[x][1] for x in xs]

    for m, c in MODEL_COLORS.items():
        rows = summary.get(m, {}).get("steering", [])
        if not rows:
            continue
        for ax, target, cond, key in [(axes[0], "wrong_placed", "gold", "to_gold"), (axes[1], "right", "other", "to_toward")]:
            xs, ys = curve(rows, target, cond, key)
            ax.plot(xs, ys, color=c, lw=1.4, marker="o", ms=2.3, label=MODEL_NAMES[m])
            rx, ry = curve(rows, target, "random", "to_gold" if target != "right" else "to_toward")
            ax.plot(rx, ry, color=c, lw=0.9, ls=":")
            ax.plot([summary[m]["onset_depth"]] * 2, [-0.06, -0.01], color=c, lw=1.6, clip_on=False)
    axes[0].set_title("a  Wrong answers pushed toward the right place", loc="left", fontweight="bold")
    axes[1].set_title("b  Right answers pushed toward another place", loc="left", fontweight="bold")
    for ax in axes:
        ax.set_xlabel("depth of the push (ticks: where answers become readable)")
        ax.set_ylim(-0.06, 1.02)
        style.clean(ax)
    axes[0].set_ylabel("gives the answer pushed toward")
    axes[0].plot([], [], color=style.MUTED, lw=0.9, ls=":", label="random push, same length")
    axes[0].legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout(w_pad=1.2)
    style.save(fig, out / "fig_steering.pdf")


def errors(summary, out):
    """(a) Among wrong free-form answers: how well does each signal predict that the model still picks
    the right option in multiple choice? (b) Questions misled by a user's wrong suggestion: share of
    states nearer the right answer's place than the suggested answer's place, by depth."""
    fig, axes = plt.subplots(1, 2, figsize=(style.PAGEW, 2.45), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    models = [m for m in MODEL_COLORS if m in summary and summary[m].get("errors_beyond_confidence", {}).get("auc_both")]
    w = 0.26
    for k, (key, c, lab) in enumerate([("auc_confidence", style.MUTED, "confidence"), ("auc_geometry", style.ACCENT, "place geometry"),
                                       ("auc_both", style.INK, "both")]):
        ax.bar(np.arange(len(models)) + (k - 1) * w, [summary[m]["errors_beyond_confidence"][key] for m in models],
               width=w, color=c, label=lab)
    ax.axhline(0.5, color=style.MUTED, lw=0.6, ls=":")
    short = {"llama8b": "Llama\n8B", "qwen7b": "Qwen\n7B", "gemma9b": "Gemma\n9B", "llama1b": "Llama\n1B"}
    ax.set_xticks(range(len(models)), [short[m] for m in models], fontsize=6.5)
    ax.set_ylim(0.4, 0.97)
    ax.set_ylabel("AUC, held-out subjects")
    ax.set_title("a  Which wrong answers are still recognised", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=6, ncol=3, loc="upper left")
    style.clean(ax)
    ax = axes[1]
    for m, c in MODEL_COLORS.items():
        h = summary.get(m, {}).get("hint")
        if not h:
            continue
        rows = h["trajectory"]
        ax.plot([r["depth"] for r in rows], [r["misled"]["near_gold"] for r in rows], color=c, lw=1.4, label=f"{MODEL_NAMES[m]}, gave a wrong answer")
        ax.plot([r["depth"] for r in rows], [r["resisted"]["near_gold"] for r in rows], color=c, lw=0.9, ls="--", alpha=0.8)
    ax.plot([], [], color=style.INK, lw=0.9, ls="--", label="dashed: kept the right answer")
    ax.axhline(0.5, color=style.MUTED, lw=0.6, ls=":")
    ax.set_xlabel("depth (fraction of layers)")
    ax.set_ylabel("nearer the right place")
    ax.set_title("b  Asked again with a wrong suggestion", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, columnspacing=1.0)
    style.clean(ax)
    fig.tight_layout(w_pad=1.2)
    style.save(fig, out / "fig_errors.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(ROOT / "runs" / "main"))
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "paper" / "figures"))
    ap.add_argument("--hero-model", default="llama8b")
    args = ap.parse_args()
    style.setup()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((Path(args.results) / "summary.json").read_text())
    prompts = [json.loads(l) for l in open(ROOT / "dataset" / "prompts.jsonl")]
    answers = json.load(open(ROOT / "dataset" / "answers.json"))
    names, cols = hero(Path(args.run_dir), args.hero_model, prompts, answers, out, onset=summary.get(args.hero_model, {}).get("onset_layer"))
    (out / "hero_legend.json").write_text(json.dumps({"names": names, "colors": cols}))
    decodability(summary, out)
    contraction(summary, out)
    steering(summary, out)
    errors(summary, out)


if __name__ == "__main__":
    main()
