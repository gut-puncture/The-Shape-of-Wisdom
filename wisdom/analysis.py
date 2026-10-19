"""All measurements for the paper, from recorded states (CPU only).

    python -m wisdom.analysis --run-dir runs/main --models llama8b qwen7b gemma9b llama1b --out results

Geometry is measured after a per-layer whitening fitted on training subjects without labels
(centre, PCA to K dimensions, scale each component to unit variance). A question's "place"
is the centroid, at that layer, of training-subject questions to which the model gave the
same answer. Every score below is computed on test subjects only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from wisdom.places import MIN_ITEMS_PER_PLACE, load_stage

ROOT = Path(__file__).resolve().parent.parent
K = 128          # whitened dimensions kept per layer
SHRINK = 0.1     # covariance shrinkage toward the mean eigenvalue
SEED = 0


class Whitener:
    def __init__(self, X: np.ndarray, k=K):
        X = X.astype(np.float64)
        self.mu = X.mean(0)
        U, S, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        var = S**2 / max(len(X) - 1, 1)
        var = (1 - SHRINK) * var + SHRINK * var.mean()
        var = np.maximum(var, 1e-6 * var.max() + 1e-12)  # never divide by a zero-variance direction
        self.P = Vt[:k].T / np.sqrt(var[:k])

    def __call__(self, X):
        return (X.astype(np.float64) - self.mu) @ self.P


def project_out(X, rows):
    """Remove the span of the given unembedding rows (the 'answer words' directions) from X."""
    Q, _ = np.linalg.qr(rows.T.astype(np.float64))
    X = X.astype(np.float64)
    return X - (X @ Q) @ Q.T


def centroids(Z, labels):
    out = {}
    for q in set(labels):
        idx = [i for i, l in enumerate(labels) if l == q]
        if len(idx) >= MIN_ITEMS_PER_PLACE:
            out[q] = Z[idx].mean(0)
    return out


def nearest(Z, cents):
    keys = sorted(cents)
    C = np.stack([cents[k] for k in keys])
    d = ((Z[:, None, :] - C[None]) ** 2).sum(-1)
    order = np.argsort(d, axis=1)
    return [keys[i] for i in order[:, 0]], d, keys


def nc_accuracy(Ztr, ytr, Zte, yte, types_tr, types_te):
    """Nearest-centroid accuracy within answer type (a city is only compared with cities).
    Returns (accuracy, chance) over test items whose label has a place."""
    hits, chance, n = 0, 0.0, 0
    for t in sorted(set(types_te)):
        tr = [i for i, x in enumerate(types_tr) if x == t]
        te = [i for i, x in enumerate(types_te) if x == t]
        cents = centroids(Ztr[tr], [ytr[i] for i in tr])
        te = [i for i in te if yte[i] in cents]
        if len(cents) < 2 or not te:
            continue
        pred, _, _ = nearest(Zte[te], cents)
        hits += sum(p == yte[i] for p, i in zip(pred, te))
        chance += len(te) / len(cents)
        n += len(te)
    return (hits / n if n else np.nan), (chance / n if n else np.nan), n


def spread_ratio(Z, labels, types):
    """Within-place variance over between-place variance (lower = tighter places), pooled by type."""
    num, den = 0.0, 0.0
    for t in sorted(set(types)):
        idx = [i for i, x in enumerate(types) if x == t]
        cents = centroids(Z[idx], [labels[i] for i in idx])
        if len(cents) < 2:
            continue
        keep = [i for i in idx if labels[i] in cents]
        within = np.mean([((Z[i] - cents[labels[i]]) ** 2).sum() for i in keep])
        C = np.stack(list(cents.values()))
        between = ((C - C.mean(0)) ** 2).sum(1).mean()
        num += within * len(keep)
        den += between * len(keep)
    return num / den if den else np.nan


def load_model_run(run_dir: Path, model: str, prompts, answers):
    d = run_dir / model
    out = {"meta": json.loads((d / "model_meta.json").read_text())}
    for stage in ("ff", "mcq", "hint"):
        if (d / stage / "results.jsonl").exists():
            out[stage] = load_stage(d / stage, prompts, answers)
    if (d / "unembed_answers.npz").exists():
        out["unembed"] = np.load(d / "unembed_answers.npz")["rows"]
    if (d / "steer" / "results.jsonl").exists():
        out["steer"] = [json.loads(l) for l in open(d / "steer" / "results.jsonl")]
    return out


def places_by_depth(recs, S, unembed=None, label_key="given_qid", shuffle=False):
    """Per layer: held-out nearest-centroid accuracy and spread ratio, plus cross-wording,
    cross-relation and answer-words-removed variants."""
    rng = np.random.default_rng(SEED)
    tr = [i for i, r in enumerate(recs) if r["split"] == "train" and r.get(label_key)]
    te = [i for i, r in enumerate(recs) if r["split"] == "test" and r.get(label_key)]
    y = [r.get(label_key) for r in recs]
    if shuffle:  # permute labels within answer type: destroys any label-state link
        for t in {r["answer_type"] for r in recs}:
            idx = [i for i in tr + te if recs[i]["answer_type"] == t]
            perm = rng.permutation([y[i] for i in idx])
            for i, l in zip(idx, perm):
                y[i] = l
    types = [r["answer_type"] for r in recs]
    rows = []
    # Slot 0 (the input to block 0) holds the same template token for every prompt: no information.
    for layer in range(1, S.shape[1]):
        X = S[:, layer]
        W = Whitener(X[tr])
        Z = W(X)
        row = {"layer": layer, "depth": layer / (S.shape[1] - 1)}
        row["acc"], row["chance"], row["n"] = nc_accuracy(Z[tr], [y[i] for i in tr], Z[te], [y[i] for i in te],
                                                          [types[i] for i in tr], [types[i] for i in te])
        row["spread"] = spread_ratio(Z[te], [y[i] for i in te], [types[i] for i in te])
        # Across wordings: places from the plain wording, tested on natural rewrites.
        a = [i for i in tr if recs[i]["wording"] == "w0"]
        b = [i for i in te if recs[i]["wording"] != "w0"]
        if b:
            row["acc_cross_wording"], _, _ = nc_accuracy(Z[a], [y[i] for i in a], Z[b], [y[i] for i in b],
                                                         [types[i] for i in a], [types[i] for i in b])
        # Across relations: places from one relation, tested on the other relations with the same answers.
        accs = []
        for t in {r["answer_type"] for r in recs}:
            rels = sorted({recs[i]["relation"] for i in tr + te if types[i] == t})
            for r_fit in rels:
                a = [i for i in tr if types[i] == t and recs[i]["relation"] == r_fit]
                b = [i for i in te if types[i] == t and recs[i]["relation"] != r_fit]
                if a and b:
                    acc, ch, n = nc_accuracy(Z[a], [y[i] for i in a], Z[b], [y[i] for i in b],
                                             [types[i] for i in a], [types[i] for i in b])
                    if n:
                        accs.append((acc, ch, n))
        if accs:
            n_tot = sum(n for _, _, n in accs)
            row["acc_cross_relation"] = sum(a * n for a, _, n in accs) / n_tot
            row["chance_cross_relation"] = sum(c * n for _, c, n in accs) / n_tot
        if unembed is not None:
            Xp = project_out(X, unembed)
            Wp = Whitener(Xp[tr])
            Zp = Wp(Xp)
            row["acc_no_answer_words"], _, _ = nc_accuracy(Zp[tr], [y[i] for i in tr], Zp[te], [y[i] for i in te],
                                                           [types[i] for i in tr], [types[i] for i in te])
        rows.append(row)
    return rows


def text_baseline(recs, label_key="given_qid"):
    """The same nearest-centroid test on the question text alone (character n-gram TF-IDF)."""
    tr = [i for i, r in enumerate(recs) if r["split"] == "train" and r.get(label_key)]
    te = [i for i, r in enumerate(recs) if r["split"] == "test" and r.get(label_key)]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2).fit([recs[i]["text"] for i in tr])
    from sklearn.decomposition import TruncatedSVD
    T = vec.transform([r["text"] for r in recs])
    Z = TruncatedSVD(min(256, T.shape[1] - 1), random_state=SEED).fit(T[tr]).transform(T)
    y = [r.get(label_key) for r in recs]
    types = [r["answer_type"] for r in recs]
    acc, ch, n = nc_accuracy(Z[tr], [y[i] for i in tr], Z[te], [y[i] for i in te], [types[i] for i in tr], [types[i] for i in te])
    return {"acc": acc, "chance": ch, "n": n}


def fit_law(rows):
    """Does log(spread) fall as a straight line in depth (constant contraction per layer)
    or as a step? Compare least-squares fits by AIC."""
    x = np.array([r["depth"] for r in rows])
    y = np.log(np.array([r["spread"] for r in rows]))
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    lin = np.polyfit(x, y, 1)
    rss_lin = ((np.polyval(lin, x) - y) ** 2).sum()
    best = None
    for c in np.linspace(0.1, 0.95, 86):  # step = two-level fit with a logistic edge of fixed width
        for w in (0.01, 0.03, 0.06):
            g = 1 / (1 + np.exp(-(x - c) / w))
            A = np.c_[np.ones(n), g]
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            rss = ((A @ coef - y) ** 2).sum()
            if best is None or rss < best[0]:
                best = (rss, c, w, coef)
    aic = lambda rss, k: n * np.log(rss / n + 1e-12) + 2 * k
    return {"linear_slope": lin[0], "aic_linear": aic(rss_lin, 2), "aic_step": aic(best[0], 4),
            "step_center": best[1], "step_width": best[2], "step_drop": best[3][1]}


def onset_layer(recs, S, label_key="given_qid"):
    """First layer where nearest-centroid accuracy covers half the way from chance to its maximum,
    chosen using training subjects only (fit on one half of them, score on the other half), so the
    layer at which errors are then examined is never selected with test data."""
    tr = [i for i, r in enumerate(recs) if r["split"] == "train" and r.get(label_key)]
    a = [i for i in tr if stable_half(recs[i]["subject_qid"])]
    b = [i for i in tr if not stable_half(recs[i]["subject_qid"])]
    y = [r.get(label_key) for r in recs]
    types = [r["answer_type"] for r in recs]
    acc, ch = [], []
    for layer in range(1, S.shape[1]):
        Z = Whitener(S[a, layer])(S[:, layer])
        x, c, _ = nc_accuracy(Z[a], [y[i] for i in a], Z[b], [y[i] for i in b], [types[i] for i in a], [types[i] for i in b])
        acc.append(x)
        ch.append(c)
    acc = np.array(acc)
    target = np.nanmean(ch) + 0.5 * (np.nanmax(acc) - np.nanmean(ch))
    return 1 + int(np.argmax(acc >= target))


def stable_half(key) -> bool:
    from wisdom.prompts import stable_unit
    return stable_unit("half", key) < 0.5


def error_anatomy(recs, S, layer, label_key="given_qid"):
    """Where each wrong answer's state sits at `layer`, relative to places fitted on training subjects.
    Classes: in_gold (nearest place is the right answer), in_given (nearest is the wrong answer it gave),
    in_third (some other place), nowhere (farther from every place than 90% of right answers are from theirs)."""
    tr = [i for i, r in enumerate(recs) if r["split"] == "train" and r.get(label_key)]
    W = Whitener(S[tr, layer])
    Z = W(S[:, layer])
    out, radius = [], {}
    for t in sorted({r["answer_type"] for r in recs}):
        tt = [i for i in tr if recs[i]["answer_type"] == t]
        cents = centroids(Z[tt], [recs[i][label_key] for i in tt])
        if len(cents) < 2:
            continue
        te = [i for i, r in enumerate(recs) if r["split"] == "test" and r["answer_type"] == t and r["wording"] == "w0"]
        pred, D, keys = nearest(Z[te], cents)
        right = [k for k, i in enumerate(te) if recs[i]["correct"] and recs[i]["answer_qid"] in cents]
        own = [D[k, keys.index(recs[te[k]]["answer_qid"])] for k in right]
        radius[t] = float(np.percentile(own, 90)) if own else np.inf
        for k, i in enumerate(te):
            r = recs[i]
            dmin = D[k].min()
            sorted_d = np.sort(D[k])
            if r["correct"]:
                cls = "right"
            elif dmin > radius[t]:
                cls = "nowhere"
            elif pred[k] == r["answer_qid"]:
                cls = "in_gold"
            elif pred[k] == r.get("given_qid"):
                cls = "in_given"
            else:
                cls = "in_third"
            out.append({"prompt_id": r["prompt_id"], "fact_id": r["fact_id"], "class": cls, "correct": r["correct"],
                        "answer_type": t, "dist_nearest": float(dmin), "place_margin": float(sorted_d[1] - sorted_d[0]),
                        "dist_gold": float(D[k, keys.index(r["answer_qid"])]) if r["answer_qid"] in cents else np.nan,
                        "seq_logprob": float(sum(r.get("token_logprobs") or [np.nan])),
                        "answer_margin": answer_margin(r),
                        "subject_qid": r["subject_qid"], "band": r["band"]})
    return out, radius


def answer_margin(r):
    lp = sorted((r.get("answer_logprobs") or {}).values(), reverse=True)
    return lp[0] - lp[1] if len(lp) >= 2 else np.nan


def auc_beyond_confidence(anat):
    """Does the place geometry predict a right answer better than the model's own confidence?
    Grouped 5-fold cross-validation by subject; confidence = the reply's log-probability and the
    margin between the two best-scoring answers at the first answer position."""
    rows = [a for a in anat if np.isfinite(a["seq_logprob"]) and np.isfinite(a["answer_margin"]) and np.isfinite(a["dist_nearest"])]
    y = np.array([a["correct"] for a in rows])
    groups = [a["subject_qid"] for a in rows]
    conf = np.array([[a["seq_logprob"], a["answer_margin"]] for a in rows])
    geo = np.array([[np.log(a["dist_nearest"] + 1e-9), np.log(a["place_margin"] + 1e-9)] for a in rows])
    res = {}
    for name, X in [("confidence", conf), ("geometry", geo), ("both", np.c_[conf, geo])]:
        p = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=GroupKFold(5), groups=groups, method="predict_proba")[:, 1]
        res[name] = roc_auc_score(y, p)
    # bootstrap CI for the gain, resampling subjects
    rng = np.random.default_rng(SEED)
    pc = cross_val_predict(LogisticRegression(max_iter=2000), conf, y, cv=GroupKFold(5), groups=groups, method="predict_proba")[:, 1]
    pb = cross_val_predict(LogisticRegression(max_iter=2000), np.c_[conf, geo], y, cv=GroupKFold(5), groups=groups, method="predict_proba")[:, 1]
    subj = np.array(groups)
    uniq = np.unique(subj)
    gains = []
    for _ in range(500):
        pick = rng.choice(uniq, len(uniq))
        idx = np.concatenate([np.where(subj == s)[0] for s in pick])
        if len(set(y[idx])) == 2:
            gains.append(roc_auc_score(y[idx], pb[idx]) - roc_auc_score(y[idx], pc[idx]))
    res["gain_ci95"] = [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))]
    res["n"] = len(rows)
    return res


def steering_summary(steer, recs_by_pid, grader_given):
    """Share of pushes after which the model gives the target answer, by target kind, condition, layer."""
    out = {}
    for s in steer:
        rec = recs_by_pid.get(s["prompt_id"])
        if rec is None:
            continue
        given = grader_given(s["reply"], rec["answer_type"])
        target = s["other_qid"] if s["condition"] == "other" else rec["answer_qid"]
        k = (s["target"], s["condition"], s["scale"], s["layer"])
        hit, n = out.get(k, (0, 0))
        out[k] = (hit + int(given == target), n + 1)
    return [{"target": k[0], "condition": k[1], "scale": k[2], "layer": k[3], "rate": h / n, "n": n}
            for k, (h, n) in sorted(out.items())]


def main():
    from wisdom.grading import Grader
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(ROOT / "runs" / "main"))
    ap.add_argument("--models", nargs="+", default=["llama8b", "qwen7b", "gemma9b", "llama1b", "llama8b_untrained"])
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    prompts = [json.loads(l) for l in open(ROOT / "dataset" / "prompts.jsonl")]
    answers = json.load(open(ROOT / "dataset" / "answers.json"))
    grader = Grader(answers)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for m in args.models:
        if not (Path(args.run_dir) / m / "ff" / "results.jsonl").exists():
            continue
        run = load_model_run(Path(args.run_dir), m, prompts, answers)
        recs, S = run["ff"]
        label = "answer_qid" if "untrained" in m else "given_qid"  # an untrained model has no beliefs to group by
        rows = places_by_depth(recs, S, run.get("unembed"), label_key=label)
        null = places_by_depth(recs, S, label_key=label, shuffle=True)
        res = {"accuracy": float(np.mean([r["correct"] for r in recs if r["wording"] == "w0"])),
               "depth": rows, "shuffled": [{"layer": r["layer"], "acc": r["acc"], "chance": r["chance"]} for r in null],
               "text_baseline": text_baseline(recs, label), "law": fit_law(rows)}
        if "untrained" not in m:
            L = onset_layer(recs, S)
            anat, radius = error_anatomy(recs, S, L)
            res.update(onset_layer=L, onset_depth=L / (S.shape[1] - 1), radius=radius,
                       anatomy_counts={c: sum(a["class"] == c for a in anat) for c in sorted({a["class"] for a in anat})},
                       auc=auc_beyond_confidence(anat))
            with open(out / f"{m}_anatomy.jsonl", "w") as f:
                for a in anat:
                    f.write(json.dumps(a) + "\n")
            if "steer" in run:
                by_pid = {r["prompt_id"]: r for r in recs}
                if "hint" in run:
                    by_pid.update({r["prompt_id"]: r for r in run["hint"][0]})
                res["steering"] = steering_summary(run["steer"], by_pid, grader.given)
        summary[m] = res
        print(m, "acc", round(res["accuracy"], 3), "onset", res.get("onset_depth"), "law", res["law"], flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
