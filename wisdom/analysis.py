"""All measurements for the paper, from recorded states (CPU only).

    python -m wisdom.analysis --run-dir runs/main --out results

Definitions used throughout
  place       the mean state, at one layer, of training-subject questions to which the model gave the
              same answer (its belief, graded conservatively; unclear replies are excluded).
  held out    every score is computed on test subjects; nothing about a test subject (its answer,
              its correctness) is used to fit a place, a whitening, a radius or a layer choice.
  unit        the subject: uncertainty comes from resampling subjects (with all their prompts).

Two complementary geometric measurements per layer:
  decodability  nearest-place accuracy in a per-layer whitened space (robust to the residual stream's
                few huge directions; answers "is the answer readable from where the state is?").
  contraction   within-place / between-place spread in the model's own (centred, unwhitened) space,
                places from training subjects, spread measured on test subjects. One fixed metric per
                layer, so a change across depth is a change in the states, not in the ruler.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from wisdom.places import MIN_ITEMS_PER_PLACE, load_stage, read_jsonl
from wisdom.prompts import stable_unit

ROOT = Path(__file__).resolve().parent.parent
K = 128           # whitened dimensions kept per layer
SHRINK = 0.1      # covariance shrinkage toward the mean eigenvalue
N_BOOT = 300
N_SHUFFLE = 5
MIN_SUBJECTS = 3  # a place needs answers from at least this many distinct training subjects


# ---------- geometry primitives ----------

class Whitener:
    def __init__(self, X, k=K):
        X = X.astype(np.float64)
        self.mu = X.mean(0)
        U, S, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        var = S**2 / max(len(X) - 1, 1)
        var = (1 - SHRINK) * var + SHRINK * var.mean()
        var = np.maximum(var, 1e-6 * var.max() + 1e-12)
        k = min(k, len(var))
        self.P = Vt[:k].T / np.sqrt(var[:k])

    def __call__(self, X):
        return (X.astype(np.float64) - self.mu) @ self.P


def place_means(Z, recs, idx, label):
    """{answer: mean of Z over idx}, keeping answers given by at least MIN_SUBJECTS training subjects
    and MIN_ITEMS_PER_PLACE prompts."""
    groups = {}
    for i in idx:
        q = recs[i].get(label)
        if q:
            groups.setdefault(q, []).append(i)
    out = {}
    for q, ii in groups.items():
        if len(ii) >= MIN_ITEMS_PER_PLACE and len({recs[i]["subject_qid"] for i in ii}) >= MIN_SUBJECTS:
            out[q] = Z[ii].mean(0)
    return out


def nearest_place(Z, means):
    keys = sorted(means)
    C = np.stack([means[k] for k in keys])
    D = ((Z[:, None, :] - C[None]) ** 2).sum(-1)
    return keys, D


def by_type(recs, idx):
    out = {}
    for i in idx:
        out.setdefault(recs[i]["answer_type"], []).append(i)
    return out


def decode(Z, recs, fit_idx, test_idx, label, test_label=None):
    """Per test item: was the nearest place (same answer type, fitted on fit_idx) the item's label?
    Returns (hits dict item->0/1, chance dict item->1/#places)."""
    test_label = test_label or label
    hits, chance = {}, {}
    fit_t, test_t = by_type(recs, fit_idx), by_type(recs, test_idx)
    for t, ti in test_t.items():
        means = place_means(Z, recs, fit_t.get(t, []), label)
        ti = [i for i in ti if recs[i].get(test_label) in means]
        if len(means) < 2 or not ti:
            continue
        keys, D = nearest_place(Z[ti], means)
        for k, i in enumerate(ti):
            hits[i] = int(keys[int(D[k].argmin())] == recs[i][test_label])
            chance[i] = 1 / len(means)
    return hits, chance


def subject_bootstrap(values: dict, recs, n=N_BOOT, seed=0):
    """Mean of per-item values with a 95% interval from resampling subjects."""
    if not values:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    subj = {}
    for i, v in values.items():
        subj.setdefault(recs[i]["subject_qid"], []).append(v)
    keys = sorted(subj)
    sums = np.array([sum(subj[k]) for k in keys])
    cnts = np.array([len(subj[k]) for k in keys])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        boots.append(sums[pick].sum() / cnts[pick].sum())
    return {"mean": float(sums.sum() / cnts.sum()), "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)), "n": int(cnts.sum()), "subjects": len(keys)}


def answer_directions(unembed_npz):
    """The residual-stream directions the output layer reads for the answer words: each token's
    unembedding row times the final norm's measured per-dimension scale. Orthonormal basis via rank-thresholded SVD."""
    z = np.load(unembed_npz)
    R = z["rows"] * z["final_norm_scale"][None, :]
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    keep = S > S.max() * 1e-6
    return Vt[keep].T  # (d, r)


def remove(X, B):
    X = X.astype(np.float64)
    return X - (X @ B) @ B.T


def random_basis(d, r, seed=0):
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, r)))
    return Q


# ---------- per-model analyses ----------

def split_idx(recs, cond=lambda r: True):
    tr = [i for i, r in enumerate(recs) if r["split"] == "train" and cond(r)]
    te = [i for i, r in enumerate(recs) if r["split"] == "test" and cond(r)]
    return tr, te


def shuffled_labels(recs, idx, label, seed):
    """Permute labels between facts (all wordings of a fact keep one shared label) within answer type."""
    rng = np.random.default_rng(seed)
    out = {}
    for t, ii in by_type(recs, idx).items():
        facts = sorted({recs[i]["fact_id"] for i in ii})
        lab = {}
        for i in ii:
            lab.setdefault(recs[i]["fact_id"], recs[i].get(label))
        perm = rng.permutation([lab[f] for f in facts])
        new = dict(zip(facts, perm))
        for i in ii:
            out[i] = new[recs[i]["fact_id"]]
    return out


def depth_profile(recs, S, label="given_qid", unembed=None, mcq=None):
    """Per layer: decodability (+ variants and nulls) and contraction, all held out."""
    tr, te = split_idx(recs)
    w0_tr = [i for i in tr if recs[i]["wording"] == "w0"]
    rw_te = [i for i in te if recs[i]["wording"] != "w0"]
    B = answer_directions(unembed) if unembed is not None else None
    Brand = random_basis(S.shape[2], B.shape[1], seed=1) if B is not None else None
    rows = []
    for layer in range(1, S.shape[1]):  # slot 0 is the same template token for every prompt
        X = S[:, layer]
        W = Whitener(X[tr])
        Z = W(X)
        row = {"layer": layer, "depth": layer / (S.shape[1] - 1)}
        hits, ch = decode(Z, recs, tr, te, label)
        row["acc"] = subject_bootstrap(hits, recs)
        row["chance"] = float(np.mean(list(ch.values()))) if ch else None
        # null: labels shuffled between facts (whitening is label-free, so it is reused)
        null = []
        for s in range(N_SHUFFLE):
            lab = shuffled_labels(recs, tr + te, label, seed=s)
            tmp = [dict(r) for r in recs]
            for i, l in lab.items():
                tmp[i]["_shuf"] = l
            h, _ = decode(Z, tmp, tr, te, "_shuf")
            null.append(np.mean(list(h.values())) if h else np.nan)
        row["acc_shuffled"] = float(np.nanmean(null))
        if rw_te:
            h, _ = decode(Z, recs, w0_tr, rw_te, label)
            row["acc_cross_wording"] = subject_bootstrap(h, recs)
        h = {}
        for t in {recs[i]["answer_type"] for i in tr}:
            rels = sorted({recs[i]["relation"] for i in tr + te if recs[i]["answer_type"] == t})
            for r_fit in rels:
                a = [i for i in tr if recs[i]["answer_type"] == t and recs[i]["relation"] == r_fit]
                b = [i for i in te if recs[i]["answer_type"] == t and recs[i]["relation"] != r_fit]
                hh, _ = decode(Z, recs, a, b, label)
                for i, v in hh.items():
                    h.setdefault(i, []).append(v)
        row["acc_cross_relation"] = subject_bootstrap({i: np.mean(v) for i, v in h.items()}, recs)
        if B is not None:
            for name, basis in (("acc_no_answer_words", B), ("acc_random_removed", Brand)):
                Xr = remove(X, basis)
                Zr = Whitener(Xr[tr])(Xr)
                hh, _ = decode(Zr, recs, tr, te, label)
                row[name] = subject_bootstrap(hh, recs)
        if mcq is not None:  # places from free-form questions, tested on multiple-choice states
            mrecs, MS = mcq
            Zm = W(MS[:, layer])
            allr = recs + mrecs
            Zall = np.concatenate([Z, Zm])
            m_te = [len(recs) + i for i, r in enumerate(mrecs) if r["split"] == "test"]
            hh, _ = decode(Zall, allr, tr, m_te, label)
            row["acc_ff_to_mcq"] = subject_bootstrap({i - len(recs): v for i, v in hh.items()}, mrecs)
        row["spread"] = contraction(X, recs, tr, te, label)
        rows.append(row)
    return rows


def contraction(X, recs, tr, te, label):
    """Within-place spread of test questions around training places, over the spread of the places
    themselves, in the model's own centred space; pooled over answer types."""
    X = X.astype(np.float64) - X[tr].astype(np.float64).mean(0)
    num, den, n = 0.0, 0.0, 0
    for t, ti in by_type(recs, te).items():
        means = place_means(X, recs, by_type(recs, tr).get(t, []), label)
        ti = [i for i in ti if recs[i].get(label) in means]
        if len(means) < 2 or not ti:
            continue
        C = np.stack(list(means.values()))
        within = np.mean([((X[i] - means[recs[i][label]]) ** 2).sum() for i in ti])
        between = ((C - C.mean(0)) ** 2).sum(1).mean()
        num += within * len(ti)
        den += between * len(ti)
        n += len(ti)
    return num / den if den else None


def shape_test(rows):
    """Does log-spread fall with depth as a step or as a straight line (a constant contraction per
    layer)? The two shapes are compared by leave-one-layer-out prediction error, which charges the
    more flexible step model for its flexibility."""
    x = np.array([r["depth"] for r in rows if r["spread"]])
    y = np.log(np.array([r["spread"] for r in rows if r["spread"]]))

    def fit_line(xs, ys):
        c = np.polyfit(xs, ys, 1)
        return lambda q: np.polyval(c, q)

    def fit_step(xs, ys):
        best = None
        for c in np.linspace(0.05, 0.95, 91):
            for w in (0.01, 0.03, 0.06, 0.1):
                g = 1 / (1 + np.exp(-(xs - c) / w))
                A = np.c_[np.ones_like(xs), g]
                coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
                rss = ((A @ coef - ys) ** 2).sum()
                if best is None or rss < best[0]:
                    best = (rss, c, w, coef)
        _, c, w, coef = best
        return lambda q: coef[0] + coef[1] / (1 + np.exp(-(q - c) / w)), (c, w, coef)

    err = {"line": [], "step": []}
    for k in range(len(x)):
        m = np.arange(len(x)) != k
        err["line"].append((fit_line(x[m], y[m])(x[k]) - y[k]) ** 2)
        err["step"].append((fit_step(x[m], y[m])[0](x[k]) - y[k]) ** 2)
    f, (c, w, coef) = fit_step(x, y)
    return {"loo_mse_line": float(np.mean(err["line"])), "loo_mse_step": float(np.mean(err["step"])),
            "step_center": float(c), "step_width": float(w), "step_drop": float(coef[1]),
            "line_slope": float(np.polyfit(x, y, 1)[0])}


def text_baseline(recs, label="given_qid"):
    """The same held-out nearest-place test on the question's words alone (character n-gram TF-IDF,
    reduced to 256 dimensions on training questions): how much of a 'place' could be read from the
    wording without any model?"""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    tr, te = split_idx(recs)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2).fit([recs[i]["text"] for i in tr])
    T = vec.transform([r["text"] for r in recs])
    Z = TruncatedSVD(min(256, T.shape[1] - 1), random_state=0).fit(T[tr]).transform(T)
    hits, ch = decode(Z, recs, tr, te, label)
    return {**subject_bootstrap(hits, recs), "chance": float(np.mean(list(ch.values()))) if ch else None}


def onset_layer(recs, S, label="given_qid"):
    """Layer where decodability first covers half the way from chance to its peak, chosen with
    training subjects only (places from one half of them, scored on the other half)."""
    tr, _ = split_idx(recs)
    a = [i for i in tr if stable_unit("half", recs[i]["subject_qid"]) < 0.5]
    b = [i for i in tr if i not in set(a)]
    acc, ch = [], []
    for layer in range(1, S.shape[1]):
        Z = Whitener(S[a, layer])(S[:, layer])
        h, c = decode(Z, recs, a, b, label)
        acc.append(np.mean(list(h.values())) if h else np.nan)
        ch.append(np.mean(list(c.values())) if c else np.nan)
    acc, base = np.array(acc), np.nanmean(ch)
    if not np.isfinite(acc).any():
        return None
    return 1 + int(np.argmax(acc >= base + 0.5 * (np.nanmax(acc) - base)))


def seq_confidence(r):
    """The model's log-probability of its own reply, and its length in tokens."""
    lp = r.get("token_logprobs") or []
    return (float(sum(lp)) if lp else np.nan), len(lp)


def anatomy(recs, S, layer, label="given_qid"):
    """Where each test question's state sits at `layer`. Radii come from training subjects only:
    places from one half of them, distances of the other half's right answers to their own place."""
    tr, te = split_idx(recs)
    W = Whitener(S[tr, layer])
    Z = W(S[:, layer])
    a = [i for i in tr if stable_unit("half", recs[i]["subject_qid"]) < 0.5]
    b = [i for i in tr if i not in set(a)]
    out = []
    for t in sorted(by_type(recs, te)):
        means = place_means(Z, recs, by_type(recs, tr).get(t, []), label)
        half = place_means(Z, recs, by_type(recs, a).get(t, []), label)
        own = [((Z[i] - half[recs[i][label]]) ** 2).sum() for i in by_type(recs, b).get(t, [])
               if recs[i]["correct"] and recs[i].get(label) in half]
        if len(means) < 2 or len(own) < 10:
            continue  # unsupported type: reported by its absence in the counts
        radius = float(np.percentile(own, 90))
        keys, D = nearest_place(Z[by_type(recs, te)[t]], means)
        for k, i in enumerate(by_type(recs, te)[t]):
            r = recs[i]
            order = np.argsort(D[k])
            near = keys[order[0]]
            gold_rank = keys.index(r["answer_qid"]) if r["answer_qid"] in means else None
            if r["correct"]:
                cls = "right"
            elif D[k, order[0]] > radius:
                cls = "nowhere"
            elif near == r["answer_qid"]:
                cls = "in_gold"
            elif near == r.get(label):
                cls = "in_given"
            else:
                cls = "in_third"
            conf, length = seq_confidence(r)
            out.append({"i": i, "prompt_id": r["prompt_id"], "fact_id": r["fact_id"], "subject_qid": r["subject_qid"],
                        "wording": r["wording"], "answer_type": t, "correct": r["correct"], "class": cls,
                        "reply_status": r.get("reply_status"), "conf": conf, "length": length,
                        "d_nearest": float(D[k, order[0]]), "d_margin": float(D[k, order[1]] - D[k, order[0]]),
                        "d_gold": float(D[k, gold_rank]) if gold_rank is not None else None,
                        "gold_rank": int(np.where(order == gold_rank)[0][0]) if gold_rank is not None else None,
                        "radius": radius})
    return out


def fact_table(ff, mcq, hint):
    """One row per fact: free-form answers by wording, MCQ choices by order, hint answer."""
    T = {}
    for r in ff:
        T.setdefault(r["fact_id"], {"fact": r})["ff_" + r["wording"]] = r
    for r in mcq or []:
        T.setdefault(r["fact_id"], {"fact": r}).setdefault("mcq", {})[r["perm"]] = r
    for r in hint or []:
        T.setdefault(r["fact_id"], {"fact": r})["hint"] = r
    for f, row in T.items():
        m = row.get("mcq", {})
        if m:
            row["mcq_correct_share"] = float(np.mean([x["correct"] for x in m.values()]))
            given = [x.get("given_qid") for x in m.values()]
            row["mcq_stable"] = len(set(given)) == 1 and given[0] is not None
        ws = [row.get(f"ff_{w}") for w in ("w0", "w1", "w2") if row.get(f"ff_{w}")]
        row["wording_agree"] = len({x.get("given_qid") for x in ws}) == 1
    return T


def errors_beyond_confidence(anat, table):
    """The non-circular headline test. Among wrong free-form answers (plain wording, test subjects),
    does the state's geometry at the onset layer (distance to the gold place, its rank, the margin to
    the nearest place) predict whether the model still recognises the right answer in multiple choice
    (majority of the four orders correct), beyond the model's confidence and reply length?"""
    rows = [a for a in anat if not a["correct"] and a["wording"] == "w0" and a["d_gold"] is not None
            and np.isfinite(a["conf"]) and table.get(a["fact_id"], {}).get("mcq_correct_share") is not None]
    if len(rows) < 40:
        return {"n": len(rows), "status": "insufficient"}
    y = np.array([table[a["fact_id"]]["mcq_correct_share"] > 0.5 for a in rows])
    if y.all() or not y.any():
        return {"n": len(rows), "status": "one class"}
    g = [a["subject_qid"] for a in rows]
    conf = np.array([[a["conf"], a["length"]] for a in rows])
    geo = np.array([[np.log(a["d_gold"] + 1e-9), a["gold_rank"], np.log(a["d_margin"] + 1e-9)] for a in rows])
    folds = GroupKFold(min(5, len(set(g))))
    res = {"n": len(rows), "positive_rate": float(y.mean())}
    preds = {}
    for name, X in (("confidence", conf), ("geometry", geo), ("both", np.c_[conf, geo])):
        preds[name] = cross_val_predict(LogisticRegression(max_iter=5000), X, y, cv=folds, groups=g, method="predict_proba")[:, 1]
        res[f"auc_{name}"] = float(roc_auc_score(y, preds[name]))
    rng = np.random.default_rng(0)
    subj = np.array(g)
    uniq = np.unique(subj)
    gains = []
    for _ in range(N_BOOT):
        idx = np.concatenate([np.where(subj == s)[0] for s in rng.choice(uniq, len(uniq))])
        if 0 < y[idx].sum() < len(idx):
            gains.append(roc_auc_score(y[idx], preds["both"][idx]) - roc_auc_score(y[idx], preds["confidence"][idx]))
    res["gain_both_minus_conf"] = [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))]
    return res


def steering(run_dir, ff, hint, answers, anat):
    from wisdom.grading import Grader
    plan = {j["job_id"]: j for j in read_jsonl(run_dir / "steer" / "plan.jsonl")}
    grader = Grader(answers)
    recs = {r["prompt_id"]: r for r in ff + (hint or [])}
    cls = {a["prompt_id"]: a["class"] for a in anat}
    out = []
    for s in read_jsonl(run_dir / "steer" / "results.jsonl"):
        j = plan.get(s["job_id"])
        r = recs.get(j["prompt_id"]) if j else None
        if r is None:
            continue
        given, status = grader.read(s["reply"], r["answer_type"])
        out.append({"target": j["target"], "class": cls.get(j["prompt_id"]), "layer": j["layer"],
                    "condition": j["condition"], "scale": j["scale"], "subject_qid": r["subject_qid"],
                    "to_gold": given == r["answer_qid"], "to_toward": given == j["toward"], "status": status})
    return out


def summarize_steering(rows, n_layers):
    groups = {}
    for r in rows:
        groups.setdefault((r["target"], r["class"], r["condition"], r["scale"], r["layer"]), []).append(r)
    out = []
    for (t, c, cond, s, l), rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        out.append({"target": t, "class": c, "condition": cond, "scale": s, "layer": l, "depth": l / n_layers,
                    "n": len(rs), "to_gold": float(np.mean([r["to_gold"] for r in rs])),
                    "to_toward": float(np.mean([r["to_toward"] for r in rs]))})
    return out


def cross_model(models: dict, layer_of: dict):
    """Facts where two models give different (graded) answers: is each model's state nearer its own
    answer's place than the other model's answer's place? Places are each model's own, from
    training subjects; tested at each model's onset layer."""
    names = sorted(models)
    out = []
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            A, B = names[x], names[y]
            ra = {r["fact_id"]: (i, r) for i, r in enumerate(models[A][0]) if r["wording"] == "w0" and r["split"] == "test"}
            rb = {r["fact_id"]: (i, r) for i, r in enumerate(models[B][0]) if r["wording"] == "w0" and r["split"] == "test"}
            facts = [f for f in ra if f in rb and ra[f][1].get("given_qid") and rb[f][1].get("given_qid")
                     and ra[f][1]["given_qid"] != rb[f][1]["given_qid"]]
            for me, other, rm, ro in ((A, B, ra, rb), (B, A, rb, ra)):
                recs, S = models[me]
                layer = layer_of[me]
                tr, _ = split_idx(recs)
                Z = Whitener(S[tr, layer])(S[:, layer])
                own_hits = {}
                for f in facts:
                    i, r = rm[f]
                    means = place_means(Z, recs, [k for k in tr if recs[k]["answer_type"] == r["answer_type"]], "given_qid")
                    q_own, q_other = r["given_qid"], ro[f][1]["given_qid"]
                    if q_own in means and q_other in means:
                        own_hits[i] = int(((Z[i] - means[q_own]) ** 2).sum() < ((Z[i] - means[q_other]) ** 2).sum())
                res = subject_bootstrap(own_hits, recs)
                out.append({"model": me, "versus": other, "layer": layer, **res})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(ROOT / "runs" / "main"))
    ap.add_argument("--models", nargs="+", default=["llama8b", "qwen7b", "gemma9b", "llama1b"])
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    prompts = [json.loads(l) for l in open(ROOT / "dataset" / "prompts.jsonl")]
    answers = json.load(open(ROOT / "dataset" / "answers.json"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary, loaded, onsets = {}, {}, {}
    for m in args.models + ["llama8b_untrained"]:
        d = run_dir / m
        if not (d / "ff" / "results.jsonl").exists():
            continue
        ff, S = load_stage(d / "ff", prompts, answers)
        res = {"n_prompts": len(ff),
               "reply_status": {s: sum(r["reply_status"] == s for r in ff) for s in ("answer", "none", "multiple", "hedged")},
               "accuracy_test_w0": float(np.mean([r["correct"] for r in ff if r["split"] == "test" and r["wording"] == "w0"]))}
        if "untrained" in m:
            res["depth_gold"] = depth_profile(ff, S, label="answer_qid")
            summary[m] = res
            continue
        mcq = load_stage(d / "mcq", prompts, answers) if (d / "mcq" / "results.jsonl").exists() else None
        hint = load_stage(d / "hint", prompts, answers, with_states=False)[0] if (d / "hint" / "results.jsonl").exists() else None
        unembed = d / "unembed_answers.npz"
        res["depth"] = depth_profile(ff, S, "given_qid", unembed if unembed.exists() else None, mcq)
        res["text_baseline"] = text_baseline(ff)
        res["depth_gold"] = depth_profile(ff, S, label="answer_qid")
        res["shape"] = shape_test(res["depth"])
        L = onset_layer(ff, S)
        res["onset_layer"], res["onset_depth"] = L, (L / (S.shape[1] - 1) if L else None)
        anat = anatomy(ff, S, L) if L else []
        table = fact_table(ff, mcq[0] if mcq else None, hint)
        res["anatomy"] = {c: sum(a["class"] == c and a["wording"] == "w0" for a in anat)
                          for c in ("right", "in_gold", "in_given", "in_third", "nowhere")}
        res["errors_beyond_confidence"] = errors_beyond_confidence(anat, table)
        if (d / "steer" / "plan.jsonl").exists() and (d / "steer" / "results.jsonl").exists():
            res["steering"] = summarize_steering(steering(d, ff, hint, answers, anat), S.shape[1] - 1)
        with open(out / f"{m}_anatomy.jsonl", "w") as f:
            for a in anat:
                f.write(json.dumps(a) + "\n")
        loaded[m], onsets[m] = (ff, S), L
        summary[m] = res
        print(m, "acc", round(res["accuracy_test_w0"], 3), "onset", res["onset_depth"], "shape", res["shape"], flush=True)
    if "llama8b_untrained" in summary and "llama8b" in loaded:
        # The same eligible questions, grouped by what the trained Llama answered: is that belief
        # structure present in an untrained network's states? (It should not be.)
        ff_u, S_u = load_stage(run_dir / "llama8b_untrained" / "ff", prompts, answers)
        trained = {r["prompt_id"]: r.get("given_qid") for r in loaded["llama8b"][0]}
        for r in ff_u:
            r["trained_given"] = trained.get(r["prompt_id"])
        summary["llama8b_untrained"]["depth_trained_labels"] = depth_profile(ff_u, S_u, label="trained_given")
    if len(loaded) >= 2:
        summary["cross_model"] = cross_model(loaded, onsets)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
