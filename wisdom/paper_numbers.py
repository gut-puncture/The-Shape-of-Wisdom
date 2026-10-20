"""Every number quoted in the paper, written as LaTeX macros from results/summary.json, so the text can
never drift from the data.   python -m wisdom.paper_numbers --results results --out paper/generated/numbers.tex
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TAGS = {"llama8b": "Llama", "qwen7b": "Qwen", "gemma9b": "Gemma", "llama1b": "LlamaSmall"}


def pct(x):
    return f"{100 * x:.0f}\\%"


def at_depth(rows, d):
    return min(rows, key=lambda r: abs(r["depth"] - d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "paper" / "generated" / "numbers.tex"))
    args = ap.parse_args()
    S = json.loads((Path(args.results) / "summary.json").read_text())
    facts = sum(1 for _ in open(ROOT / "dataset" / "facts.jsonl"))
    prompts = [json.loads(l) for l in open(ROOT / "dataset" / "prompts.jsonl")]
    out = {"NFacts": f"{len({p['fact_id'] for p in prompts}):,}", "NPromptsPerModel": f"{len(prompts):,}",
           "NRewrites": f"{sum(p['kind'] == 'ff' and p['wording'] != 'w0' for p in prompts):,}",
           "NFactsBuilt": f"{facts:,}"}
    for m, tag in TAGS.items():
        if m not in S:
            continue
        s = S[m]
        rows = s["depth"]
        peak = max(rows, key=lambda r: r["acc"]["mean"] or 0)
        out[f"Acc{tag}"] = pct(s["accuracy_test_w0"])
        out[f"Onset{tag}"] = pct(s['onset_depth'])
        out[f"OnsetLayer{tag}"] = str(s["onset_layer"])
        out[f"Layers{tag}"] = str(len(rows))
        out[f"Peak{tag}"] = pct(peak["acc"]["mean"])
        out[f"Shuffled{tag}"] = pct(sum(r["acc_shuffled"] for r in rows) / len(rows))
        out[f"Text{tag}"] = pct(s["text_baseline"]["mean"])
        out[f"StepMse{tag}"] = f"{s['shape']['loo_mse_step']:.3f}"
        out[f"LineMse{tag}"] = f"{s['shape']['loo_mse_line']:.3f}"
        out[f"StepCenter{tag}"] = f"{s['shape']['step_center']:.2f}"
        late = at_depth(rows, 0.9)
        for key, name in [("acc_cross_relation", "XRel"), ("acc_cross_wording", "XWord"), ("acc_no_answer_words", "NoWords"),
                          ("acc_ff_to_mcq", "ToMcq")]:
            if late.get(key) and late[key].get("mean") is not None:
                out[f"{name}{tag}"] = pct(late[key]["mean"])
        e = s.get("errors_beyond_confidence", {})
        if e.get("auc_both"):
            out[f"AucConf{tag}"] = f"{e['auc_confidence']:.2f}"
            out[f"AucGeo{tag}"] = f"{e['auc_geometry']:.2f}"
            out[f"AucBoth{tag}"] = f"{e['auc_both']:.2f}"
            out[f"AucGainLo{tag}"] = f"{e['gain_both_minus_conf'][0]:.2f}"
            out[f"AucGainHi{tag}"] = f"{e['gain_both_minus_conf'][1]:.2f}"
            out[f"ErrN{tag}"] = f"{e['n']:,}"
        st = s.get("steering", [])
        if st:
            def rate(target, cond, key, which):
                agg = {}
                for r in st:
                    if r["target"] == target and r["condition"] == cond and r["scale"] == 1.0:
                        h, n = agg.get(r["layer"], (0, 0))
                        agg[r["layer"]] = (h + r[key] * r["n"], n + r["n"])
                layers = sorted(agg)
                layer = layers[-1] if which == "last" else layers[0]
                return agg[layer][0] / agg[layer][1], agg[layer][1], layer
            g, n, layer = rate("wrong_placed", "gold", "to_gold", "last")
            out[f"PushGold{tag}"], out[f"PushN{tag}"], out[f"PushLayer{tag}"] = pct(g), str(n), str(layer)
            out[f"PushRandom{tag}"] = pct(rate("wrong_placed", "random", "to_gold", "last")[0])
            out[f"PushOther{tag}"] = pct(rate("wrong_placed", "other", "to_toward", "last")[0])
            out[f"PushRightOther{tag}"] = pct(rate("right", "other", "to_toward", "last")[0])
            out[f"PushEarly{tag}"] = pct(rate("wrong_placed", "gold", "to_gold", "first")[0])
            out[f"PushMisled{tag}"] = pct(rate("misled", "gold", "to_gold", "last")[0])
        h = s.get("hint", {})
        if h:
            traj = h["trajectory"]
            peak = max((r for r in traj if r["misled"]["near_gold"] is not None), key=lambda r: r["misled"]["near_gold"])
            out[f"HintPeak{tag}"] = pct(peak["misled"]["near_gold"])
            out[f"HintPeakDepth{tag}"] = f"{peak['depth']:.2f}"
            out[f"HintEnd{tag}"] = pct(traj[-1]["misled"]["near_gold"])
            out[f"HintResistEnd{tag}"] = pct(traj[-1]["resisted"]["near_gold"])
            out[f"HintMisledN{tag}"] = str(peak["misled"]["n"])
        an = s.get("anatomy", {})
        if an:
            on, off = an["on_list"], an["off_list"]
            wrong = sum(on.values()) + sum(off.values())
            out[f"AnatInGiven{tag}"] = pct(on["in_given"] / max(1, sum(on.values())))
            out[f"AnatInGold{tag}"] = pct((on["in_gold"] + off["in_gold"]) / max(1, wrong))
            out[f"AnatWrongN{tag}"] = str(wrong)
    # ranges across the four models, computed rather than hand-picked
    models = [m for m in TAGS if m in S]
    def rng(name, values, fmt=pct):
        out[f"{name}Min"], out[f"{name}Max"] = fmt(min(values)), fmt(max(values))
    def push(m, target, cond, key, which="last"):
        agg = {}
        for r in S[m].get("steering", []):
            if r["target"] == target and r["condition"] == cond and r["scale"] == 1.0:
                h, n, rel = agg.get(r["layer"], (0, 0, []))
                agg[r["layer"]] = (h + r[key] * r["n"], n + r["n"], rel + [r.get("push_rel")])
        layers = sorted(agg)
        layer = layers[-1] if which == "last" else layers[0]
        h, n, rel = agg[layer]
        return h / n, n, (np.median([x for x in rel if x is not None]) if any(x is not None for x in rel) else None)
    rng("PushGold", [push(m, "wrong_placed", "gold", "to_gold")[0] for m in models])
    rng("PushUnplaced", [push(m, "wrong_unplaced", "gold", "to_gold")[0] for m in models])
    rng("PushRandom", [push(m, "wrong_placed", "random", "to_gold")[0] for m in models])
    rng("PushOther", [push(m, "wrong_placed", "other", "to_toward")[0] for m in models])
    rng("PushRightOther", [push(m, "right", "other", "to_toward")[0] for m in models])
    rng("PushEarly", [push(m, "wrong_placed", "gold", "to_gold", "first")[0] for m in models])
    rng("PushN", [push(m, "wrong_placed", "gold", "to_gold")[1] for m in models], fmt=str)
    rng("PushNUnplaced", [push(m, "wrong_unplaced", "gold", "to_gold")[1] for m in models], fmt=str)
    rng("PushRelEarly", [push(m, "wrong_placed", "gold", "to_gold", "first")[2] for m in models])
    rng("PushRelLate", [push(m, "wrong_placed", "gold", "to_gold")[2] for m in models])
    rng("StepRatio", [S[m]["shape"]["loo_mse_line"] / S[m]["shape"]["loo_mse_step"] for m in models], fmt=lambda v: f"{v:.0f}")
    rng("Onset", [S[m]["onset_depth"] for m in models], fmt=lambda v: f"{100 * v:.0f}\\%")
    rng("ErrPos", [S[m]["errors_beyond_confidence"]["positive_rate"] for m in models])
    for m in models:
        tag, rows = TAGS[m], S[m]["depth"]
        L = S[m]["onset_layer"]
        before = next(r for r in rows if r["layer"] == L - 1)
        at = next(r for r in rows if r["layer"] == L)
        out[f"JumpFrom{tag}"], out[f"JumpTo{tag}"] = pct(before["acc"]["mean"]), pct(at["acc"]["mean"])
        out[f"JumpLayerBefore{tag}"] = str(L - 1)
        sp = {r["layer"]: np.log(r["spread"]) for r in rows if r["spread"]}
        total = max(sp.values()) - min(sp.values())
        out[f"TightAtJump{tag}"] = pct(max(0.0, (sp[L - 1] - sp[min(L + 1, max(sp))])) / total)
        gold_late = at_depth(S[m]["depth_gold"], 0.9)["acc"]["mean"]
        given_late = at_depth(rows, 0.9)["acc"]["mean"]
        out[f"GoldLate{tag}"], out[f"GivenLate{tag}"] = pct(gold_late), pct(given_late)
        out[f"ErrPos{tag}"] = pct(S[m]["errors_beyond_confidence"]["positive_rate"])
    # readability by kind of answer: names (countries, cities, languages) pooled, and sports (common nouns)
    def kind_acc(row, sport):
        ks = [v for t, v in row["acc_by_type"].items() if (t == "sport") == sport]
        return sum(v["mean"] * v["n"] for v in ks) / sum(v["n"] for v in ks)
    kinds = {}
    for m in models:
        tag, rows, L = TAGS[m], S[m]["depth"], S[m]["onset_layer"]
        before = next(r for r in rows if r["layer"] == L - 1)
        at = next(r for r in rows if r["layer"] == L)
        late = at_depth(rows, 0.9)
        for sport, name in [(False, "Names"), (True, "Sport")]:
            jump, lt = kind_acc(at, sport) - kind_acc(before, sport), kind_acc(late, sport)
            out[f"{name}Jump{tag}"], out[f"{name}Late{tag}"] = f"{100 * jump:.0f}", pct(lt)
            kinds.setdefault(name, []).append((jump, lt))
    for name, v in kinds.items():
        rng(f"{name}Jump", [j for j, _ in v], fmt=lambda x: f"{100 * x:.0f}")
        rng(f"{name}Late", [l for _, l in v])
    # hand-checked phrases in the text, now computed
    early = []
    for m in models:
        L = S[m]["onset_layer"]
        early += [r["acc"]["mean"] for r in S[m]["depth"] if r["layer"] < L - 1 and r["acc"]["mean"] is not None]
    out["EarlyMin"], out["EarlyMax"] = pct(min(early)), pct(max(early))
    folds = []
    for m in models:
        sp = [r["spread"] for r in S[m]["depth"] if r.get("spread")]
        folds.append(max(sp) / min(sp))
    rng("TightFold", folds, fmt=lambda v: f"{v:.0f}")
    # logistic step: the 10%-90% rise spans 2 ln 9 widths of depth
    rng("StepRise", [2 * np.log(9) * S[m]["shape"]["step_width"] for m in models])
    mc = [r["acc_ff_to_mcq"]["mean"] for r in S["llama8b"]["depth"] if r.get("acc_ff_to_mcq") and r["acc_ff_to_mcq"].get("mean") is not None]
    out["McqPeakLlama"] = pct(max(mc))
    rng("RankRecog", [S[m]["errors_beyond_confidence"]["gold_rank_median_recognised"] for m in models], fmt=lambda v: f"{v:.0f}")
    rng("RankNot", [S[m]["errors_beyond_confidence"]["gold_rank_median_not"] for m in models], fmt=lambda v: f"{v:.0f}")
    rng("Shuffled", [sum(r["acc_shuffled"] for r in S[m]["depth"]) / len(S[m]["depth"]) for m in models])
    rng("Text", [S[m]["text_baseline"]["mean"] for m in models])
    rng("Unclear", [S[m]["reply_status"]["unclear"] / sum(S[m]["reply_status"].values()) for m in models],
        fmt=lambda v: f"{100 * v:.1f}\\%")
    rng("GoldLate", [at_depth(S[m]["depth_gold"], 0.9)["acc"]["mean"] for m in models])
    rng("GivenLate", [at_depth(S[m]["depth"], 0.9)["acc"]["mean"] for m in models])
    # random moves at the deepest layer: how often they change the reply at all
    def last_random_changed(m):
        rs = [r for r in S[m]["steering"] if r["target"] == "wrong_placed" and r["condition"] == "random" and r["scale"] == 1.0]
        last = max(r["layer"] for r in rs)
        rs = [r for r in rs if r["layer"] == last]
        return sum(r["changed"] * r["n"] for r in rs) / sum(r["n"] for r in rs)
    rng("RandomChanged", [last_random_changed(m) for m in models])
    rng("CountryAucRank", [S[m]["errors_beyond_confidence"]["country"]["auc_rank"] for m in models], fmt=lambda v: f"{v:.2f}")
    rng("CountryAucConf", [S[m]["errors_beyond_confidence"]["country"]["auc_conf_mean"] for m in models], fmt=lambda v: f"{v:.2f}")
    size = Path(args.results) / "size_check_llama1b.json"
    if size.exists():
        sc = json.loads(size.read_text())["results"]
        pre = [r for r in sc if r["layer"] < 10]
        out["SizePreSmallMax"] = pct(max(r["right"] for r in pre if r["condition"] == "right" and r["size"] == 0.17))
        out["SizePreLargeMax"] = pct(max(r["right"] for r in pre if r["condition"] == "right" and r["size"] == 0.44))
        out["SizePreRandomMax"] = pct(max(r["right"] for r in pre if r["condition"] == "random"))
        out["SizeOnsetRight"] = pct(next(r["right"] for r in sc if r["layer"] == 10 and r["condition"] == "right"))
        out["SizeOnsetRandom"] = pct(next(r["right"] for r in sc if r["layer"] == 10 and r["condition"] == "random"))
        out["SizePreLargeChangedMax"] = pct(max(r["changed"] for r in pre if r["condition"] == "right" and r["size"] == 0.44))
        out["SizeN"] = str(sc[0]["n"])
    # matched-size moves in the main runs: the full move at the last depth before the onset against the
    # half move at the first depth at or after it
    pre_size, pre_right, post_size, post_right, all_pre = [], [], [], [], []
    for m in models:
        agg = {}
        for r in S[m]["steering"]:
            if r["target"] == "wrong_placed" and r["condition"] == "gold":
                h, n, rel = agg.get((r["layer"], r["scale"]), (0, 0, 0))
                agg[(r["layer"], r["scale"])] = (h + r["to_gold"] * r["n"], n + r["n"], rel + r["push_rel"] * r["n"])
        on = S[m]["onset_layer"]
        all_pre.append(max(agg[(l, sc)][0] / agg[(l, sc)][1] for l, sc in agg if l < on and sc == 1.0))
        before = max(l for l, sc in agg if l < on)
        after = min(l for l, sc in agg if l >= on)
        h, n, rel = agg[(before, 1.0)]
        pre_right.append(h / n); pre_size.append(rel / n)
        h, n, rel = agg[(after, 0.5)]
        post_right.append(h / n); post_size.append(rel / n)
    rng("MatchPreSize", pre_size); rng("MatchPreRight", pre_right)
    rng("MatchPostSize", post_size); rng("MatchPostRight", post_right)
    rng("PushAllPre", all_pre)
    rng("JumpFrom", [next(r for r in S[m]["depth"] if r["layer"] == S[m]["onset_layer"] - 1)["acc"]["mean"] for m in models])
    if all("auc_output_rank" in S[m]["errors_beyond_confidence"] for m in models):
        for m in models:
            out[f"AucOutRank{TAGS[m]}"] = f"{S[m]['errors_beyond_confidence']['auc_output_rank']:.2f}"
        rng("AucOutRank", [S[m]["errors_beyond_confidence"]["auc_output_rank"] for m in models], fmt=lambda v: f"{v:.2f}")
    out["NPromptsAnalysed"] = f"{sum(p['kind'] in ('ff', 'mcq') for p in prompts):,}"
    for m in models:
        e = S[m]["errors_beyond_confidence"]
        out[f"AucRank{TAGS[m]}"] = f"{e['auc_rank']:.2f}"
        out[f"AucConfMean{TAGS[m]}"] = f"{e['auc_conf_mean']:.2f}"
        out[f"RankDiffLo{TAGS[m]}"] = f"{e['rank_minus_conf_mean'][0]:.2f}"
        out[f"RankDiffHi{TAGS[m]}"] = f"{e['rank_minus_conf_mean'][1]:.2f}"
    f2 = lambda v: f"{v:.2f}"
    rng("AucRank", [S[m]["errors_beyond_confidence"]["auc_rank"] for m in models], fmt=f2)
    rng("AucConfMean", [S[m]["errors_beyond_confidence"]["auc_conf_mean"] for m in models], fmt=f2)
    rng("AucGeo", [S[m]["errors_beyond_confidence"]["auc_geometry"] for m in models], fmt=f2)
    rng("AucConf", [S[m]["errors_beyond_confidence"]["auc_confidence"] for m in models], fmt=f2)
    rng("TightAtJump", [float(out[f"TightAtJump{TAGS[m]}"].rstrip("\\%")) / 100 for m in models])
    if "llama8b_untrained" in S:
        u = S["llama8b_untrained"]
        out["UntrainedPeak"] = pct(max(r["acc"]["mean"] for r in u["depth_trained_labels"] if r["acc"]["mean"] is not None))
    cm = S.get("cross_model", [])
    if cm:
        vals = [c["mean"] for c in cm if c.get("mean") is not None]
        out["CrossModelMin"], out["CrossModelMax"] = pct(min(vals)), pct(max(vals))
        out["CrossModelPairs"] = str(len(vals))
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(out.items())))
    print(f"{len(out)} macros -> {path}")


if __name__ == "__main__":
    main()
