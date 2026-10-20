"""Every number quoted in the paper, written as LaTeX macros from results/summary.json, so the text can
never drift from the data.   python -m wisdom.paper_numbers --results results --out paper/generated/numbers.tex
"""

import argparse
import json
from pathlib import Path

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
        out[f"Onset{tag}"] = f"{s['onset_depth']:.2f}"
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
