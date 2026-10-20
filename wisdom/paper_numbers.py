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
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(out.items())))
    print(f"{len(out)} macros -> {path}")


if __name__ == "__main__":
    main()
