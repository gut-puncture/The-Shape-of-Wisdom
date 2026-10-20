"""Is the failure of moves before the onset just their small size? (Llama-3.2-1B; runs on a laptop.)

    python -m wisdom.size_check --device mps

Before the onset, the move from the given answer's centroid to the right answer's centroid is only a few per cent
of the state's length. Here we rescale that move to the sizes that work after the onset (17% and 44% of the median
state length at the layer), with random moves of the same size as controls, at three layers before the onset and
at the onset layer itself. Writes results/size_check_llama1b.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wisdom.grading import Grader
from wisdom.models import MODELS
from wisdom.places import load_stage
from wisdom.run import answer_first_ids, decoder_layers, generate, load_model
from wisdom.steer import Push, load_or_make_plan, seed_of

ROOT = Path(__file__).resolve().parent.parent
CONDITIONS = [(3, "right", 0.17), (3, "right", 0.44), (3, "random", 0.44),
              (6, "right", 0.17), (6, "right", 0.44), (6, "random", 0.44),
              (8, "right", 0.17), (8, "right", 0.44), (8, "random", 0.44),
              (10, "right", 0.17), (10, "random", 0.17)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(ROOT / "runs" / "main" / "llama1b"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=str(ROOT / "results" / "size_check_llama1b.json"))
    args = ap.parse_args()
    run = Path(args.run_dir)
    prompts = [json.loads(l) for l in open(ROOT / "dataset" / "prompts.jsonl")]
    answers = json.load(open(ROOT / "dataset" / "answers.json"))
    by_id = {p["prompt_id"]: p for p in prompts}
    model, tok = load_model(MODELS["llama1b"], args.device)
    first = answer_first_ids(tok, answers)
    places, jobs = load_or_make_plan(run, prompts, answers, len(decoder_layers(model)))
    ff, S = load_stage(run / "ff", prompts, answers)
    test = [i for i, r in enumerate(ff) if r["split"] == "test"]
    # the same wrongly answered test questions as the main experiment (their given answer has a centroid)
    move = {j["prompt_id"]: (j["start"], j["toward"]) for j in jobs if j["target"] == "wrong_placed" and j["condition"] == "gold"}
    targets = sorted(move)
    grader = Grader(answers)
    out = []
    for layer, cond, rel in CONDITIONS:
        norm = float(np.median(np.linalg.norm(S[test, layer], axis=1)))
        P = places[layer]
        push = Push(decoder_layers(model)[layer - 1])  # state index l is the output of block l-1
        right = changed = 0
        try:
            for k in range(0, len(targets), 32):
                batch = targets[k:k + 32]
                vecs = []
                for pid in batch:
                    given, gold = move[pid]
                    v = P[gold] - P[given] if cond == "right" else np.random.default_rng(seed_of(f"{pid}|{layer}")).standard_normal(P[gold].shape)
                    vecs.append(v / np.linalg.norm(v) * rel * norm)
                push.vectors = torch.tensor(np.stack(vecs).astype(np.float32), device=args.device)
                for pid, row in zip(batch, generate(model, tok, [by_id[p] for p in batch], args.device, first)):
                    reply, _ = grader.read(row["reply"], by_id[pid]["answer_type"])
                    right += reply == by_id[pid]["answer_qid"]
                    changed += reply != move[pid][0]
        finally:
            push.remove()
        out.append({"layer": layer, "condition": cond, "size": rel, "n": len(targets),
                    "right": right / len(targets), "changed": changed / len(targets)})
        print(out[-1], flush=True)
    Path(args.out).write_text(json.dumps({"model": "llama1b", "onset_layer": 10, "results": out}, indent=1))


if __name__ == "__main__":
    main()
