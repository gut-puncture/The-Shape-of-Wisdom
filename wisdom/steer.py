"""The causal test: push a question's state from where it is toward an answer's place at one layer,
then let the model answer.

Places are fitted on training subjects; every pushed question is a test subject. At each layer:
  gold    push toward the gold answer's place (scales 0.5 and 1.0 of the full distance)
  other   push toward a different answer's place (does the model follow wherever it is sent?)
  random  a random direction with the same length as the gold push (is any push of that size enough?)
Targets: wrong answers (gold, other, random), a sample of right answers (other, random), and
hint-misled answers whose plain question was right (gold, random).
"""

import json
import random
import time

import numpy as np
import torch

from wisdom.places import fit_places, load_stage
from wisdom.prompts import sha
from wisdom.run import batches, decoder_layers, generate

LAYER_FRACTIONS = [0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9]
N_CORRECT = 200


class Push:
    """Adds one vector per batch row to a block's output at the last prompt position (prefill only)."""

    def __init__(self, layer_module):
        self.vectors = None
        self.handle = layer_module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if self.vectors is None or h.shape[1] == 1:
            return output
        h[:, -1, :] += self.vectors.to(h.dtype)
        return output

    def remove(self):
        self.handle.remove()


def plan(ff_records, ff_states, hint_records, n_layers, answers):
    """Every (target, layer, condition, scale) to run, with its push vector."""
    rng = random.Random(0)
    gen = np.random.default_rng(0)
    layers = sorted({max(1, min(n_layers, round(f * n_layers))) for f in LAYER_FRACTIONS})
    w0_test = [i for i, r in enumerate(ff_records) if r["split"] == "test" and r["wording"] == "w0"]
    clean_ok = {ff_records[i]["fact_id"] for i in w0_test if ff_records[i]["correct"]}
    answer_type = {q: a["answer_type"] for q, a in answers.items()}

    wrong = [i for i in w0_test if not ff_records[i]["correct"]]
    right = rng.sample([i for i in w0_test if ff_records[i]["correct"]], min(N_CORRECT, len(clean_ok)))
    misled = [r for r in hint_records if r["split"] == "test" and r["fact_id"] in clean_ok and not r["correct"]]

    jobs = []
    for layer in layers:
        places = fit_places(ff_records, ff_states, layer)
        center = ff_states[:, layer].astype(np.float32).mean(0)

        def start(rec):
            return places.get(rec.get("given_qid"), center)

        def add(rec, kind, cond, scale, vec, extra=None):
            jobs.append({"prompt": rec, "target": kind, "layer": layer, "condition": cond, "scale": scale,
                         "vector": vec.astype(np.float32), **(extra or {})})

        for kind, recs, conds in [("wrong", [ff_records[i] for i in wrong], ("gold", "other", "random")),
                                  ("right", [ff_records[i] for i in right], ("other", "random")),
                                  ("misled", misled, ("gold", "random"))]:
            for rec in recs:
                gold_vec = places[rec["answer_qid"]] - start(rec) if rec["answer_qid"] in places else None
                # The other answer is drawn once per question (not per layer), from all answers of the same type.
                pool = sorted(q for q, t in answer_type.items() if t == rec["answer_type"]
                              and q not in (rec["answer_qid"], rec.get("given_qid")))
                other_q = random.Random(rec["prompt_id"]).choice(pool) if pool else None
                other_vec = places[other_q] - start(rec) if other_q in places else None
                # The random control is sized like the push it controls for: gold for errors, other for right answers.
                primary = gold_vec if "gold" in conds else other_vec
                if "gold" in conds and gold_vec is not None:
                    for s in (0.5, 1.0):
                        add(rec, kind, "gold", s, s * gold_vec)
                if "other" in conds and other_vec is not None:
                    add(rec, kind, "other", 1.0, other_vec, {"other_qid": other_q})
                if "random" in conds and primary is not None:
                    d = gen.standard_normal(primary.shape)
                    add(rec, kind, "random", 1.0, d / np.linalg.norm(d) * np.linalg.norm(primary))
    return jobs


def run_steer(model, tok, prompts, answers, run_dir, batch_size, device):
    ff_records, ff_states = load_stage(run_dir / "ff", prompts, answers)
    hint_records, _ = load_stage(run_dir / "hint", prompts, answers) if (run_dir / "hint" / "results.jsonl").exists() else ([], None)
    n_layers = len(decoder_layers(model))
    jobs = plan(ff_records, ff_states, hint_records, n_layers, answers)

    out_dir = run_dir / "steer"
    out_dir.mkdir(exist_ok=True)
    results = out_dir / "results.jsonl"
    key = lambda j: (j["prompt"]["prompt_id"], sha(j["prompt"]["text"]), j["layer"], j["condition"], j["scale"])
    done = set()
    if results.exists():
        for line in open(results):
            r = json.loads(line)
            done.add((r["prompt_id"], r["text_sha"], r["layer"], r["condition"], r["scale"]))
    todo = [j for j in jobs if key(j) not in done]
    print(f"[steer] {len(todo)} of {len(jobs)} pushes to run", flush=True)

    t0, n = time.time(), 0
    groups = {}
    for j in todo:
        groups.setdefault(j["layer"], []).append(j)
    for layer, js in sorted(groups.items()):
        push = Push(decoder_layers(model)[layer - 1])  # state index l is the output of block l-1
        try:
            for batch in batches(js, batch_size):
                push.vectors = torch.tensor(np.stack([j["vector"] for j in batch]), device=device)
                rows = generate(model, tok, [j["prompt"] for j in batch], device)
                with open(results, "a") as f:
                    for j, r in zip(batch, rows):
                        f.write(json.dumps({"prompt_id": j["prompt"]["prompt_id"], "text_sha": sha(j["prompt"]["text"]),
                                            "target": j["target"], "layer": j["layer"], "condition": j["condition"],
                                            "scale": j["scale"], "other_qid": j.get("other_qid"),
                                            "push_norm": round(float(np.linalg.norm(j["vector"])), 3), **r}) + "\n")
                n += len(batch)
                if n % (batch_size * 10) < batch_size:
                    print(f"[steer] {n}/{len(todo)}  {n / (time.time() - t0):.1f}/s", flush=True)
        finally:
            push.remove()
    print("[steer] done", flush=True)
