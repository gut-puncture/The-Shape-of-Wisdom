"""The causal test: push a question's state from where it is toward an answer's place at one layer,
then let the model answer.

The plan is made once and frozen (steer/plan.jsonl + steer/places.npz): the places used, every push,
and a per-push random seed. Resuming replays exactly that plan, so results can never mix pushes
computed from different places. Places are fitted on training subjects; every pushed question is a
test subject (plain wording).

Conditions at each layer (scale = fraction of the distance between the two places):
  gold    toward the gold answer's place                              (errors and hint-misled answers)
  other   toward a different answer's place, drawn once per question  (right answers and errors)
  random  a random direction with the same length as the push it controls for
Targets: wrong_placed (the given wrong answer has a place), wrong_unplaced (it has none; the push
starts from the training mean), right (a sample of right answers), misled (wrong under the hint
although right without it).
"""

import hashlib
import json
import random
import time

import numpy as np
import torch

from wisdom.places import fit_places, load_stage, read_jsonl
from wisdom.prompts import sha
from wisdom.run import batches, decoder_layers, generate

LAYER_FRACTIONS = [0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9]
SCALES = (0.5, 1.0)
N_RIGHT = 200


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


def seed_of(job_id: str) -> int:
    return int(hashlib.sha256(job_id.encode()).hexdigest()[:8], 16)


def make_plan(ff_records, ff_states, hint_records, n_layers, answers):
    layers = sorted({max(1, min(n_layers, round(f * n_layers))) for f in LAYER_FRACTIONS})
    answer_type = {q: a["answer_type"] for q, a in answers.items()}
    train = [i for i, r in enumerate(ff_records) if r["split"] == "train"]
    test_w0 = [r for r in ff_records if r["split"] == "test" and r["wording"] == "w0"]
    clean_right = {r["fact_id"] for r in test_w0 if r["correct"]}
    right = random.Random(0).sample(sorted((r for r in test_w0 if r["correct"]), key=lambda r: r["prompt_id"]),
                                    min(N_RIGHT, len(clean_right)))
    wrong = [r for r in test_w0 if not r["correct"]]
    misled = [r for r in hint_records if r["split"] == "test" and r["fact_id"] in clean_right
              and not r["correct"] and r.get("given_qid")]

    places, jobs = {}, []
    for layer in layers:
        P = fit_places(ff_records, ff_states, layer)
        places[layer] = {**P, "__center__": ff_states[train, layer].astype(np.float32).mean(0)}
        for rec in wrong + right + misled:
            gold, given = rec["answer_qid"], rec.get("given_qid")
            if rec["kind"] == "hint":
                kind = "misled"
            elif rec["correct"]:
                kind = "right"
            else:
                kind = "wrong_placed" if given in P else "wrong_unplaced"
            start = given if given in P else "__center__"
            pool = sorted(q for q, t in answer_type.items() if t == rec["answer_type"] and q not in (gold, given) and q in P)
            other = random.Random(rec["prompt_id"]).choice(pool) if pool else None
            pushes = []
            if kind != "right" and gold in P:
                pushes += [("gold", s, gold) for s in SCALES]
            if kind != "misled" and other:
                pushes += [("other", 1.0, other)]
            controlled = other if kind == "right" else gold  # the random push matches this push's length
            if controlled in P:
                pushes += [("random", s, controlled) for s in ((1.0,) if kind == "right" else SCALES)]
            for cond, scale, toward in pushes:
                jobs.append({"job_id": f"{rec['prompt_id']}|{layer}|{cond}|{scale}", "prompt_id": rec["prompt_id"],
                             "text_sha": sha(rec["text"]), "target": kind, "layer": layer, "condition": cond,
                             "scale": scale, "start": start, "toward": toward})
    return places, jobs


def vector(job, places):
    P = places[job["layer"]]
    v = job["scale"] * (P[job["toward"]] - P[job["start"]])
    if job["condition"] == "random":
        d = np.random.default_rng(seed_of(job["job_id"])).standard_normal(v.shape)
        v = d / np.linalg.norm(d) * np.linalg.norm(v)
    return v.astype(np.float32)


def load_or_make_plan(run_dir, prompts, answers, n_layers):
    out = run_dir / "steer"
    out.mkdir(exist_ok=True)
    if (out / "plan.jsonl").exists():
        z = np.load(out / "places.npz")
        places = {}
        for key in z.files:
            layer, q = key.split(":", 1)
            places.setdefault(int(layer), {})[q] = z[key]
        return places, read_jsonl(out / "plan.jsonl")
    ff_records, ff_states = load_stage(run_dir / "ff", prompts, answers)
    hint_path = run_dir / "hint" / "results.jsonl"
    hint_records = load_stage(run_dir / "hint", prompts, answers, with_states=False)[0] if hint_path.exists() else []
    places, jobs = make_plan(ff_records, ff_states, hint_records, n_layers, answers)
    np.savez(out / "places.npz", **{f"{layer}:{q}": c for layer, P in places.items() for q, c in P.items()})
    tmp = out / "plan.jsonl.tmp"
    with open(tmp, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    tmp.rename(out / "plan.jsonl")  # the plan exists only once it is complete
    return places, jobs


def run_steer(model, tok, prompts, answers, run_dir, batch_size, device, first_ids):
    places, jobs = load_or_make_plan(run_dir, prompts, answers, len(decoder_layers(model)))
    by_id = {p["prompt_id"]: p for p in prompts}
    results = run_dir / "steer" / "results.jsonl"
    done = {r["job_id"] for r in read_jsonl(results)} if results.exists() else set()
    todo = [j for j in jobs if j["job_id"] not in done and j["prompt_id"] in by_id
            and sha(by_id[j["prompt_id"]]["text"]) == j["text_sha"]]
    print(f"[steer] {len(todo)} of {len(jobs)} planned pushes to run", flush=True)

    t0, n = time.time(), 0
    for layer in sorted({j["layer"] for j in todo}):
        push = Push(decoder_layers(model)[layer - 1])  # state index l is the output of block l-1
        try:
            for batch in batches([j for j in todo if j["layer"] == layer], batch_size):
                vecs = np.stack([vector(j, places) for j in batch])
                push.vectors = torch.tensor(vecs, device=device)
                rows = generate(model, tok, [by_id[j["prompt_id"]] for j in batch], device, first_ids)
                with open(results, "a") as f:
                    for j, v, r in zip(batch, vecs, rows):
                        f.write(json.dumps({"job_id": j["job_id"], "push_norm": round(float(np.linalg.norm(v)), 3), **r}) + "\n")
                n += len(batch)
                if n % (batch_size * 10) < batch_size:
                    print(f"[steer] {n}/{len(todo)}  {n / (time.time() - t0):.1f}/s", flush=True)
        finally:
            push.remove()
    print("[steer] done", flush=True)
