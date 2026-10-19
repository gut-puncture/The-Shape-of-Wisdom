"""Run one model over the prompts and record what the answer position holds at every layer.

    python -m wisdom.run --model llama8b --stages ff,mcq,hint,steer --run-dir runs/main

Stages
  ff, hint  greedy-generate a short answer; save the last prompt position's residual
            stream after the embeddings and after every block (before the final norm).
  mcq       one forward pass; save the same states and the four letter logits.
  steer     needs ff: add a "place" vector at one layer and regenerate (see wisdom/steer.py).

Stopping and resuming: kill the process at any time and rerun the same command. Finished
prompts are skipped. A run directory is one task, defined by its invariants (model, exact
weights, precision, what is recorded, answer length); resume refuses if any of them change.
Everything else (batch size, stages, prompts added to the dataset) may change between resumes.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from wisdom.models import MODELS
from wisdom.prompts import sha

ROOT = Path(__file__).resolve().parent.parent
MAX_NEW_TOKENS = 12
DTYPE = torch.bfloat16
CAPTURE = "last prompt position; embeddings output + every decoder block output, pre final norm; float16"
SHARD_SIZE = 512


def load_model(spec, device="cuda"):
    tok = AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(dtype=DTYPE, attn_implementation=spec.attn_implementation)
    if spec.untrained:
        torch.manual_seed(0)
        config = AutoConfig.from_pretrained(spec.repo, revision=spec.revision)
        model = AutoModelForCausalLM.from_config(config, **kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(spec.repo, revision=spec.revision, **kw)
    return model.to(device).eval(), tok


def decoder_layers(model):
    return model.model.layers


class Capture:
    """Records the last position's residual stream during the prompt's (prefill) pass only."""

    def __init__(self, model):
        self.layers = decoder_layers(model)
        self.buf = [None] * (len(self.layers) + 1)
        self.armed = False
        self.handles = [model.get_input_embeddings().register_forward_hook(self._hook(0))]
        self.handles += [layer.register_forward_hook(self._hook(i + 1)) for i, layer in enumerate(self.layers)]

    def _hook(self, slot):
        def fn(module, inputs, output):
            if not self.armed:
                return
            h = output[0] if isinstance(output, tuple) else output
            self.buf[slot] = h[:, -1, :].float().cpu().numpy().astype(np.float16)
            if slot == len(self.layers):
                self.armed = False  # later calls are single-token decoding steps
        return fn

    def take(self):
        states = np.stack(self.buf, axis=1)  # (batch, layers + 1, d)
        self.buf = [None] * len(self.buf)
        return states

    def remove(self):
        for h in self.handles:
            h.remove()


def render(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)


def encode(tok, texts, device):
    return tok([render(tok, t) for t in texts], return_tensors="pt", padding=True, add_special_tokens=False).to(device)


def letter_token_ids(tok):
    ids = {}
    for letter in "ABCD":
        variants = {tuple(tok.encode(v, add_special_tokens=False)) for v in (letter, " " + letter)}
        ids[letter] = sorted({v[0] for v in variants if len(v) == 1})
    return ids


@torch.no_grad()
def generate(model, tok, batch, device, pre_hook=None):
    enc = encode(tok, [p["text"] for p in batch], device)
    out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, output_scores=True,
                         return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
    new = out.sequences[:, enc["input_ids"].shape[1]:]
    first = torch.log_softmax(out.scores[0].float(), dim=-1)
    top2 = first.topk(2, dim=-1).values
    rows = []
    for i, p in enumerate(batch):
        rows.append({"reply": tok.decode(new[i], skip_special_tokens=True),
                     "first_logprob": round(top2[i, 0].item(), 4),
                     "first_margin": round((top2[i, 0] - top2[i, 1]).item(), 4)})
    return rows


@torch.no_grad()
def letter_scores(model, tok, batch, device, letters):
    enc = encode(tok, [p["text"] for p in batch], device)
    logits = model(**enc).logits[:, -1, :].float()
    rows = []
    for i in range(len(batch)):
        scores = {L: max(logits[i, j].item() for j in ids) for L, ids in letters.items()}
        rows.append({"letter_logits": {k: round(v, 4) for k, v in scores.items()},
                     "argmax_token": tok.decode([int(logits[i].argmax())])})
    return rows


class StageStore:
    """Append-only results plus float16 state shards; a crash loses at most the unflushed shard."""

    def __init__(self, path: Path):
        self.path = path
        (path / "states").mkdir(parents=True, exist_ok=True)
        self.results = path / "results.jsonl"
        self.done = set()
        if self.results.exists():
            with open(self.results) as f:
                for line in f:
                    r = json.loads(line)
                    self.done.add((r["prompt_id"], r["text_sha"]))
        self.pending_rows, self.pending_states = [], []

    def todo(self, prompts):
        return [p for p in prompts if (p["prompt_id"], sha(p["text"])) not in self.done]

    def add(self, rows, states=None):
        self.pending_rows += rows
        if states is not None:
            self.pending_states.append(states)
        if len(self.pending_rows) >= SHARD_SIZE:
            self.flush()

    def flush(self):
        if not self.pending_rows:
            return
        shard = f"{int(time.time() * 1000)}"
        if self.pending_states:
            arr = np.concatenate(self.pending_states)
            tmp = self.path / "states" / f"{shard}.tmp.npy"
            np.save(tmp, arr)
            os.replace(tmp, self.path / "states" / f"{shard}.npy")
            for i, r in enumerate(self.pending_rows):
                r["shard"], r["row"] = shard, i
        with open(self.results, "a") as f:
            for r in self.pending_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                self.done.add((r["prompt_id"], r["text_sha"]))
            f.flush()
            os.fsync(f.fileno())
        self.pending_rows, self.pending_states = [], []


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def run_extract(stage, model, tok, prompts, store, batch_size, device):
    cap = Capture(model)
    letters = letter_token_ids(tok) if stage == "mcq" else None
    todo = sorted(store.todo(prompts), key=lambda p: len(p["text"]))
    t0, n = time.time(), 0
    try:
        for batch in batches(todo, batch_size):
            cap.armed = True
            rows = letter_scores(model, tok, batch, device, letters) if stage == "mcq" else generate(model, tok, batch, device)
            states = cap.take()
            for p, r in zip(batch, rows):
                r.update(prompt_id=p["prompt_id"], text_sha=sha(p["text"]))
            store.add(rows, states)
            n += len(batch)
            if n % (batch_size * 10) < batch_size:
                print(f"[{stage}] {n}/{len(todo)}  {n / (time.time() - t0):.1f} prompts/s", flush=True)
        store.flush()
    finally:
        cap.remove()


def save_unembedding(model, tok, answers, path):
    """Unembedding rows of every answer's first tokens, for the 'project out the output words' control."""
    ids = set()
    for a in answers.values():
        for name in [a["label"], *a.get("aliases", [])]:
            for variant in (name, " " + name):
                t = tok.encode(variant, add_special_tokens=False)
                if t:
                    ids.add(t[0])
    ids = sorted(ids)
    W = model.get_output_embeddings().weight[ids].float().cpu().numpy()
    np.savez(path, token_ids=np.array(ids), rows=W.astype(np.float32))


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def check_invariants(run_dir: Path, spec, tok):
    inv = {"model": spec.key, "repo": spec.repo, "revision": spec.revision, "untrained": spec.untrained,
           "dtype": str(DTYPE), "capture": CAPTURE, "max_new_tokens": MAX_NEW_TOKENS,
           "chat_template_sha": sha(tok.chat_template or "")}
    path = run_dir / "run.json"
    if path.exists():
        old = json.loads(path.read_text())
        diff = {k: (old.get(k), v) for k, v in inv.items() if old.get(k) != v}
        if diff:
            raise SystemExit(f"Refusing to resume {run_dir}: invariants changed {diff}. Use a new --run-dir.")
    else:
        path.write_text(json.dumps(inv, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--stages", default="ff,mcq,hint,steer")
    ap.add_argument("--run-dir", default="runs/main")
    ap.add_argument("--prompts", default=str(ROOT / "dataset" / "prompts.jsonl"))
    ap.add_argument("--answers", default=str(ROOT / "dataset" / "answers.json"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit-facts", type=int, default=0, help="pilot: use only the first N facts")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    spec = MODELS[args.model]
    run_dir = Path(args.run_dir) / spec.key
    run_dir.mkdir(parents=True, exist_ok=True)
    prompts = [json.loads(l) for l in open(args.prompts)]
    if args.limit_facts:
        keep = sorted({p["fact_id"] for p in prompts})[:args.limit_facts]
        prompts = [p for p in prompts if p["fact_id"] in set(keep)]
    answers = json.load(open(args.answers))

    model, tok = load_model(spec, args.device)
    check_invariants(run_dir, spec, tok)
    with open(run_dir / "resume_log.jsonl", "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "commit": git_commit(), "args": vars(args)}) + "\n")
    meta = run_dir / "model_meta.json"
    if not meta.exists():
        meta.write_text(json.dumps({"n_layers": len(decoder_layers(model)), "d_model": model.config.hidden_size}))
        save_unembedding(model, tok, answers, run_dir / "unembed_answers.npz")

    for stage in args.stages.split(","):
        if stage == "steer":
            from wisdom.steer import run_steer
            run_steer(model, tok, prompts, answers, run_dir, args.batch_size, args.device)
            continue
        kind = {"ff": "ff", "mcq": "mcq", "hint": "hint"}[stage]
        store = StageStore(run_dir / stage)
        run_extract(stage, model, tok, [p for p in prompts if p["kind"] == kind], store, args.batch_size, args.device)
        print(f"[{stage}] done", flush=True)


if __name__ == "__main__":
    main()
