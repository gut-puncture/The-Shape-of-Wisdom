"""Run one model over the prompts and record what the answer position holds at every layer.

    python -m wisdom.run --model llama8b --stages ff,mcq,hint,steer --run-dir runs/main

Stages
  ff, hint  greedy-generate a short answer; save the last prompt position's residual
            stream after the embeddings and after every block (before the final norm).
  mcq       one forward pass; save the same states and the four letter logits.
  steer     needs ff: add a "place" vector at one layer and regenerate (see wisdom/steer.py).

Stopping and resuming: kill the process at any time and rerun the same command. Finished
prompts are skipped. A run directory is one task, defined by its invariants (model, exact
weights, precision, attention kernel, what is recorded, answer length, chat template, the answer
list, library versions); resume refuses if any of them change.
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
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from wisdom.models import MODELS
from wisdom.places import read_jsonl
from wisdom.prompts import sha, stable_unit

ROOT = Path(__file__).resolve().parent.parent
MAX_NEW_TOKENS = 12
DTYPE = torch.bfloat16
CAPTURE = "last prompt position; input to block 0 + output of every block, pre final norm; bfloat16 bits"
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
        model = AutoModelForCausalLM.from_pretrained(spec.repo, revision=spec.revision, device_map=device, **kw)
    return model.to(device).eval(), tok


def decoder_layers(model):
    return model.model.layers


class Capture:
    """Records the last position's residual stream at every block boundary during the prompt's
    (prefill) pass only: slot 0 is the input to the first block, slot i the output of block i.
    Hooking block boundaries (not the embedding module) makes this exact for any architecture,
    e.g. Gemma scales its embeddings after the embedding module."""

    def __init__(self, model):
        self.layers = decoder_layers(model)
        self.buf = [None] * (len(self.layers) + 1)
        self.armed = False
        self.handles = [self.layers[0].register_forward_pre_hook(self._pre_hook)]
        self.handles += [layer.register_forward_hook(self._hook(i + 1)) for i, layer in enumerate(self.layers)]

    def _pre_hook(self, module, args, kwargs=None):
        if self.armed:
            self.buf[0] = args[0][:, -1, :].detach().float()

    def _hook(self, slot):
        def fn(module, inputs, output):
            if not self.armed:
                return
            h = output[0] if isinstance(output, tuple) else output
            self.buf[slot] = h[:, -1, :].detach().float()
            if slot == len(self.layers):
                self.armed = False  # later calls are single-token decoding steps
        return fn

    def take(self):
        # The model computes in bfloat16; store exactly those bits (as int16) so nothing overflows or rounds.
        states = torch.stack(self.buf, dim=1).to(torch.bfloat16).view(torch.int16).cpu().numpy()
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


def answer_first_ids(tok, answers):
    """{answer_type: {qid: first-token ids of its names}}: the tokens that can start each answer."""
    out = {}
    for qid, a in answers.items():
        ids = set()
        for name in [a["label"], *a.get("aliases", [])]:
            for variant in (name, " " + name):
                t = tok.encode(variant, add_special_tokens=False)
                if t:
                    ids.add(t[0])
        out.setdefault(a["answer_type"], {})[qid] = sorted(ids)
    return out


@torch.no_grad()
def generate(model, tok, batch, device, first_ids):
    """Greedy answer plus confidence about the answer itself: the log-prob of every generated token,
    and, at the first answer position, the best logit among each candidate answer's first tokens."""
    enc = encode(tok, [p["text"] for p in batch], device)
    out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, output_scores=True,
                         return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
    new = out.sequences[:, enc["input_ids"].shape[1]:]
    steps = len(out.scores)
    chosen = torch.stack([torch.log_softmax(out.scores[k].float(), dim=-1).gather(-1, new[:, k:k + 1])[:, 0]
                          for k in range(steps)], dim=1).cpu()  # log-prob of each generated token
    first = torch.log_softmax(out.scores[0].float(), dim=-1).cpu()
    eos = model.generation_config.eos_token_id
    stop_ids = set(eos if isinstance(eos, list) else [eos]) | {tok.eos_token_id}
    rows = []
    for i, p in enumerate(batch):
        ans = {q: round(max(first[i, j].item() for j in ids), 4) for q, ids in first_ids[p["answer_type"]].items() if ids}
        ends = [k for k, t in enumerate(new[i].tolist()) if t in stop_ids]
        n = min(ends[0] + 1 if ends else new.shape[1], chosen.shape[1])
        rows.append({"reply": tok.decode(new[i], skip_special_tokens=True),
                     "token_logprobs": [round(x, 4) for x in chosen[i, :n].tolist()],
                     "answer_logprobs": ans})
    return rows


@torch.no_grad()
def letter_scores(model, tok, batch, device, letters):
    enc = encode(tok, [p["text"] for p in batch], device)
    logits = model(**enc, logits_to_keep=1, use_cache=False).logits[:, -1, :].float()
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
            for r in read_jsonl(self.results):  # a torn final line from a crash is simply redone
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
            with open(tmp, "wb") as fh:
                np.save(fh, arr)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path / "states" / f"{shard}.npy")  # data is durable before results point to it
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


def run_extract(stage, model, tok, prompts, store, batch_size, device, first_ids):
    cap = Capture(model)
    letters = letter_token_ids(tok) if stage == "mcq" else None
    todo = sorted(store.todo(prompts), key=lambda p: len(p["text"]))
    t0, n = time.time(), 0
    try:
        for batch in batches(todo, batch_size):
            cap.armed = True
            rows = letter_scores(model, tok, batch, device, letters) if stage == "mcq" else generate(model, tok, batch, device, first_ids)
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
    """Unembedding rows of every token of every answer name (with and without a leading space), and the
    final norm's weight, for the control that removes the answer words' output directions."""
    ids = set()
    for a in answers.values():
        for name in [a["label"], *a.get("aliases", [])]:
            for variant in (name, " " + name):
                ids.update(tok.encode(variant, add_special_tokens=False))
    ids = sorted(ids)
    W = model.get_output_embeddings().weight[ids].detach().float().cpu().numpy()
    # The final norm's per-dimension scale, measured rather than assumed (Gemma multiplies by 1 + weight):
    # the norm applied to a vector of ones (RMS exactly 1) returns that scale.
    norm = model.model.norm
    ones = torch.ones(1, model.config.hidden_size, device=norm.weight.device, dtype=torch.float32)
    norm_w = norm.float()(ones)[0].detach().cpu().numpy()
    norm.to(DTYPE)
    np.savez(path, token_ids=np.array(ids), rows=W.astype(np.float32), final_norm_scale=norm_w)


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def check_invariants(run_dir: Path, spec, tok, answers_sha):
    inv = {"model": spec.key, "repo": spec.repo, "revision": spec.revision, "untrained": spec.untrained,
           "attn_implementation": spec.attn_implementation,
           "dtype": str(DTYPE), "capture": CAPTURE, "max_new_tokens": MAX_NEW_TOKENS,
           "chat_template_sha": sha(tok.chat_template or ""), "answers_sha": answers_sha,
           "torch": torch.__version__.split("+")[0], "transformers": transformers.__version__}
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
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit-facts", type=int, default=0, help="pilot: a deterministic sample of N facts")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    spec = MODELS[args.model]
    run_dir = Path(args.run_dir) / spec.key
    run_dir.mkdir(parents=True, exist_ok=True)
    prompts = [json.loads(l) for l in open(args.prompts)]
    if args.limit_facts:
        facts = sorted({p["fact_id"] for p in prompts}, key=lambda f: stable_unit("pilot", f))
        keep = set(facts[:args.limit_facts])  # a deterministic sample spread over every type and relation
        prompts = [p for p in prompts if p["fact_id"] in keep]
    answers = json.load(open(args.answers))

    model, tok = load_model(spec, args.device)
    check_invariants(run_dir, spec, tok, sha(json.dumps(answers, sort_keys=True)))
    with open(run_dir / "resume_log.jsonl", "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "commit": git_commit(), "args": vars(args)}) + "\n")
    meta = run_dir / "model_meta.json"
    if not meta.exists():  # written last, so a crash here simply repeats this step on resume
        save_unembedding(model, tok, answers, run_dir / "unembed_answers.npz")
        meta.write_text(json.dumps({"n_layers": len(decoder_layers(model)), "d_model": model.config.hidden_size}))

    for stage in args.stages.split(","):
        if stage == "steer":
            from wisdom.steer import run_steer
            run_steer(model, tok, prompts, answers, run_dir, args.batch_size, args.device, answer_first_ids(tok, answers))
            continue
        kind = {"ff": "ff", "mcq": "mcq", "hint": "hint"}[stage]
        store = StageStore(run_dir / stage)
        run_extract(stage, model, tok, [p for p in prompts if p["kind"] == kind], store, args.batch_size, args.device,
                    answer_first_ids(tok, answers))
        print(f"[{stage}] done", flush=True)


if __name__ == "__main__":
    main()
