"""Load recorded states and fit answer "places": the mean state of all questions whose answer
the model gave was the same entity (belief, not truth), fitted on training subjects only."""

import json
from pathlib import Path

import numpy as np

from wisdom.grading import Grader
from wisdom.prompts import sha

MIN_ITEMS_PER_PLACE = 5


def read_jsonl(path):
    """Records of an append-only log; a torn final line (a crash mid-write) is skipped."""
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def load_stage(stage_dir: Path, prompts: list, answers: dict, with_states: bool = True):
    """Records for the current prompts (latest result per prompt, matching the current text),
    with graded answers, and their states as one (n, layers + 1, d) float16 array."""
    by_id = {p["prompt_id"]: p for p in prompts}
    latest = {}
    for r in read_jsonl(stage_dir / "results.jsonl"):
        p = by_id.get(r["prompt_id"])
        if p is not None and r["text_sha"] == sha(p["text"]):
            latest[r["prompt_id"]] = r
    grader = Grader(answers)
    records, index = [], []
    for pid, r in latest.items():
        rec = {**by_id[pid], **r}
        if "reply" in r:
            rec["given_qid"], rec["reply_status"] = grader.read(r["reply"], rec["answer_type"])
        elif "letter_logits" in r:
            # Only an actual letter counts as an answer; otherwise the model did not answer the format.
            letter = r["argmax_token"].strip()
            rec["given_letter"] = letter if letter in ("A", "B", "C", "D") else None
            rec["given_qid"] = rec["options"]["ABCD".index(letter)] if rec["given_letter"] else None
        rec.setdefault("reply_status", "answer" if rec.get("given_qid") else "unclear")
        rec["answered"] = rec["reply_status"] in ("answer", "other")  # unclear replies are neither right nor wrong
        rec["correct"] = rec.get("given_qid") == rec["answer_qid"]
        records.append(rec)
        index.append((r["shard"], r["row"]))
    if not with_states:
        return records, None
    shards = {s: np.load(stage_dir / "states" / f"{s}.npy", mmap_mode="r") for s in {s for s, _ in index}}
    states = np.stack([shards[s][i] for s, i in index]) if index else np.zeros((0, 0, 0), np.int16)
    return records, bf16_to_f32(states)


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    """Recorded states are bfloat16 bit patterns stored as int16; widen them exactly to float32."""
    return (bits.astype(np.uint16).astype(np.uint32) << 16).view(np.float32)


def fit_places(records, states, layer: int, split="train"):
    """{qid: centroid} at `layer` from items of `split` grouped by the answer the model gave."""
    groups = {}
    for i, r in enumerate(records):
        if r["split"] == split and r.get("given_qid"):
            groups.setdefault(r["given_qid"], []).append(i)
    return {q: states[idx, layer].astype(np.float32).mean(0) for q, idx in groups.items() if len(idx) >= MIN_ITEMS_PER_PLACE}
