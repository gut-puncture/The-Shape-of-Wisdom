"""Load recorded states and fit answer "places": the mean state of all questions whose answer
the model gave was the same entity (belief, not truth), fitted on training subjects only."""

import json
from pathlib import Path

import numpy as np

from wisdom.grading import Grader
from wisdom.prompts import sha

MIN_ITEMS_PER_PLACE = 5


def load_stage(stage_dir: Path, prompts: list, answers: dict, with_states: bool = True):
    """Records for the current prompts (latest result per prompt, matching the current text),
    with graded answers, and their states as one (n, layers + 1, d) float16 array."""
    by_id = {p["prompt_id"]: p for p in prompts}
    latest = {}
    with open(stage_dir / "results.jsonl") as f:
        for line in f:
            r = json.loads(line)
            p = by_id.get(r["prompt_id"])
            if p is not None and r["text_sha"] == sha(p["text"]):
                latest[r["prompt_id"]] = r
    grader = Grader(answers)
    records, index = [], []
    for pid, r in latest.items():
        rec = {**by_id[pid], **r}
        if "reply" in r:
            rec["mentions"] = grader.mentions(r["reply"], rec["answer_type"])
            rec["given_qid"] = grader.given(r["reply"], rec["answer_type"], rec.get("hint_qid"))
        elif "letter_logits" in r:
            # Only an actual letter counts as an answer; otherwise the model did not answer the format.
            letter = r["argmax_token"].strip()
            rec["given_letter"] = letter if letter in ("A", "B", "C", "D") else None
            rec["given_qid"] = rec["options"]["ABCD".index(letter)] if rec["given_letter"] else None
        rec["correct"] = rec.get("given_qid") == rec["answer_qid"]
        records.append(rec)
        index.append((r["shard"], r["row"]))
    if not with_states:
        return records, None
    shards = {s: np.load(stage_dir / "states" / f"{s}.npy", mmap_mode="r") for s in {s for s, _ in index}}
    states = np.stack([shards[s][i] for s, i in index]) if index else np.zeros((0, 0, 0), np.float16)
    return records, states


def fit_places(records, states, layer: int, split="train"):
    """{qid: centroid} at `layer` from items of `split` grouped by the answer the model gave."""
    groups = {}
    for i, r in enumerate(records):
        if r["split"] == split and r.get("given_qid"):
            groups.setdefault(r["given_qid"], []).append(i)
    return {q: states[idx, layer].astype(np.float32).mean(0) for q, idx in groups.items() if len(idx) >= MIN_ITEMS_PER_PLACE}
