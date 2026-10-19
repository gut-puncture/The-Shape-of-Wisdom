"""Natural rewordings of each question, written by language-model helpers that never see the answer.

    python -m wisdom.rewrites prep     # write dataset/rewrite_jobs/chunk_XX.json for the helpers
    python -m wisdom.rewrites merge    # check every rewrite, keep the good ones -> dataset/rewrites.jsonl

Each fact gets two styles drawn from STYLES so that wording varies in register, syntax and length.
A rewrite is kept only if it names the subject exactly, asks a question, mentions no answer of
that type (or its demonyms/aliases), and is not a near-copy of the plain wording or its sibling.
"""

import json
import random
import re
import sys
from pathlib import Path

from wisdom.grading import normalize
from wisdom.prompts import TEMPLATES

DATASET = Path(__file__).resolve().parent.parent / "dataset"
JOBS = DATASET / "rewrite_jobs"
CHUNK = 200

ASKS = {
    "P27": "which country the person {s} is a citizen of",
    "P17": "which country the place {s} is in",
    "P495": "which country the film {s} comes from",
    "P19": "which city the person {s} was born in",
    "P159": "which city the organisation {s} has its headquarters in",
    "P1412": "which language the person {s} spoke or wrote",
    "P364": "what the original language of the film {s} is",
    "P641": "which sport the person {s} plays or played",
}

STYLES = [
    "a casual chat message in lowercase, possibly with one small typo (not in the name)",
    "a formal written exam question",
    "a curious ten-year-old asking a parent",
    "a pub-quiz host reading out a question",
    "a journalist double-checking a detail with a colleague",
    "someone who half-remembers and asks for confirmation without guessing",
    "a terse search-style question of at most eight words",
    "a teacher prompting a student",
    "an indirect question starting 'Do you know' or 'Could you tell me'",
    "a speaker of English as a second language, slightly unusual grammar",
    "an archivist cataloguing records",
    "a question with the subject placed at the end of the sentence",
    "a polite email-style request in one sentence",
    "a trivia app notification",
]


def prep():
    facts = [json.loads(l) for l in open(DATASET / "facts.jsonl")]
    JOBS.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    jobs = []
    for f in facts:
        s1, s2 = rng.sample(STYLES, 2)
        jobs.append({"fact_id": f["fact_id"], "subject": f["subject_label"],
                     "ask": ASKS[f["relation"]].format(s=f["subject_label"]), "style1": s1, "style2": s2})
    rng.shuffle(jobs)  # mix relations within each chunk
    for i in range(0, len(jobs), CHUNK):
        (JOBS / f"chunk_{i // CHUNK:02d}.json").write_text(json.dumps(jobs[i:i + CHUNK], indent=0, ensure_ascii=False))
    print(f"{len(jobs)} facts in {(len(jobs) + CHUNK - 1) // CHUNK} chunks")


def shingles(text, n=4):
    t = normalize(text)
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def jaccard(a, b):
    A, B = shingles(a), shingles(b)
    return len(A & B) / max(1, len(A | B))


def merge():
    facts = {f["fact_id"]: f for f in (json.loads(l) for l in open(DATASET / "facts.jsonl"))}
    answers = json.load(open(DATASET / "answers.json"))
    forbidden = {}
    for a in answers.values():
        terms = [a["label"], *a.get("aliases", []), *a.get("leak_terms", []), *a.get("demonyms", [])]
        forbidden.setdefault(a["answer_type"], set()).update(normalize(t).strip() for t in terms if len(t) >= 3)
    kept, reasons = [], {}
    for path in sorted(JOBS.glob("out_*.jsonl")):
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                reasons["bad_json"] = reasons.get("bad_json", 0) + 1
                continue
            f = facts.get(r.get("fact_id"))
            if f is None:
                continue
            plain = TEMPLATES[f["relation"]].format(s=f["subject_label"])
            row = {"fact_id": f["fact_id"]}
            for w in ("w1", "w2"):
                q = (r.get(w) or "").strip()
                why = None
                if f["subject_label"] not in q:
                    why = "subject_missing"
                elif "?" not in q:
                    why = "not_a_question"
                elif not 3 <= len(q.split()) <= 40:
                    why = "length"
                elif any(f" {t} " in normalize(q.replace(f["subject_label"], " ")) for t in forbidden[f["answer_type"]]):
                    why = "mentions_an_answer"
                elif jaccard(q, plain) > 0.8 or (w == "w2" and row.get("w1") and jaccard(q, row["w1"]) > 0.8):
                    why = "near_copy"
                if why:
                    reasons[why] = reasons.get(why, 0) + 1
                else:
                    row[w] = q
            kept.append(row)
    with open(DATASET / "rewrites.jsonl", "w") as out:
        for row in kept:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = sum(("w1" in r) + ("w2" in r) for r in kept)
    print(f"kept {n} rewrites for {len(kept)} facts; rejected: {reasons}")


if __name__ == "__main__":
    {"prep": prep, "merge": merge}[sys.argv[1]]()
