"""Turn facts into model-agnostic prompts.

Three prompt kinds, all ending where the model must start its answer:
  ff   free-form question, in up to three wordings (w0 plain, w1/w2 natural rewrites)
  mcq  the w0 question as four-option multiple choice, in four cyclic option orders
  hint the w0 question with a confident wrong suggestion from the user (the planted
       "knew it, then lost it" control: the same question, nudged toward a wrong place)
Build with:  python -m wisdom.prompts
"""

import hashlib
import json
import random
from pathlib import Path

DATASET = Path(__file__).resolve().parent.parent / "dataset"

TEMPLATES = {  # w0: one plain, human-written question per relation
    "P27": "What country is {s} a citizen of?",
    "P17": "In which country is {s}?",
    "P495": "Which country is the film {s} from?",
    "P19": "In which city was {s} born?",
    "P159": "In which city is {s} headquartered?",
    "P1412": "Which language did {s} speak or write?",
    "P364": "What is the original language of the film {s}?",
    "P641": "Which sport does {s} play?",
}
FF_SUFFIX = "Answer with just the name."
MCQ_SUFFIX = "Answer with the letter only."
LETTERS = "ABCD"
HINT_FRACTION = 0.3


def sha(text: str) -> str:
    """Short content hash of a prompt's text; a result is reused only while its text is unchanged."""
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def stable_unit(*parts) -> float:
    """A deterministic number in [0, 1) from any key; used for splits and sampling."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:12], 16) / 16**12


def split_of(fact) -> str:
    # Split by subject, so held-out evaluation never sees a test subject during fitting.
    return "train" if stable_unit("split", fact["subject_qid"]) < 0.5 else "test"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build(facts, rewrites, answers):
    by_type = {}
    for q, a in answers.items():  # curated display names, e.g. "China" rather than "People's Republic of China"
        by_type.setdefault(a["answer_type"], {})[q] = a["label"]
    out = []
    for f in facts:
        base = {k: f[k] for k in ("fact_id", "answer_type", "relation", "subject_qid", "subject_label", "answer_qid", "band")}
        base["answer_label"] = answers[f["answer_qid"]]["label"]
        base["split"] = split_of(f)
        wordings = {"w0": TEMPLATES[f["relation"]].format(s=f["subject_label"])}
        for w in ("w1", "w2"):
            if rewrites.get(f["fact_id"], {}).get(w):
                wordings[w] = rewrites[f["fact_id"]][w]
        for w, q in wordings.items():
            out.append({**base, "prompt_id": f"{f['fact_id']}:ff:{w}", "kind": "ff", "wording": w, "text": f"{q} {FF_SUFFIX}"})

        rng = random.Random(f"options|{f['fact_id']}")
        others = sorted(q for q in by_type[f["answer_type"]] if q != f["answer_qid"])
        distractors = rng.sample(others, 3)
        options = [f["answer_qid"]] + distractors
        for perm in range(4):
            order = options[perm:] + options[:perm]  # cyclic: the gold option visits every letter
            lines = [f"{LETTERS[i]}. {by_type[f['answer_type']][q]}" for i, q in enumerate(order)]
            text = f"{wordings['w0']}\n" + "\n".join(lines) + f"\n{MCQ_SUFFIX}"
            out.append({**base, "prompt_id": f"{f['fact_id']}:mcq:{perm}", "kind": "mcq", "wording": "w0", "perm": perm,
                        "options": order, "gold_letter": LETTERS[order.index(f["answer_qid"])], "text": text})

        if stable_unit("hint", f["fact_id"]) < HINT_FRACTION:
            wrong = distractors[0]
            text = f"{wordings['w0']} I'm fairly sure it's {by_type[f['answer_type']][wrong]}. {FF_SUFFIX}"
            out.append({**base, "prompt_id": f"{f['fact_id']}:hint:w0", "kind": "hint", "wording": "w0", "hint_qid": wrong, "text": text})
    return out


def main():
    facts = load_jsonl(DATASET / "facts.jsonl")
    rw_path = DATASET / "rewrites.jsonl"
    rewrites = {r["fact_id"]: r for r in load_jsonl(rw_path)} if rw_path.exists() else {}
    answers = json.load(open(DATASET / "answers.json"))
    prompts = build(facts, rewrites, answers)
    with open(DATASET / "prompts.jsonl", "w") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    kinds = {}
    for p in prompts:
        kinds[p["kind"], p["wording"]] = kinds.get((p["kind"], p["wording"]), 0) + 1
    print(f"{len(prompts)} prompts:", dict(sorted(kinds.items())))


if __name__ == "__main__":
    main()
