"""Read which answer a model's free-form reply commits to.

A reply's core is its first line, cut at the first comma or bracket ("Paris, France" -> "Paris"),
without a leading "the" or a stock preamble ("The answer is ..."). Status:
  answer   the core is exactly one curated name of a candidate answer        -> that answer
  other    a single, plain answer that is not a candidate ("El Salvador")    -> a real wrong answer
  unclear  empty, hedged, refusing, or naming several candidates             -> not an answer at all
"""

import re
import unicodedata

PREAMBLES = ["the answer is", "answer", "it is", "it s", "that would be", "that is", "i believe it is", "i think it is"]
HEDGES = {"not", "no", "never", "cannot", "can", "unsure", "unknown", "uncertain", "maybe", "perhaps", "possibly",
          "might", "probably", "unclear", "sorry", "neither", "nor", "or", "either", "don", "isn", "wasn", "couldn"}
MAX_WORDS = 6  # a longer core is a sentence, not a name


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return " " + re.sub(r"[^a-z0-9]+", " ", text).strip() + " "


class Grader:
    def __init__(self, answers: dict):
        # answers: {qid: {"answer_type": ..., "label": display name, "aliases": [other accepted names]}}
        self.forms = {}
        for qid, a in answers.items():
            names = [a["label"], *a.get("aliases", [])]
            if a["answer_type"] == "language":
                names += [f"{n} language" for n in names]
            for n in names:
                self.forms.setdefault(a["answer_type"], {})[normalize(n).strip()] = qid

    def mentions(self, text: str, answer_type: str):
        line = normalize(text)
        return {q for form, q in self.forms[answer_type].items() if f" {form} " in line}

    def read(self, reply: str, answer_type: str):
        """(committed qid or None, status in {"answer", "other", "unclear"})."""
        lines = [l for l in reply.strip().split("\n") if l.strip()]
        if not lines:
            return None, "unclear"
        line = lines[0]
        words = set(normalize(line).split())
        if words & HEDGES or len(self.mentions(line, answer_type)) > 1:
            return None, "unclear"
        core = normalize(re.split(r"[,(\[;:]| - ", line)[0]).strip()
        for p in PREAMBLES:
            if core.startswith(p + " "):
                core = core[len(p) + 1:]
        core = core[4:] if core.startswith("the ") else core
        if not core:
            return None, "unclear"
        qid = self.forms[answer_type].get(core)
        if qid:
            return qid, "answer"
        return (None, "other") if len(core.split()) <= MAX_WORDS else (None, "unclear")

    def given(self, reply: str, answer_type: str):
        return self.read(reply, answer_type)[0]
