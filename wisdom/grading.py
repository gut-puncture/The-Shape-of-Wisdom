"""Read which answer entity a model's free-form reply names."""

import re
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return " " + re.sub(r"[^a-z0-9]+", " ", text).strip() + " "


class Grader:
    """Maps a reply to the answer entity (of the expected type) it names first."""

    def __init__(self, answers: dict):
        # answers: {qid: {"answer_type": ..., "label": ..., "aliases": [...]}}
        self.names = {}
        for qid, a in answers.items():
            forms = {normalize(n).strip() for n in [a["label"], *a.get("aliases", [])]}
            forms = {f for f in forms if len(f) >= 2}
            self.names.setdefault(a["answer_type"], []).append((qid, forms))

    def given(self, reply: str, answer_type: str):
        """The qid named earliest in the first line of the reply, or None."""
        first_line = normalize(reply.strip().split("\n")[0] if reply.strip() else "")
        best = None
        for qid, forms in self.names[answer_type]:
            for form in forms:
                i = first_line.find(f" {form} ")
                # Earliest mention wins; at the same position prefer the longer name ("New Delhi" over "Delhi").
                if i >= 0 and (best is None or (i, -len(form)) < (best[0], -best[1])):
                    best = (i, len(form), qid)
        return best[2] if best else None
