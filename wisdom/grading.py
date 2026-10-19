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

    def mentions(self, reply: str, answer_type: str):
        """Answer entities named in the first line of the reply, in order of first mention.
        At the same position the longer name wins ("New Delhi" over "Delhi")."""
        first_line = normalize(reply.strip().split("\n")[0] if reply.strip() else "")
        found = {}
        for qid, forms in self.names[answer_type]:
            for form in forms:
                i = first_line.find(f" {form} ")
                if i >= 0 and (qid not in found or (i, -len(form)) < found[qid]):
                    found[qid] = (i, -len(form))
        spans = sorted(found.items(), key=lambda kv: kv[1])
        # drop a name that is only a piece of a longer name found at the same place
        out, taken = [], []
        for qid, (i, neg_len) in spans:
            if any(i >= j and i - neg_len <= j + L for j, L in taken):
                continue
            out.append(qid)
            taken.append((i, -neg_len))
        return out

    def given(self, reply: str, answer_type: str, hint_qid=None):
        """The answer the reply commits to. When the user suggested `hint_qid` and the reply names
        both it and another answer ("Germany? No, it's France"), the other answer is the commitment."""
        named = self.mentions(reply, answer_type)
        if hint_qid is not None:
            others = [q for q in named if q != hint_qid]
            if others:
                return others[0]
        return named[0] if named else None
