import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sow.v2.span_parser import parse_prompt_spans  # noqa: E402


class TestSpanParserNonOverlapping(unittest.TestCase):
    def test_option_spans_do_not_overlap_post_options(self) -> None:
        text = (
            "Read carefully.\n"
            "Question: Pick one.\n"
            "A) alpha\n"
            "B) beta\n"
            "C) gamma\n"
            "D) delta\n\n"
            "Answer:"
        )
        spans = parse_prompt_spans(text)
        self.assertTrue(spans, msg="expected non-empty span parse")

        by_label = {s.label: s for s in spans}
        self.assertIn("option_D", by_label)
        self.assertIn("post_options", by_label)

        opt_d = by_label["option_D"]
        post = by_label["post_options"]
        self.assertLessEqual(
            opt_d.end_char,
            post.start_char,
            msg=(
                "option_D must end before post_options begins; overlapping spans contaminate "
                "span deletion effects and label stability"
            ),
        )

    def test_spans_are_monotonic_non_overlapping(self) -> None:
        text = (
            "Intro\n"
            "Question: test?\n"
            "A. one\n"
            "B. two\n"
            "C. three\n"
            "D. four\n"
            "Rationale follows."
        )
        spans = parse_prompt_spans(text)
        ordered = sorted(spans, key=lambda s: (s.start_char, s.end_char))
        for i in range(1, len(ordered)):
            prev = ordered[i - 1]
            cur = ordered[i]
            self.assertLessEqual(
                prev.end_char,
                cur.start_char,
                msg=f"overlap detected: {prev.label}[{prev.start_char},{prev.end_char}) and {cur.label}[{cur.start_char},{cur.end_char})",
            )


if __name__ == "__main__":
    unittest.main()
