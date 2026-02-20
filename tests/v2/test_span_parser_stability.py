import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


from sow.v2.span_counterfactuals import delete_span  # noqa: E402
from sow.v2.span_parser import parse_prompt_spans  # noqa: E402


def _label_set(text: str) -> set[str]:
    return {s.label for s in parse_prompt_spans(text)}


class TestSpanParserStability(unittest.TestCase):
    def test_option_styles_are_stably_detected(self) -> None:
        prompts = {
            "paren": "Question: 2+2?\n(A) 4\n(B) 3\n(C) 5\n(D) 6\nAnswer:",
            "bracket": "Question: 2+2?\n[A] 4\n[B] 3\n[C] 5\n[D] 6\nAnswer:",
            "bullet": "Question: 2+2?\n- A) 4\n- B) 3\n- C) 5\n- D) 6\nAnswer:",
            "dotted": "Question: 2+2?\nA. 4\nB. 3\nC. 5\nD. 6\nAnswer:",
            "colon": "Question: 2+2?\nA: 4\nB: 3\nC: 5\nD: 6\nAnswer:",
        }
        expected = {"option_A", "option_B", "option_C", "option_D"}
        for style, text in prompts.items():
            with self.subTest(style=style):
                labels = _label_set(text)
                missing = sorted(expected - labels)
                self.assertFalse(missing, msg=f"missing option spans for style={style}; labels={sorted(labels)}")

    def test_delete_span_clips_to_bounds_and_is_deterministic(self) -> None:
        text = "abcdef"
        self.assertEqual(delete_span(text, start_char=-5, end_char=2), "cdef")
        self.assertEqual(delete_span(text, start_char=2, end_char=100), "ab")
        self.assertEqual(delete_span(text, start_char=4, end_char=4), "abcdef")


if __name__ == "__main__":
    unittest.main()
