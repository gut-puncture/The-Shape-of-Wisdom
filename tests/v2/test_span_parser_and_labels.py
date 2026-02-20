import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sow.v2.span_counterfactuals import label_span_effects
from sow.v2.span_parser import parse_prompt_spans


class TestSpanParserAndLabels(unittest.TestCase):
    def test_parse_prompt_spans(self) -> None:
        text = "Read carefully.\n\nQuestion: What is 2+2?\nA) 3\nB) 4\nC) 5\nD) 6\n\nAnswer: "
        spans = parse_prompt_spans(text)
        labels = [s.label for s in spans]
        self.assertIn("instruction", labels)
        self.assertIn("question_stem", labels)
        self.assertIn("option_A", labels)
        self.assertIn("option_B", labels)

    def test_label_span_effects(self) -> None:
        df = pd.DataFrame.from_records(
            [
                {"effect_delta": 0.2},
                {"effect_delta": -0.2},
                {"effect_delta": 0.0},
            ]
        )
        out = label_span_effects(df)
        self.assertEqual(out["span_label"].tolist(), ["evidence", "distractor", "neutral"])


if __name__ == "__main__":
    unittest.main()
