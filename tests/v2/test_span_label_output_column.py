import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sow.v2.span_counterfactuals import label_span_effects  # noqa: E402


class TestSpanLabelOutputColumn(unittest.TestCase):
    def test_output_column_does_not_overwrite_span_role(self) -> None:
        df = pd.DataFrame.from_records(
            [
                {"span_role": "question_stem", "effect_delta": 0.2},
                {"span_role": "option_B", "effect_delta": -0.2},
            ]
        )
        out = label_span_effects(df, output_col="span_label")
        self.assertEqual(out["span_role"].tolist(), ["question_stem", "option_B"])
        self.assertEqual(out["span_label"].tolist(), ["evidence", "distractor"])


if __name__ == "__main__":
    unittest.main()
