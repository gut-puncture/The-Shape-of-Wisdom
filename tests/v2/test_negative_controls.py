import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sow.v2.causal.span_deletion import run_negative_controls, summarize_span_deletion_effects  # noqa: E402


class TestNegativeControls(unittest.TestCase):
    def test_controls_and_summary(self) -> None:
        df = pd.DataFrame.from_records(
            [
                {"span_label": "evidence", "effect_delta": 0.2},
                {"span_label": "distractor", "effect_delta": -0.2},
                {"span_label": "neutral", "effect_delta": 0.0},
            ]
        )
        s = summarize_span_deletion_effects(df)
        self.assertEqual(set(s["span_label"].tolist()), {"evidence", "distractor", "neutral"})
        c = run_negative_controls(df, seed=1)
        self.assertEqual(set(c["control"].tolist()), {"observed", "shuffled", "sign_flipped"})


if __name__ == "__main__":
    unittest.main()
