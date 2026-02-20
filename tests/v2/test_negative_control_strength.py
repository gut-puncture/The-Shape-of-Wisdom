import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


from sow.v2.causal.span_deletion import run_negative_controls  # noqa: E402


class TestNegativeControlStrength(unittest.TestCase):
    def test_shuffled_control_is_not_identical_to_observed_effect(self) -> None:
        df = pd.DataFrame.from_records(
            [
                {"span_label": "evidence", "effect_delta": 1.00},
                {"span_label": "evidence", "effect_delta": 0.85},
                {"span_label": "distractor", "effect_delta": 0.70},
                {"span_label": "neutral", "effect_delta": 0.55},
            ]
        )
        out = run_negative_controls(df, seed=7)
        observed = float(out.loc[out["control"] == "observed", "mean_effect_delta"].iloc[0])
        shuffled = float(out.loc[out["control"] == "shuffled", "mean_effect_delta"].iloc[0])
        self.assertNotAlmostEqual(
            observed,
            shuffled,
            delta=1e-12,
            msg=f"negative control should not be mean-invariant; observed={observed} shuffled={shuffled}",
        )


if __name__ == "__main__":
    unittest.main()
