import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sow.v2.metrics import compute_row_decision_metrics  # noqa: E402


class TestDecisionMetrics(unittest.TestCase):
    def test_delta_boundary_drift(self) -> None:
        row = {
            "layerwise": [
                {"candidate_logits": {"A": 1.0, "B": 0.0, "C": 0.2, "D": 0.1}, "candidate_probs": {"A": 0.5, "B": 0.2, "C": 0.2, "D": 0.1}, "candidate_entropy": 1.2},
                {"candidate_logits": {"A": 1.3, "B": 0.1, "C": 0.3, "D": 0.0}, "candidate_probs": {"A": 0.6, "B": 0.1, "C": 0.2, "D": 0.1}, "candidate_entropy": 1.0},
            ]
        }
        out = compute_row_decision_metrics(row, correct_key="A")
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0].delta, 0.8)
        self.assertAlmostEqual(out[0].boundary, 0.8)
        self.assertAlmostEqual(out[0].drift, out[1].delta - out[0].delta)


if __name__ == "__main__":
    unittest.main()
