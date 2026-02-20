import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sow.v2.trajectory_types import classify_trajectory  # noqa: E402


class TestTrajectoryClassification(unittest.TestCase):
    def test_stable_correct(self) -> None:
        delta = np.asarray([0.2, 0.3, 0.35, 0.4], dtype=float)
        drift = np.asarray([0.1, 0.05, 0.05, 0.0], dtype=float)
        t = classify_trajectory(delta, is_correct=True, drift=drift)
        self.assertEqual(t, "stable_correct")

    def test_unstable_wrong(self) -> None:
        delta = np.asarray([0.1, -0.2, 0.3, -0.1], dtype=float)
        drift = np.asarray([0.3, -0.5, 0.4, -0.4], dtype=float)
        t = classify_trajectory(delta, is_correct=False, drift=drift)
        self.assertEqual(t, "unstable_wrong")


if __name__ == "__main__":
    unittest.main()
