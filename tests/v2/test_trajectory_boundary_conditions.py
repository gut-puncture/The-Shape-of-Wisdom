import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


from sow.v2.trajectory_types import classify_trajectory  # noqa: E402


class TestTrajectoryBoundaryConditions(unittest.TestCase):
    def test_exact_tail_floor_is_stable(self) -> None:
        delta = np.asarray([0.5] * 10, dtype=float)
        drift = np.asarray([0.3] * 10, dtype=float)
        t = classify_trajectory(delta, is_correct=True, drift=drift, tail_len=8, max_late_flip_count=0, min_abs_delta_tail_floor=0.5)
        self.assertEqual(t, "stable_correct")

    def test_below_tail_floor_is_unstable(self) -> None:
        delta = np.asarray([0.49] * 10, dtype=float)
        drift = np.asarray([0.01] * 10, dtype=float)
        t = classify_trajectory(delta, is_correct=True, drift=drift, tail_len=8, max_late_flip_count=0, min_abs_delta_tail_floor=0.5)
        self.assertEqual(t, "unstable_correct")

    def test_late_flip_in_tail_is_unstable(self) -> None:
        delta = np.asarray([0.9, 0.9, 0.9, 0.9, 0.9, -0.9, -0.9, -0.9, -0.9, -0.9], dtype=float)
        drift = np.asarray([0.0] * 10, dtype=float)
        t = classify_trajectory(delta, is_correct=False, drift=drift, tail_len=8, max_late_flip_count=0, min_abs_delta_tail_floor=0.5)
        self.assertEqual(t, "unstable_wrong")

    def test_high_drift_without_late_flip_and_with_floor_remains_stable(self) -> None:
        delta = np.asarray([0.6, 1.2, 0.7, 1.4, 0.9, 1.3, 0.8, 1.1, 0.7, 1.2], dtype=float)
        drift = np.asarray([0.6, -0.5, 0.7, -0.5, 0.4, -0.5, 0.3, -0.4, 0.5, 0.0], dtype=float)
        t = classify_trajectory(delta, is_correct=True, drift=drift, tail_len=8, max_late_flip_count=0, min_abs_delta_tail_floor=0.5)
        self.assertEqual(t, "stable_correct")


if __name__ == "__main__":
    unittest.main()
