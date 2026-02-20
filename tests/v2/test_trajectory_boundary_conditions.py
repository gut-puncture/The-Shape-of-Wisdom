import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


from sow.v2.trajectory_types import classify_trajectory  # noqa: E402


class TestTrajectoryBoundaryConditions(unittest.TestCase):
    def test_exact_drift_threshold_is_stable(self) -> None:
        delta = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=float)
        drift = np.asarray([0.15] * 8, dtype=float)
        t = classify_trajectory(delta, is_correct=True, drift=drift)
        self.assertEqual(t, "stable_correct")

    def test_above_drift_threshold_is_unstable(self) -> None:
        delta = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=float)
        drift = np.asarray([0.150001] * 8, dtype=float)
        t = classify_trajectory(delta, is_correct=True, drift=drift)
        self.assertEqual(t, "unstable_correct")

    def test_zero_crossings_do_not_create_spurious_instability(self) -> None:
        # Crossing exactly through zero should not be treated as a sign flip.
        delta = np.asarray([0.25, 0.0, 0.20, 0.30], dtype=float)
        drift = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=float)
        t = classify_trajectory(delta, is_correct=True, drift=drift)
        self.assertEqual(
            t,
            "stable_correct",
            msg=f"zero crossings should not force unstable classification; delta={delta.tolist()} drift={drift.tolist()}",
        )


if __name__ == "__main__":
    unittest.main()
