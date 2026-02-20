import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sow.v2.tracing.decomposition import component_scalar  # noqa: E402


class TestDecisionAlignedComponentScalar(unittest.TestCase):
    def test_directional_projection_when_direction_provided(self) -> None:
        update = np.asarray([1.0, -2.0, 3.0], dtype=np.float64)
        direction = np.asarray([0.5, 0.0, 0.5], dtype=np.float64)
        self.assertAlmostEqual(component_scalar(update, decision_direction=direction), 2.0)

    def test_norm_fallback_without_direction(self) -> None:
        update = np.asarray([3.0, 4.0], dtype=np.float64)
        self.assertAlmostEqual(component_scalar(update), 5.0)


if __name__ == "__main__":
    unittest.main()
