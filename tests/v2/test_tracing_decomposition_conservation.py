import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sow.v2.tracing.decomposition import drift_reconstruction_quality, drift_series_from_deltas  # noqa: E402


class TestTracingDecomposition(unittest.TestCase):
    def test_drift_series_from_deltas(self) -> None:
        d = np.asarray([0.2, 0.5, -0.1], dtype=np.float64)
        g = drift_series_from_deltas(d)
        self.assertEqual(g.tolist(), [0.3, -0.6, 0.0])

    def test_reconstruction_quality(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.normal(size=200)
        m = rng.normal(size=200)
        y = 0.7 * a + 0.2 * m + 0.05
        q = drift_reconstruction_quality(observed_drift=y, attn_scalar=a, mlp_scalar=m)
        self.assertGreater(q["r2"], 0.98)


if __name__ == "__main__":
    unittest.main()
