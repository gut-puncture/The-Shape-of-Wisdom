import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


from sow.v2.tracing.decomposition import drift_reconstruction_quality, drift_series_from_deltas  # noqa: E402


class TestTracingDecompositionEdgeInvariants(unittest.TestCase):
    def test_mismatched_lengths_raise(self) -> None:
        with self.assertRaises(ValueError):
            drift_reconstruction_quality(
                observed_drift=np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
                attn_scalar=np.asarray([0.1, 0.2], dtype=np.float64),
                mlp_scalar=np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
            )

    def test_constant_series_perfect_fit_reports_r2_one(self) -> None:
        y = np.asarray([0.2] * 6, dtype=np.float64)
        a = np.asarray([0.1] * 6, dtype=np.float64)
        m = np.asarray([0.1] * 6, dtype=np.float64)
        q = drift_reconstruction_quality(observed_drift=y, attn_scalar=a, mlp_scalar=m)
        self.assertAlmostEqual(q["r2"], 1.0, places=6, msg=f"constant perfect fit should report r2=1.0; got={q}")

    def test_drift_series_telescopes_to_delta_change(self) -> None:
        d = np.asarray([0.25, 0.10, -0.05, 0.20], dtype=np.float64)
        g = drift_series_from_deltas(d)
        self.assertAlmostEqual(float(np.sum(g[:-1])), float(d[-1] - d[0]), places=12)


if __name__ == "__main__":
    unittest.main()
