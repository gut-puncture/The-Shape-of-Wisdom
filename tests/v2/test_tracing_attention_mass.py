import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sow.v2.tracing.decomposition import attention_mass_by_span_per_layer  # noqa: E402


class TestAttentionMassBySpan(unittest.TestCase):
    def test_per_layer_mass(self) -> None:
        # layers=2, batch=1, heads=1, q=3, k=4
        layer0 = np.zeros((1, 1, 3, 4), dtype=np.float64)
        layer1 = np.zeros((1, 1, 3, 4), dtype=np.float64)

        # last query attention distribution
        layer0[0, 0, -1, :] = np.asarray([0.1, 0.2, 0.3, 0.4])
        layer1[0, 0, -1, :] = np.asarray([0.5, 0.2, 0.2, 0.1])

        spans = {
            "instruction": [0],
            "question_stem": [1, 2],
            "option_A": [3],
        }

        masses = attention_mass_by_span_per_layer([layer0, layer1], span_token_indices=spans)
        self.assertEqual(len(masses), 2)

        self.assertAlmostEqual(masses[0]["instruction"], 0.1)
        self.assertAlmostEqual(masses[0]["question_stem"], 0.5)
        self.assertAlmostEqual(masses[0]["option_A"], 0.4)

        self.assertAlmostEqual(masses[1]["instruction"], 0.5)
        self.assertAlmostEqual(masses[1]["question_stem"], 0.4)
        self.assertAlmostEqual(masses[1]["option_A"], 0.1)


if __name__ == "__main__":
    unittest.main()
