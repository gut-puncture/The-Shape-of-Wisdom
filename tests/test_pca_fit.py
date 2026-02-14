import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


from sow.pca.fit_pca import _canonicalize_component_signs, _basis_hash, fit_pca_basis_from_hidden  # noqa: E402


class TestPcaFit(unittest.TestCase):
    def test_canonicalize_component_signs(self) -> None:
        comps = np.array(
            [
                [-1.0, 0.1, 0.2],  # max abs at idx 0 -> should flip
                [0.01, -2.0, 0.3],  # max abs at idx 1 -> should flip
                [0.0, 0.0, 5.0],  # already positive
            ],
            dtype=np.float32,
        )
        out = _canonicalize_component_signs(comps)
        self.assertTrue(np.allclose(out[0], np.array([1.0, -0.1, -0.2], dtype=np.float32)))
        self.assertTrue(np.allclose(out[1], np.array([-0.01, 2.0, -0.3], dtype=np.float32)))
        self.assertTrue(np.allclose(out[2], comps[2]))

        # Idempotent.
        out2 = _canonicalize_component_signs(out)
        self.assertTrue(np.allclose(out2, out))

    def test_basis_hash_stable(self) -> None:
        mean = np.arange(8, dtype=np.float32)
        comps = np.eye(2, 8, dtype=np.float32)
        h1 = _basis_hash(mean=mean, components=comps)
        h2 = _basis_hash(mean=mean.copy(), components=comps.copy())
        self.assertEqual(h1, h2)

    def test_fit_pca_basis_reproducible_for_fixed_seed(self) -> None:
        rng = np.random.RandomState(123)
        hidden = rng.randn(5, 3, 8).astype(np.float32)  # 15 pooled rows, 8 dims
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "hidden.npz"
            np.savez(p, hidden=hidden.astype(np.float16))
            fit1 = fit_pca_basis_from_hidden(hidden_npz=p, n_components=3, seed=7)
            fit2 = fit_pca_basis_from_hidden(hidden_npz=p, n_components=3, seed=7)
            self.assertEqual(fit1["basis_hash"], fit2["basis_hash"])

