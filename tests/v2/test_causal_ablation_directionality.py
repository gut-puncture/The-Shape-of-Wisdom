import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sow.v2.causal.ablations import run_component_ablation  # noqa: E402


class TestCausalAblationDirectionality(unittest.TestCase):
    def test_ablation_changes_final_delta(self) -> None:
        rows = []
        for li in range(4):
            rows.append(
                {
                    "model_id": "m",
                    "prompt_uid": "u",
                    "layer_index": li,
                    "delta": 0.1 + 0.2 * li,
                    "drift": 0.2,
                    "s_attn": 0.15,
                    "s_mlp": 0.05,
                }
            )
        df = pd.DataFrame.from_records(rows)
        out = run_component_ablation(df, component="attention", target_layers=[1, 2, 3])
        self.assertEqual(out.shape[0], 1)
        self.assertLess(float(out.iloc[0]["delta_final_ablate"]), float(out.iloc[0]["delta_final_base"]))


if __name__ == "__main__":
    unittest.main()
