import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sow.v2.span_counterfactuals import completed_span_keys_for_mode  # noqa: E402


class TestSpanCounterfactualResumeMode(unittest.TestCase):
    def test_model_mode_ignores_proxy_rows(self) -> None:
        df = pd.DataFrame.from_records(
            [
                {
                    "model_id": "m",
                    "prompt_uid": "u",
                    "span_id": "s1",
                    "counterfactual_mode": "proxy",
                },
                {
                    "model_id": "m",
                    "prompt_uid": "u",
                    "span_id": "s2",
                    "counterfactual_mode": "model",
                },
            ]
        )
        keys = completed_span_keys_for_mode(df, mode="model")
        self.assertEqual(keys, {("m", "u", "s2")})

    def test_proxy_mode_uses_proxy_rows(self) -> None:
        df = pd.DataFrame.from_records(
            [
                {
                    "model_id": "m",
                    "prompt_uid": "u",
                    "span_id": "s1",
                    "counterfactual_mode": "proxy",
                }
            ]
        )
        keys = completed_span_keys_for_mode(df, mode="proxy")
        self.assertEqual(keys, {("m", "u", "s1")})


if __name__ == "__main__":
    unittest.main()
