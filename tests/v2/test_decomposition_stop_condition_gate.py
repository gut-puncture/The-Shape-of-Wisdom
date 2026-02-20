import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestDecompositionStopConditionGate(unittest.TestCase):
    def test_stage08_fails_when_drift_is_not_explained(self) -> None:
        run_id = "test_v2_decomposition_gate"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            rows = []
            for li, drift in enumerate([1.0, -1.0, 1.0, -1.0, 1.0, -1.0]):
                rows.append(
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_uid": "u0",
                        "layer_index": li,
                        "drift": drift,
                        "s_attn": 0.0,
                        "s_mlp": 0.0,
                    }
                )
            pd.DataFrame.from_records(rows).to_parquet(out_root / "tracing_scalars.parquet", index=False)

            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "v2" / "08_attention_and_mlp_decomposition.py"),
                    "--run-id",
                    run_id,
                ],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 2)

            report = json.loads((out_root / "08_attention_and_mlp_decomposition.report.json").read_text(encoding="utf-8"))
            self.assertFalse(bool(report.get("pass")))
            self.assertIn("Qwen/Qwen2.5-7B-Instruct", report.get("failing_models") or [])
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)


if __name__ == "__main__":
    unittest.main()
