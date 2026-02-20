import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestGenerateAssetsGate(unittest.TestCase):
    def test_fails_when_required_files_missing(self) -> None:
        run_id = "test_v2_assets_gate"
        run_root = REPO_ROOT / "runs" / run_id / "v2"
        final_root = REPO_ROOT / "artifacts" / "final_result_v2" / run_id

        if run_root.parent.exists():
            shutil.rmtree(run_root.parent)
        if final_root.exists():
            shutil.rmtree(final_root)

        try:
            run_root.mkdir(parents=True, exist_ok=True)
            metrics = pd.DataFrame.from_records(
                [
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_uid": "u0",
                        "layer_index": 0,
                        "delta": 0.1,
                        "drift": 0.0,
                        "boundary": 0.1,
                        "is_correct": True,
                        "entropy": 0.5,
                    }
                ]
            )
            ptypes = pd.DataFrame.from_records(
                [
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_uid": "u0",
                        "trajectory_type": "stable_correct",
                        "is_correct": True,
                        "sign_flip_count": 0,
                        "late_flip_count": 0,
                        "mean_abs_drift_last8": 0.0,
                    }
                ]
            )
            metrics.to_parquet(run_root / "decision_metrics.parquet", index=False)
            ptypes.to_parquet(run_root / "prompt_types.parquet", index=False)

            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "v2" / "11_generate_paper_assets.py"),
                    "--run-id",
                    run_id,
                ],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 2)

            report_path = final_root / "final_report.json"
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(bool(report.get("pass")))
            self.assertGreater(len(report.get("missing_required_files") or []), 0)
        finally:
            if run_root.parent.exists():
                shutil.rmtree(run_root.parent)
            if final_root.exists():
                shutil.rmtree(final_root)


if __name__ == "__main__":
    unittest.main()
