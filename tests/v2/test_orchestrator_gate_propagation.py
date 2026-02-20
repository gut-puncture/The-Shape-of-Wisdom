import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestOrchestratorGatePropagation(unittest.TestCase):
    def test_orchestrator_writes_report_on_stage_failure(self) -> None:
        run_id = "test_v2_orchestrator_gate_failure"
        run_root = REPO_ROOT / "runs" / run_id
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            # Intentionally do not create baseline outputs so stage 01 fails.
            (run_root / "manifests").mkdir(parents=True, exist_ok=True)
            (run_root / "manifests" / "ccc_baseline.jsonl").write_text("", encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "v2" / "00_run_experiment.py"),
                    "--run-id",
                    run_id,
                    "--mode",
                    "full",
                    "--model-name",
                    "qwen2.5-7b-instruct",
                ],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)

            report_path = run_root / "v2" / "00_run_experiment.report.json"
            self.assertTrue(report_path.exists(), msg="orchestrator must emit report even when a stage fails")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(bool(report.get("pass")))
            self.assertFalse(bool(report.get("complete")))
            self.assertIn("failed_script", report)
            self.assertIn("failed_exit_code", report)
            self.assertEqual(str(report.get("failed_script")), "01_extract_baseline.py")
            snapshot_path = run_root / "v2" / "meta" / "run_start_metadata_snapshot.json"
            self.assertTrue(snapshot_path.exists(), msg="orchestrator must snapshot metadata contract at run start")
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)


if __name__ == "__main__":
    unittest.main()
