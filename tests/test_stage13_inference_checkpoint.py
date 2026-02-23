import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "v2" / "run_with_thermal_resume.sh"
SCRIPT13 = REPO_ROOT / "scripts" / "v2" / "13_baseline_rerun_decision.py"


class TestStage13InferenceCheckpoint(unittest.TestCase):
    def test_loops_on_95_then_exits_on_done_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            done = root / "done.json"
            marker = root / "count.txt"
            cmd = (
                "c=$(cat \"$1\" 2>/dev/null || echo 0); "
                "c=$((c+1)); "
                "echo \"$c\" > \"$1\"; "
                "if [ \"$c\" -eq 1 ]; then exit 95; fi; "
                "echo '{\"pass\":true}' > \"$2\"; "
                "exit 0"
            )
            proc = subprocess.run(
                [
                    str(SCRIPT),
                    "--done-sentinel",
                    str(done),
                    "--cooldown-seconds",
                    "0",
                    "--",
                    "sh",
                    "-c",
                    cmd,
                    "_",
                    str(marker),
                    str(done),
                ],
                cwd=str(REPO_ROOT),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertTrue(done.exists())
            self.assertEqual((marker.read_text(encoding="utf-8").strip()), "2")
            payload = json.loads(done.read_text(encoding="utf-8"))
            self.assertTrue(bool(payload.get("pass")))

    def test_non_95_failure_propagates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            done = root / "done.json"
            proc = subprocess.run(
                [
                    str(SCRIPT),
                    "--done-sentinel",
                    str(done),
                    "--cooldown-seconds",
                    "0",
                    "--",
                    "sh",
                    "-c",
                    "exit 7",
                ],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 7)
            self.assertFalse(done.exists())

    def test_existing_done_sentinel_short_circuits(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            done = root / "done.json"
            done.write_text("{\"pass\":true}\n", encoding="utf-8")
            marker = root / "marker.txt"
            proc = subprocess.run(
                [
                    str(SCRIPT),
                    "--done-sentinel",
                    str(done),
                    "--cooldown-seconds",
                    "0",
                    "--",
                    "sh",
                    "-c",
                    f"echo should-not-run > {marker}",
                ],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 0)
            self.assertFalse(marker.exists())

    def test_stage13_rerun_decision_for_publication_run_requires_full_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_id = "test_stage13_publication_decision"
            run_root = REPO_ROOT / "runs" / run_id / "v2"
            run_root.mkdir(parents=True, exist_ok=True)
            decision_path = run_root / "meta" / "stage13_baseline_rerun_decision.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT13),
                    "--run-id",
                    run_id,
                    "--final-publication-run",
                ],
                cwd=str(REPO_ROOT),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertTrue(decision_path.exists())
            obj = json.loads(decision_path.read_text(encoding="utf-8"))
            self.assertTrue(bool(obj.get("requires_full_baseline_regeneration")))
            self.assertEqual(str(obj.get("policy_mode")), "final_publication_run")

    def test_stage13_rerun_decision_for_remediation_run_is_explicit(self) -> None:
        run_id = "test_stage13_remediation_decision"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        if run_root.exists():
            import shutil

            shutil.rmtree(run_root)
        try:
            out_root.mkdir(parents=True, exist_ok=True)
            import pandas as pd

            pd.DataFrame.from_records(
                [
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_uid": "u0",
                        "layer_index": 0,
                        "correct_key": "A",
                        "delta": 0.2,
                    }
                ]
            ).to_parquet(out_root / "decision_metrics.parquet", index=False)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT13),
                    "--run-id",
                    run_id,
                ],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 0)
            decision_path = out_root / "meta" / "stage13_baseline_rerun_decision.json"
            self.assertTrue(decision_path.exists())
            obj = json.loads(decision_path.read_text(encoding="utf-8"))
            self.assertEqual(str(obj.get("policy_mode")), "remediation_verification")
            self.assertIn("requires_full_baseline_regeneration", obj)
            self.assertIn("rationale", obj)
        finally:
            if run_root.exists():
                import shutil

                shutil.rmtree(run_root)


if __name__ == "__main__":
    unittest.main()
