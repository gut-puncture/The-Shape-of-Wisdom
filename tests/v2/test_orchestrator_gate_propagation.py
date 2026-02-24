import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestOrchestratorGatePropagation(unittest.TestCase):
    def test_snapshot_only_reports_ready_for_execution(self) -> None:
        run_id = "test_v2_orchestrator_snapshot_ready"
        run_root = REPO_ROOT / "runs" / run_id
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "v2" / "00_run_experiment.py"),
                    "--run-id",
                    run_id,
                    "--mode",
                    "full",
                    "--snapshot-only",
                ],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 0)
            report_path = run_root / "v2" / "00_run_experiment.report.json"
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(bool(report.get("snapshot_only")))
            self.assertTrue(bool(report.get("pass")))
            self.assertFalse(bool(report.get("complete")))
            self.assertTrue(bool(report.get("ready_to_execute_full_experiment")))
            self.assertEqual(report.get("executed_scripts"), [])
            self.assertEqual(report.get("skipped_scripts"), [])
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

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
            self.assertFalse(bool(report.get("ready_to_execute_full_experiment")))
            self.assertIn("failed_script", report)
            self.assertIn("failed_exit_code", report)
            self.assertEqual(str(report.get("failed_script")), "00a_generate_baseline_outputs.py")
            snapshot_path = run_root / "v2" / "meta" / "run_start_metadata_snapshot.json"
            self.assertTrue(snapshot_path.exists(), msg="orchestrator must snapshot metadata contract at run start")
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_run_full_script_invokes_stage00_snapshot_before_stage_execution(self) -> None:
        script = REPO_ROOT / "scripts" / "v2" / "run_full_local_v2.sh"
        text = script.read_text(encoding="utf-8")
        self.assertIn("00_run_experiment.py", text, msg="full local runner must call stage00 provenance snapshot path")
        self.assertIn("00a_generate_baseline_outputs.py", text, msg="full local runner must call baseline generation stage")
        self.assertIn("01_extract_baseline.py", text)
        self.assertLess(
            text.index("00_run_experiment.py"),
            text.index("00a_generate_baseline_outputs.py"),
            msg="stage00 snapshot must happen before stage00a",
        )
        self.assertLess(
            text.index("00a_generate_baseline_outputs.py"),
            text.index("01_extract_baseline.py"),
            msg="stage00a must happen before stage01",
        )

    def test_run_full_script_maps_stage00a_done_sentinel_and_thermal_wrapper(self) -> None:
        script = REPO_ROOT / "scripts" / "v2" / "run_full_local_v2.sh"
        text = script.read_text(encoding="utf-8")
        self.assertIn(
            "00a_generate_baseline_outputs.py) echo \"${out_root}/00a_generate_baseline_outputs.done\"",
            text,
            msg="stage00a must define done sentinel path for thermal resume loop",
        )
        self.assertIn(
            "run_with_thermal_resume.sh",
            text,
            msg="local runner must execute checkpoint-exit stages through thermal resume wrapper",
        )

    def test_orchestrator_fails_closed_when_stage00a_fails(self) -> None:
        run_id = "test_v2_orchestrator_stage00a_failure"
        run_root = REPO_ROOT / "runs" / run_id
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            cfg = run_root / "cfg.yaml"
            run_root.mkdir(parents=True, exist_ok=True)
            cfg.write_text(
                "\n".join(
                    [
                        "models:",
                        "  - name: qwen2.5-7b-instruct",
                        "    model_id: Qwen/Qwen2.5-7B-Instruct",
                        "    revision: r0",
                        "data_scope:",
                        "  baseline_manifest_source: /nonexistent/path/ccc_baseline_v1_3000.jsonl",
                        "runtime_estimator:",
                        "  require_measured_rps_for_full: true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

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
                    "--config",
                    str(cfg),
                ],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            report = json.loads((run_root / "v2" / "00_run_experiment.report.json").read_text(encoding="utf-8"))
            self.assertEqual(str(report.get("failed_script")), "00a_generate_baseline_outputs.py")
            self.assertIn("baseline_prompt_count_current", report)
            skipped = report.get("skipped_scripts") or []
            skipped_names = {str(x.get("script")) for x in skipped}
            self.assertIn("01_extract_baseline.py", skipped_names)
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)


if __name__ == "__main__":
    unittest.main()
