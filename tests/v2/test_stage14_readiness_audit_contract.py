import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "v2" / "14_readiness_audit.py"


class TestStage14ReadinessAuditContract(unittest.TestCase):
    def _run_stage14(self, *, run_id: str, ledger_path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--run-id",
                run_id,
                "--ledger-path",
                str(ledger_path),
            ],
            cwd=str(REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )

    def test_go_when_all_required_evidence_is_present(self) -> None:
        run_id = "test_v2_stage14_go"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            (out_root / "meta").mkdir(parents=True, exist_ok=True)
            (out_root / "03_classify_trajectories.report.json").write_text(json.dumps({"pass": True}) + "\n", encoding="utf-8")
            (out_root / "05_span_counterfactuals.report.json").write_text(json.dumps({"pass": True}) + "\n", encoding="utf-8")
            (out_root / "prompt_types.parquet").write_text("stub\n", encoding="utf-8")

            ledger_path = run_root / "ledger.yaml"
            ledger = {
                "requirements": [
                    {
                        "requirement_id": "RQ-001",
                        "claim_type": "descriptive",
                        "acceptance_criteria": "trajectory report exists and passes",
                        "required_artifacts": ["03_classify_trajectories.report.json", "prompt_types.parquet"],
                        "required_controls": ["03_classify_trajectories.report.json"],
                        "pass_conditions": {
                            "all_of": [
                                {"type": "json_equals", "path": "pass", "value": True},
                            ]
                        },
                    },
                    {
                        "requirement_id": "RQ-002",
                        "claim_type": "mechanistic",
                        "acceptance_criteria": "stage05 report passes",
                        "required_artifacts": ["05_span_counterfactuals.report.json"],
                        "required_controls": ["05_span_counterfactuals.report.json"],
                        "pass_conditions": {
                            "all_of": [
                                {"type": "json_equals", "path": "pass", "value": True},
                            ]
                        },
                    },
                ]
            }
            ledger_path.write_text(yaml.safe_dump(ledger, sort_keys=False), encoding="utf-8")

            proc = self._run_stage14(run_id=run_id, ledger_path=ledger_path)
            self.assertEqual(proc.returncode, 0)
            audit_path = out_root / "meta" / "readiness_audit.json"
            self.assertTrue(audit_path.exists())
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(str(audit.get("verdict")), "GO")
            self.assertEqual(audit.get("failing_requirements"), [])
            req = (audit.get("requirements") or [])[0]
            self.assertIn("failing_conditions", req)
            self.assertIn("schema_errors", req)
            self.assertIn("evidence_checked", req)
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_no_go_when_evidence_missing(self) -> None:
        run_id = "test_v2_stage14_nogo"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            (out_root / "meta").mkdir(parents=True, exist_ok=True)
            (out_root / "03_classify_trajectories.report.json").write_text(json.dumps({"pass": False}) + "\n", encoding="utf-8")

            ledger_path = run_root / "ledger.yaml"
            ledger = {
                "requirements": [
                    {
                        "requirement_id": "RQ-001",
                        "claim_type": "descriptive",
                        "acceptance_criteria": "stage03 must pass",
                        "required_artifacts": ["03_classify_trajectories.report.json", "missing.parquet"],
                        "required_controls": ["03_classify_trajectories.report.json"],
                        "pass_conditions": {
                            "all_of": [
                                {"type": "json_equals", "path": "pass", "value": True},
                            ]
                        },
                    }
                ]
            }
            ledger_path.write_text(yaml.safe_dump(ledger, sort_keys=False), encoding="utf-8")

            proc = self._run_stage14(run_id=run_id, ledger_path=ledger_path)
            self.assertEqual(proc.returncode, 2)
            audit = json.loads((out_root / "meta" / "readiness_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(str(audit.get("verdict")), "NO-GO")
            self.assertIn("RQ-001", set(audit.get("failing_requirements") or []))
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_unknown_pass_condition_type_fails_closed(self) -> None:
        run_id = "test_v2_stage14_unknown_condition_type"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            (out_root / "meta").mkdir(parents=True, exist_ok=True)
            report_path = out_root / "03_classify_trajectories.report.json"
            report_path.write_text(json.dumps({"pass": True}) + "\n", encoding="utf-8")
            (out_root / "prompt_types.parquet").write_text("stub\n", encoding="utf-8")

            ledger_path = run_root / "ledger.yaml"
            ledger_path.write_text(
                yaml.safe_dump(
                    {
                        "requirements": [
                            {
                                "requirement_id": "RQ-001",
                                "claim_type": "descriptive",
                                "acceptance_criteria": "unknown condition type must fail",
                                "required_artifacts": ["03_classify_trajectories.report.json", "prompt_types.parquet"],
                                "required_controls": ["03_classify_trajectories.report.json"],
                                "pass_conditions": {"all_of": [{"type": "unknown_condition", "path": "pass", "value": True}]},
                            }
                        ]
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            proc = self._run_stage14(run_id=run_id, ledger_path=ledger_path)
            self.assertEqual(proc.returncode, 2, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            audit = json.loads((out_root / "meta" / "readiness_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(str(audit.get("verdict")), "NO-GO")
            req = (audit.get("requirements") or [])[0]
            self.assertFalse(bool(req.get("pass")))
            self.assertTrue(len(req.get("schema_errors") or []) > 0)
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_malformed_pass_conditions_schema_fails_closed(self) -> None:
        run_id = "test_v2_stage14_malformed_conditions"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            (out_root / "meta").mkdir(parents=True, exist_ok=True)
            report_path = out_root / "05_span_counterfactuals.report.json"
            report_path.write_text(json.dumps({"pass": True}) + "\n", encoding="utf-8")
            ledger_path = run_root / "ledger.yaml"
            ledger_path.write_text(
                yaml.safe_dump(
                    {
                        "requirements": [
                            {
                                "requirement_id": "RQ-004",
                                "claim_type": "descriptive",
                                "acceptance_criteria": "malformed pass_conditions must fail",
                                "required_artifacts": ["05_span_counterfactuals.report.json"],
                                "required_controls": ["05_span_counterfactuals.report.json"],
                                "pass_conditions": {"all_of": {"type": "json_equals", "path": "pass", "value": True}},
                            }
                        ]
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            proc = self._run_stage14(run_id=run_id, ledger_path=ledger_path)
            self.assertEqual(proc.returncode, 2, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            audit = json.loads((out_root / "meta" / "readiness_audit.json").read_text(encoding="utf-8"))
            req = (audit.get("requirements") or [])[0]
            self.assertFalse(bool(req.get("pass")))
            self.assertTrue(len(req.get("schema_errors") or []) > 0)
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_non_json_control_fails_closed(self) -> None:
        run_id = "test_v2_stage14_non_json_control"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            (out_root / "meta").mkdir(parents=True, exist_ok=True)
            (out_root / "artifact.parquet").write_text("stub\n", encoding="utf-8")
            (out_root / "control.txt").write_text("pass=true\n", encoding="utf-8")
            ledger_path = run_root / "ledger.yaml"
            ledger_path.write_text(
                yaml.safe_dump(
                    {
                        "requirements": [
                            {
                                "requirement_id": "RQ-900",
                                "claim_type": "descriptive",
                                "acceptance_criteria": "controls must be json and pass=true",
                                "required_artifacts": ["artifact.parquet"],
                                "required_controls": ["control.txt"],
                                "pass_conditions": {"all_of": []},
                            }
                        ]
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            proc = self._run_stage14(run_id=run_id, ledger_path=ledger_path)
            self.assertEqual(proc.returncode, 2, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            audit = json.loads((out_root / "meta" / "readiness_audit.json").read_text(encoding="utf-8"))
            req = (audit.get("requirements") or [])[0]
            self.assertFalse(bool(req.get("pass")))
            self.assertIn("control.txt", set(req.get("failing_controls") or []))
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_rq014_self_contract_does_not_require_preexisting_audit_file(self) -> None:
        run_id = "test_v2_stage14_self_contract"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            (out_root / "meta").mkdir(parents=True, exist_ok=True)
            (out_root / "03_classify_trajectories.report.json").write_text(json.dumps({"pass": True}) + "\n", encoding="utf-8")
            (out_root / "prompt_types.parquet").write_text("stub\n", encoding="utf-8")
            ledger_path = run_root / "ledger.yaml"
            ledger = {
                "requirements": [
                    {
                        "requirement_id": "RQ-001",
                        "claim_type": "descriptive",
                        "acceptance_criteria": "stage03 must pass",
                        "required_artifacts": ["03_classify_trajectories.report.json", "prompt_types.parquet"],
                        "required_controls": ["03_classify_trajectories.report.json"],
                        "pass_conditions": {"all_of": [{"type": "json_equals", "path": "pass", "value": True}]},
                    },
                    {
                        "requirement_id": "RQ-014",
                        "claim_type": "descriptive",
                        "acceptance_criteria": "audit must report deterministic GO",
                        "required_artifacts": ["meta/readiness_audit.json"],
                        "required_controls": ["meta/readiness_audit.json"],
                        "pass_conditions": {"all_of": [{"type": "self_contract"}]},
                    },
                ]
            }
            ledger_path.write_text(yaml.safe_dump(ledger, sort_keys=False), encoding="utf-8")

            proc = self._run_stage14(run_id=run_id, ledger_path=ledger_path)
            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            audit = json.loads((out_root / "meta" / "readiness_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(str(audit.get("verdict")), "GO")
            self.assertEqual(audit.get("failing_requirements"), [])
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)


if __name__ == "__main__":
    unittest.main()
