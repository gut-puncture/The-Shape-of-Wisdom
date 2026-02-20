import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestCausalValidationGates(unittest.TestCase):
    def test_stage09_does_not_pass_with_empty_patching_evidence(self) -> None:
        run_id = "test_v2_causal_gate_stage09"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame.from_records(
                [
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_uid": "u0",
                        "layer_index": 0,
                        "delta": -0.2,
                        "drift": 0.0,
                        "s_attn": 0.0,
                        "s_mlp": 0.0,
                    }
                ]
            ).to_parquet(out_root / "tracing_scalars.parquet", index=False)
            pd.DataFrame.from_records(
                [
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_uid": "u0",
                        "trajectory_type": "stable_wrong",
                    }
                ]
            ).to_parquet(out_root / "prompt_types.parquet", index=False)

            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "v2" / "09_causal_tests.py"),
                    "--run-id",
                    run_id,
                ],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 2)

            report = json.loads((out_root / "09_causal_tests.report.json").read_text(encoding="utf-8"))
            self.assertEqual(int(report.get("patching_rows") or 0), 0)
            self.assertFalse(
                bool(report.get("pass")),
                msg="stage 09 must fail-closed when patching evidence is absent",
            )
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_stage10_does_not_pass_without_evidence_and_distractor_labels(self) -> None:
        run_id = "test_v2_causal_gate_stage10"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame.from_records(
                [
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_uid": "u0",
                        "span_id": "instruction",
                        "span_label": "neutral",
                        "effect_delta": 0.0,
                    }
                ]
            ).to_parquet(out_root / "span_labels.parquet", index=False)

            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "v2" / "10_causal_validation_tools.py"),
                    "--run-id",
                    run_id,
                ],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 2)

            report = json.loads((out_root / "10_causal_validation_tools.report.json").read_text(encoding="utf-8"))
            stats = report.get("evidence_vs_distractor") or {}
            self.assertEqual(float(stats.get("evidence_mean", 0.0)), 0.0)
            self.assertEqual(float(stats.get("distractor_mean", 0.0)), 0.0)
            self.assertFalse(
                bool(report.get("pass")),
                msg="stage 10 must fail-closed when evidence/distractor contrasts are unavailable",
            )
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_stage09_fails_when_expected_model_missing(self) -> None:
        run_id = "test_v2_causal_gate_stage09_missing_model"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        cfg_path = run_root / "cfg.yaml"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  - name: qwen2.5-7b-instruct",
                        "    model_id: Qwen/Qwen2.5-7B-Instruct",
                        "    revision: r0",
                        "  - name: llama-3.1-8b-instruct",
                        "    model_id: meta-llama/Llama-3.1-8B-Instruct",
                        "    revision: r1",
                        "validators:",
                        "  stage09:",
                        "    min_ablation_rows: 1",
                        "    min_patching_rows: 1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            trace_rows = []
            for prompt_uid, base, drift in [("u_fail", -0.4, 0.1), ("u_ok", 0.4, -0.1)]:
                for li in range(3):
                    trace_rows.append(
                        {
                            "model_id": "Qwen/Qwen2.5-7B-Instruct",
                            "prompt_uid": prompt_uid,
                            "layer_index": li,
                            "delta": base + 0.05 * li,
                            "drift": drift,
                            "s_attn": 0.03,
                            "s_mlp": 0.02,
                        }
                    )
            pd.DataFrame.from_records(trace_rows).to_parquet(out_root / "tracing_scalars.parquet", index=False)
            pd.DataFrame.from_records(
                [
                    {"model_id": "Qwen/Qwen2.5-7B-Instruct", "prompt_uid": "u_fail", "trajectory_type": "stable_wrong"},
                    {"model_id": "Qwen/Qwen2.5-7B-Instruct", "prompt_uid": "u_ok", "trajectory_type": "stable_correct"},
                ]
            ).to_parquet(out_root / "prompt_types.parquet", index=False)

            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "v2" / "09_causal_tests.py"),
                    "--run-id",
                    run_id,
                    "--config",
                    str(cfg_path),
                ],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 2)

            report = json.loads((out_root / "09_causal_tests.report.json").read_text(encoding="utf-8"))
            self.assertFalse(bool(report.get("pass")))
            self.assertIn("ablation_expected_models_present", report.get("failing_gates") or [])
            self.assertIn("patching_expected_models_present", report.get("failing_gates") or [])
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_stage10_fails_when_expected_model_missing(self) -> None:
        run_id = "test_v2_causal_gate_stage10_missing_model"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        cfg_path = run_root / "cfg.yaml"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  - name: qwen2.5-7b-instruct",
                        "    model_id: Qwen/Qwen2.5-7B-Instruct",
                        "    revision: r0",
                        "  - name: llama-3.1-8b-instruct",
                        "    model_id: meta-llama/Llama-3.1-8B-Instruct",
                        "    revision: r1",
                        "validators:",
                        "  stage10:",
                        "    min_evidence_rows: 5",
                        "    min_distractor_rows: 5",
                        "    alpha: 0.05",
                        "    min_gap_ci_lo: 0.2",
                        "    min_observed_minus_shuffled: 0.05",
                        "    min_observed_minus_sign_flipped: 0.05",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows = []
            for i in range(20):
                rows.append(
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_uid": f"u{i}",
                        "span_id": f"e{i}",
                        "span_label": "evidence",
                        "effect_delta": 0.95 + (0.01 if i % 2 == 0 else -0.01),
                    }
                )
                rows.append(
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_uid": f"u{i}",
                        "span_id": f"d{i}",
                        "span_label": "distractor",
                        "effect_delta": -0.95 + (0.01 if i % 2 == 0 else -0.01),
                    }
                )
            pd.DataFrame.from_records(rows).to_parquet(out_root / "span_labels.parquet", index=False)

            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "v2" / "10_causal_validation_tools.py"),
                    "--run-id",
                    run_id,
                    "--config",
                    str(cfg_path),
                ],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 2)

            report = json.loads((out_root / "10_causal_validation_tools.report.json").read_text(encoding="utf-8"))
            self.assertFalse(bool(report.get("pass")))
            self.assertIn("expected_models_present", report.get("failing_gates") or [])
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)


if __name__ == "__main__":
    unittest.main()
