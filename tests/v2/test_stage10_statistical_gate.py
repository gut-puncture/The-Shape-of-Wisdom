import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_cfg(
    path: Path,
    *,
    alpha: float,
    min_gap_ci_lo: float,
    min_train_rows: int = 0,
    min_test_rows: int = 0,
    split_train_fraction: float = 0.7,
    require_split_direction_match: bool = False,
) -> None:
    path.write_text(
        "\n".join(
            [
                "validators:",
                "  stage10:",
                "    min_evidence_rows: 5",
                "    min_distractor_rows: 5",
                f"    alpha: {alpha}",
                f"    min_gap_ci_lo: {min_gap_ci_lo}",
                "    min_observed_minus_shuffled: 0.05",
                "    min_observed_minus_sign_flipped: 0.05",
                f"    split_train_fraction: {split_train_fraction}",
                f"    min_train_rows: {min_train_rows}",
                f"    min_test_rows: {min_test_rows}",
                f"    require_split_direction_match: {'true' if require_split_direction_match else 'false'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


class TestStage10StatisticalGate(unittest.TestCase):
    def test_stage10_fails_when_gap_is_not_significant(self) -> None:
        run_id = "test_v2_stage10_not_significant"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        cfg_path = run_root / "cfg.yaml"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            _write_cfg(cfg_path, alpha=0.05, min_gap_ci_lo=0.0)

            rows = []
            for i in range(10):
                rows.append(
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_uid": f"u{i}",
                        "span_id": f"e{i}",
                        "span_label": "evidence",
                        "effect_delta": 0.10 + (0.01 if i % 2 == 0 else -0.01),
                    }
                )
                rows.append(
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_uid": f"u{i}",
                        "span_id": f"d{i}",
                        "span_label": "distractor",
                        "effect_delta": 0.09 + (0.01 if i % 2 == 0 else -0.01),
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
            self.assertIn("bh_significant", report.get("failing_gates") or [])
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_stage10_passes_when_gap_is_strong(self) -> None:
        run_id = "test_v2_stage10_significant"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        cfg_path = run_root / "cfg.yaml"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            _write_cfg(
                cfg_path,
                alpha=0.05,
                min_gap_ci_lo=0.2,
                min_train_rows=5,
                min_test_rows=5,
                split_train_fraction=0.5,
                require_split_direction_match=True,
            )

            rows = []
            for i in range(20):
                rows.append(
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_uid": f"u{i}",
                        "span_id": f"e{i}",
                        "span_label": "evidence",
                        "effect_delta": 0.95 + (0.02 if i % 2 == 0 else -0.02),
                    }
                )
                rows.append(
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_uid": f"u{i}",
                        "span_id": f"d{i}",
                        "span_label": "distractor",
                        "effect_delta": -0.95 + (0.02 if i % 2 == 0 else -0.02),
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
            self.assertEqual(proc.returncode, 0)

            report = json.loads((out_root / "10_causal_validation_tools.report.json").read_text(encoding="utf-8"))
            self.assertTrue(bool(report.get("pass")))
            self.assertEqual(report.get("failing_gates") or [], [])
            self.assertIn("split_contract", report)
            self.assertTrue(bool(report.get("gates", {}).get("split_train_rows_min")))
            self.assertTrue(bool(report.get("gates", {}).get("split_test_rows_min")))
            self.assertTrue(bool(report.get("gates", {}).get("split_direction_consistent")))
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_stage10_fails_when_split_min_rows_unmet(self) -> None:
        run_id = "test_v2_stage10_split_rows_fail"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        cfg_path = run_root / "cfg.yaml"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            _write_cfg(
                cfg_path,
                alpha=0.05,
                min_gap_ci_lo=0.0,
                min_train_rows=100,
                min_test_rows=100,
                split_train_fraction=0.7,
                require_split_direction_match=False,
            )

            rows = []
            for i in range(6):
                rows.append(
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_uid": f"u{i}",
                        "span_id": f"e{i}",
                        "span_label": "evidence",
                        "effect_delta": 0.40 + (0.01 if i % 2 == 0 else -0.01),
                    }
                )
                rows.append(
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_uid": f"u{i}",
                        "span_id": f"d{i}",
                        "span_label": "distractor",
                        "effect_delta": 0.10 + (0.01 if i % 2 == 0 else -0.01),
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
            self.assertIn("split_train_rows_min", report.get("failing_gates") or [])
            self.assertIn("split_test_rows_min", report.get("failing_gates") or [])
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_stage10_split_is_prompt_disjoint(self) -> None:
        run_id = "test_v2_stage10_split_prompt_disjoint"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        cfg_path = run_root / "cfg.yaml"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            _write_cfg(
                cfg_path,
                alpha=0.05,
                min_gap_ci_lo=0.0,
                min_train_rows=1,
                min_test_rows=1,
                split_train_fraction=0.6,
                require_split_direction_match=False,
            )

            rows = []
            for i in range(8):
                prompt_uid = "u0" if i < 5 else "u1"
                rows.append(
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_uid": prompt_uid,
                        "span_id": f"e{i}",
                        "span_label": "evidence",
                        "effect_delta": 0.8 + (0.01 if i % 2 == 0 else -0.01),
                    }
                )
                rows.append(
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_uid": prompt_uid,
                        "span_id": f"d{i}",
                        "span_label": "distractor",
                        "effect_delta": -0.8 + (0.01 if i % 2 == 0 else -0.01),
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
            self.assertEqual(proc.returncode, 0)

            report = json.loads((out_root / "10_causal_validation_tools.report.json").read_text(encoding="utf-8"))
            split = (report.get("stats_per_model") or {}).get("Qwen/Qwen2.5-7B-Instruct", {}).get("split") or {}
            self.assertEqual(split.get("split_unit"), "prompt_uid")
            self.assertEqual(int(split.get("overlap_prompt_count", -1)), 0)
            self.assertTrue(bool(report.get("gates", {}).get("split_prompt_disjoint")))
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)


if __name__ == "__main__":
    unittest.main()
