import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_cfg(path: Path, *, min_count_per_type_per_model: int = 1) -> None:
    path.write_text(
        "\n".join(
            [
                "models:",
                "  - name: qwen2.5-7b-instruct",
                "    model_id: Qwen/Qwen2.5-7B-Instruct",
                "    revision: r0",
                "validators:",
                "  stage03_trajectory:",
                "    required_types: [stable_correct, stable_wrong, unstable_correct, unstable_wrong]",
                f"    min_count_per_type_per_model: {int(min_count_per_type_per_model)}",
                "    tail_len: 8",
                "    max_late_flip_count: 0",
                "    min_abs_delta_tail_floor: 0.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _metrics_for_prompt(*, prompt_uid: str, is_correct: bool, deltas: list[float]) -> list[dict]:
    rows = []
    for idx, delta in enumerate(deltas):
        if idx + 1 < len(deltas):
            drift = float(deltas[idx + 1] - delta)
        else:
            drift = 0.0
        rows.append(
            {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "model_revision": "r0",
                "prompt_uid": prompt_uid,
                "example_id": f"e_{prompt_uid}",
                "wrapper_id": "plain_exam",
                "coarse_domain": "biology",
                "is_correct": bool(is_correct),
                "correct_key": "A",
                "layer_index": int(idx),
                "delta": float(delta),
                "boundary": float(abs(delta)),
                "drift": float(drift),
                "competitor": "B",
                "p_correct": 0.5,
                "prob_margin": 0.1,
                "entropy": 1.0,
            }
        )
    return rows


class TestStage03TrajectoryGates(unittest.TestCase):
    def test_stage03_fails_closed_when_required_types_missing(self) -> None:
        run_id = "test_v2_stage03_missing_types"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        cfg_path = run_root / "cfg.yaml"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            _write_cfg(cfg_path)

            rows = []
            for i in range(6):
                rows.extend(_metrics_for_prompt(prompt_uid=f"u{i}", is_correct=True, deltas=[0.7] * 10))
            pd.DataFrame.from_records(rows).to_parquet(out_root / "decision_metrics.parquet", index=False)

            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "v2" / "03_classify_trajectories.py"),
                    "--run-id",
                    run_id,
                    "--config",
                    str(cfg_path),
                ],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 2)
            report = json.loads((out_root / "03_classify_trajectories.report.json").read_text(encoding="utf-8"))
            self.assertFalse(bool(report.get("pass")))
            self.assertTrue(report.get("failing_gates"))
            self.assertIn("required_types_per_model", report.get("failing_gates") or [])
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_stage03_passes_when_all_required_types_present(self) -> None:
        run_id = "test_v2_stage03_all_types"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        cfg_path = run_root / "cfg.yaml"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            _write_cfg(cfg_path)

            rows = []
            rows.extend(_metrics_for_prompt(prompt_uid="stable_c", is_correct=True, deltas=[0.8] * 10))
            rows.extend(_metrics_for_prompt(prompt_uid="stable_w", is_correct=False, deltas=[-0.8] * 10))
            rows.extend(_metrics_for_prompt(prompt_uid="unstable_c", is_correct=True, deltas=[0.8, 0.8, 0.8, 0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8]))
            rows.extend(_metrics_for_prompt(prompt_uid="unstable_w", is_correct=False, deltas=[-0.8, -0.8, -0.8, -0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]))
            pd.DataFrame.from_records(rows).to_parquet(out_root / "decision_metrics.parquet", index=False)

            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "v2" / "03_classify_trajectories.py"),
                    "--run-id",
                    run_id,
                    "--config",
                    str(cfg_path),
                ],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 0)
            report = json.loads((out_root / "03_classify_trajectories.report.json").read_text(encoding="utf-8"))
            self.assertTrue(bool(report.get("pass")))
            self.assertIn("gates", report)
            self.assertIn("failing_gates", report)
            self.assertIn("per_model_type_counts", report)
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_stage03_fails_when_min_count_per_type_per_model_unmet(self) -> None:
        run_id = "test_v2_stage03_min_count_fail"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        cfg_path = run_root / "cfg.yaml"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            _write_cfg(cfg_path, min_count_per_type_per_model=2)

            rows = []
            rows.extend(_metrics_for_prompt(prompt_uid="stable_c", is_correct=True, deltas=[0.8] * 10))
            rows.extend(_metrics_for_prompt(prompt_uid="stable_w", is_correct=False, deltas=[-0.8] * 10))
            rows.extend(_metrics_for_prompt(prompt_uid="unstable_c", is_correct=True, deltas=[0.8, 0.8, 0.8, 0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8]))
            rows.extend(_metrics_for_prompt(prompt_uid="unstable_w", is_correct=False, deltas=[-0.8, -0.8, -0.8, -0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]))
            pd.DataFrame.from_records(rows).to_parquet(out_root / "decision_metrics.parquet", index=False)

            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "v2" / "03_classify_trajectories.py"),
                    "--run-id",
                    run_id,
                    "--config",
                    str(cfg_path),
                ],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 2)
            report = json.loads((out_root / "03_classify_trajectories.report.json").read_text(encoding="utf-8"))
            self.assertFalse(bool(report.get("pass")))
            self.assertIn("min_count_per_type_per_model", report.get("failing_gates") or [])
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)


if __name__ == "__main__":
    unittest.main()
