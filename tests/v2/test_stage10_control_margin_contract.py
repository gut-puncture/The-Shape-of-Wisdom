import importlib.util
import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT10 = REPO_ROOT / "scripts" / "v2" / "10_causal_validation_tools.py"


def _load_script10():
    if str(SCRIPT10.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT10.parent))
    spec = importlib.util.spec_from_file_location("s10_control_margin", SCRIPT10)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {SCRIPT10}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_cfg(path: Path, *, min_obs_minus_shuffled: float, min_obs_minus_sign_flipped: float) -> None:
    path.write_text(
        "\n".join(
            [
                "models:",
                "  - name: qwen2.5-7b-instruct",
                "    model_id: Qwen/Qwen2.5-7B-Instruct",
                "    revision: r0",
                "validators:",
                "  stage10:",
                "    min_evidence_rows: 5",
                "    min_distractor_rows: 5",
                "    alpha: 0.05",
                "    min_gap_ci_lo: -1.0",
                f"    min_observed_minus_shuffled: {float(min_obs_minus_shuffled)}",
                f"    min_observed_minus_sign_flipped: {float(min_obs_minus_sign_flipped)}",
                "    split_train_fraction: 0.7",
                "    min_train_rows: 0",
                "    min_test_rows: 0",
                "    require_split_direction_match: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_span_labels(path: Path) -> None:
    rows = []
    for i in range(20):
        rows.append(
            {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "prompt_uid": f"u{i}",
                "span_id": f"e{i}",
                "span_label": "evidence",
                "effect_delta": 0.8 + (0.01 if i % 2 == 0 else -0.01),
            }
        )
        rows.append(
            {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "prompt_uid": f"u{i}",
                "span_id": f"d{i}",
                "span_label": "distractor",
                "effect_delta": -0.8 + (0.01 if i % 2 == 0 else -0.01),
            }
        )
    pd.DataFrame.from_records(rows).to_parquet(path, index=False)


class TestStage10ControlMarginContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod10 = _load_script10()

    def test_stage10_fails_when_observed_minus_shuffled_below_floor(self) -> None:
        run_id = "test_v2_stage10_observed_minus_shuffled_fail"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        cfg_path = run_root / "cfg.yaml"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            _write_cfg(cfg_path, min_obs_minus_shuffled=0.05, min_obs_minus_sign_flipped=-1.0)
            _write_span_labels(out_root / "span_labels.parquet")

            def _fake_negative_controls(_labels, seed=123):
                return pd.DataFrame.from_records(
                    [
                        {"control": "observed", "mean_effect_delta": 0.10},
                        {"control": "shuffled", "mean_effect_delta": 0.08},
                        {"control": "sign_flipped", "mean_effect_delta": -0.20},
                    ]
                )

            argv = [
                "prog",
                "--run-id",
                run_id,
                "--config",
                str(cfg_path),
            ]
            with patch.object(self.mod10, "run_negative_controls", side_effect=_fake_negative_controls), patch.object(
                sys, "argv", argv
            ):
                rc = self.mod10.main()

            self.assertEqual(rc, 2)
            report = json.loads((out_root / "10_causal_validation_tools.report.json").read_text(encoding="utf-8"))
            self.assertFalse(bool(report.get("pass")))
            self.assertIn("observed_minus_shuffled", report.get("failing_gates") or [])

        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_stage10_fails_when_observed_minus_sign_flipped_below_floor(self) -> None:
        run_id = "test_v2_stage10_observed_minus_sign_flipped_fail"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        cfg_path = run_root / "cfg.yaml"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            _write_cfg(cfg_path, min_obs_minus_shuffled=-1.0, min_obs_minus_sign_flipped=0.05)
            _write_span_labels(out_root / "span_labels.parquet")

            def _fake_negative_controls(_labels, seed=123):
                return pd.DataFrame.from_records(
                    [
                        {"control": "observed", "mean_effect_delta": 0.10},
                        {"control": "shuffled", "mean_effect_delta": -0.20},
                        {"control": "sign_flipped", "mean_effect_delta": 0.08},
                    ]
                )

            argv = [
                "prog",
                "--run-id",
                run_id,
                "--config",
                str(cfg_path),
            ]
            with patch.object(self.mod10, "run_negative_controls", side_effect=_fake_negative_controls), patch.object(
                sys, "argv", argv
            ):
                rc = self.mod10.main()

            self.assertEqual(rc, 2)
            report = json.loads((out_root / "10_causal_validation_tools.report.json").read_text(encoding="utf-8"))
            self.assertFalse(bool(report.get("pass")))
            self.assertIn("observed_minus_sign_flipped", report.get("failing_gates") or [])
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)


if __name__ == "__main__":
    unittest.main()
