import importlib.util
import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT09 = REPO_ROOT / "scripts" / "v2" / "09_causal_tests.py"


def _load_script09():
    if str(SCRIPT09.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT09.parent))
    spec = importlib.util.spec_from_file_location("s09_target_layers", SCRIPT09)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {SCRIPT09}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestStage09TargetLayerContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod09 = _load_script09()

    def test_stage09_uses_distinct_ablation_and_patching_layer_lists(self) -> None:
        run_id = "test_v2_stage09_target_layers"
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
                        "causal:",
                        "  ablation_target_layers: [1, 2]",
                        "  patching_target_layers: [5, 6]",
                        "validators:",
                        "  stage09:",
                        "    min_ablation_rows: 1",
                        "    min_patching_rows: 1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            pd.DataFrame.from_records(
                [
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_uid": "u_fail",
                        "layer_index": 0,
                        "delta": -0.2,
                        "drift": 0.1,
                        "s_attn": 0.0,
                        "s_mlp": 0.0,
                    },
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_uid": "u_ok",
                        "layer_index": 0,
                        "delta": 0.2,
                        "drift": -0.1,
                        "s_attn": 0.0,
                        "s_mlp": 0.0,
                    },
                ]
            ).to_parquet(out_root / "tracing_scalars.parquet", index=False)
            pd.DataFrame.from_records(
                [
                    {"model_id": "Qwen/Qwen2.5-7B-Instruct", "prompt_uid": "u_fail", "trajectory_type": "stable_wrong"},
                    {"model_id": "Qwen/Qwen2.5-7B-Instruct", "prompt_uid": "u_ok", "trajectory_type": "stable_correct"},
                ]
            ).to_parquet(out_root / "prompt_types.parquet", index=False)

            ablation_calls: list[list[int]] = []
            patching_calls: list[list[int]] = []

            def _fake_ablation(_df, *, component, target_layers):
                ablation_calls.append([int(x) for x in target_layers])
                return pd.DataFrame.from_records(
                    [
                        {
                            "model_id": "Qwen/Qwen2.5-7B-Instruct",
                            "component": str(component),
                            "delta_shift": 0.1,
                        }
                    ]
                )

            def _fake_patching(_failing, _success, *, component, target_layers):
                patching_calls.append([int(x) for x in target_layers])
                return pd.DataFrame.from_records(
                    [
                        {
                            "model_id": "Qwen/Qwen2.5-7B-Instruct",
                            "component": str(component),
                            "delta_shift": 0.1,
                        }
                    ]
                )

            argv = [
                "prog",
                "--run-id",
                run_id,
                "--model-name",
                "qwen2.5-7b-instruct",
                "--config",
                str(cfg_path),
            ]
            with patch.object(self.mod09, "run_component_ablation", side_effect=_fake_ablation), patch.object(
                self.mod09, "run_activation_patching", side_effect=_fake_patching
            ), patch.object(sys, "argv", argv):
                rc = self.mod09.main()

            self.assertEqual(rc, 0)
            self.assertTrue(ablation_calls)
            self.assertTrue(patching_calls)
            self.assertEqual(ablation_calls[0], [1, 2])
            self.assertEqual(patching_calls[0], [5, 6])

            report = json.loads((out_root / "09_causal_tests.report.json").read_text(encoding="utf-8"))
            self.assertEqual(report.get("ablation_target_layers"), [1, 2])
            self.assertEqual(report.get("patching_target_layers"), [5, 6])
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_stage09_fails_closed_when_delta_shift_is_non_finite(self) -> None:
        run_id = "test_v2_stage09_non_finite_delta_shift"
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
                        "validators:",
                        "  stage09:",
                        "    min_ablation_rows: 1",
                        "    min_patching_rows: 1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            pd.DataFrame.from_records(
                [
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_uid": "u_fail",
                        "layer_index": 0,
                        "delta": -0.2,
                        "drift": 0.1,
                        "s_attn": 0.0,
                        "s_mlp": 0.0,
                    },
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_uid": "u_ok",
                        "layer_index": 0,
                        "delta": 0.2,
                        "drift": -0.1,
                        "s_attn": 0.0,
                        "s_mlp": 0.0,
                    },
                ]
            ).to_parquet(out_root / "tracing_scalars.parquet", index=False)
            pd.DataFrame.from_records(
                [
                    {"model_id": "Qwen/Qwen2.5-7B-Instruct", "prompt_uid": "u_fail", "trajectory_type": "stable_wrong"},
                    {"model_id": "Qwen/Qwen2.5-7B-Instruct", "prompt_uid": "u_ok", "trajectory_type": "stable_correct"},
                ]
            ).to_parquet(out_root / "prompt_types.parquet", index=False)

            def _fake_ablation(_df, *, component, target_layers):
                return pd.DataFrame.from_records(
                    [
                        {
                            "model_id": "Qwen/Qwen2.5-7B-Instruct",
                            "component": str(component),
                            "delta_shift": np.nan,
                        }
                    ]
                )

            def _fake_patching(_failing, _success, *, component, target_layers):
                return pd.DataFrame.from_records(
                    [
                        {
                            "model_id": "Qwen/Qwen2.5-7B-Instruct",
                            "component": str(component),
                            "delta_shift": np.nan,
                        }
                    ]
                )

            argv = [
                "prog",
                "--run-id",
                run_id,
                "--model-name",
                "qwen2.5-7b-instruct",
                "--config",
                str(cfg_path),
            ]
            with patch.object(self.mod09, "run_component_ablation", side_effect=_fake_ablation), patch.object(
                self.mod09, "run_activation_patching", side_effect=_fake_patching
            ), patch.object(sys, "argv", argv):
                rc = self.mod09.main()

            self.assertEqual(rc, 2)
            report = json.loads((out_root / "09_causal_tests.report.json").read_text(encoding="utf-8"))
            self.assertFalse(bool(report.get("pass")))
            self.assertIn("ablation_delta_shift_finite", report.get("failing_gates") or [])
            self.assertIn("patching_delta_shift_finite", report.get("failing_gates") or [])
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)


if __name__ == "__main__":
    unittest.main()
