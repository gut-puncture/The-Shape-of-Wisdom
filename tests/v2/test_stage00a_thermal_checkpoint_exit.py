import importlib.util
import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_00A = REPO_ROOT / "scripts" / "v2" / "00a_generate_baseline_outputs.py"


def _load_stage00a_module():
    if str(SCRIPT_00A.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT_00A.parent))
    spec = importlib.util.spec_from_file_location("sow_v2_stage00a_thermal", SCRIPT_00A)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module: {SCRIPT_00A}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestStage00aThermalCheckpointExit(unittest.TestCase):
    def test_stage00a_returns_95_on_thermal_checkpoint_exit(self) -> None:
        mod = _load_stage00a_module()
        run_id = "test_v2_stage00a_thermal_exit"
        run_root = REPO_ROOT / "runs" / run_id
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            cfg_path = run_root / "cfg.yaml"
            run_root.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  - name: qwen2.5-7b-instruct",
                        "    model_id: Qwen/Qwen2.5-7B-Instruct",
                        "    revision: r0",
                        "data_scope:",
                        "  baseline_manifest_source: /Users/shaileshrana/shape-of-wisdom/prompt_packs/ccc_baseline_v1_3000.jsonl",
                        "  baseline_manifest_expected_rows_full: 1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            def _fake_run_baseline_for_model(*, run_id, model, out_path, **kwargs):
                out_path.parent.mkdir(parents=True, exist_ok=True)
                return {
                    "pass": True,
                    "stopped_early": True,
                    "rows_written": 0,
                    "output_path": str(out_path),
                    "rows_per_second": 0.0,
                    "batch_sizes_used": [1],
                    "stop_reason": "thermal_checkpoint",
                    "thermal_action": {"checkpoint_exit": True},
                }

            with patch.object(mod, "run_baseline_for_model", side_effect=_fake_run_baseline_for_model), patch.object(
                sys,
                "argv",
                [
                    str(SCRIPT_00A),
                    "--run-id",
                    run_id,
                    "--model-name",
                    "qwen2.5-7b-instruct",
                    "--config",
                    str(cfg_path),
                    "--max-prompts",
                    "1",
                ],
            ):
                rc = mod.main()

            self.assertEqual(rc, 95)
            report = json.loads((run_root / "v2" / "00a_generate_baseline_outputs.report.json").read_text(encoding="utf-8"))
            self.assertFalse(bool(report.get("pass")))
            self.assertTrue(bool(report.get("stopped_early")))
            self.assertEqual(str(report.get("stop_reason")), "thermal_checkpoint")
            self.assertIsNotNone(report.get("thermal_action"))
            self.assertIsNone(report.get("done_sentinel"))
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_stage00a_stops_after_first_thermal_checkpoint_model(self) -> None:
        mod = _load_stage00a_module()
        run_id = "test_v2_stage00a_thermal_stop_after_first_model"
        run_root = REPO_ROOT / "runs" / run_id
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            cfg_path = run_root / "cfg.yaml"
            run_root.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  - name: qwen2.5-7b-instruct",
                        "    model_id: Qwen/Qwen2.5-7B-Instruct",
                        "    revision: r0",
                        "  - name: llama-3.1-8b-instruct",
                        "    model_id: meta-llama/Llama-3.1-8B-Instruct",
                        "    revision: r0",
                        "data_scope:",
                        "  baseline_manifest_source: /Users/shaileshrana/shape-of-wisdom/prompt_packs/ccc_baseline_v1_3000.jsonl",
                        "  baseline_manifest_expected_rows_full: 1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            calls = {"n": 0}

            def _fake_run_baseline_for_model(*, run_id, model, out_path, **kwargs):
                calls["n"] += 1
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if calls["n"] > 1:
                    raise AssertionError("stage00a should stop iterating models after first thermal checkpoint")
                return {
                    "pass": True,
                    "stopped_early": True,
                    "rows_written": 0,
                    "output_path": str(out_path),
                    "rows_per_second": 0.0,
                    "batch_sizes_used": [1],
                    "stop_reason": "thermal_checkpoint",
                    "thermal_action": {"checkpoint_exit": True},
                }

            with patch.object(mod, "run_baseline_for_model", side_effect=_fake_run_baseline_for_model), patch.object(
                sys,
                "argv",
                [
                    str(SCRIPT_00A),
                    "--run-id",
                    run_id,
                    "--config",
                    str(cfg_path),
                    "--max-prompts",
                    "1",
                ],
            ):
                rc = mod.main()

            self.assertEqual(rc, 95)
            self.assertEqual(calls["n"], 1)
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)


if __name__ == "__main__":
    unittest.main()
