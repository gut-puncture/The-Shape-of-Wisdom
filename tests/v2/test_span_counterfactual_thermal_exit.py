import importlib.util
import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "v2" / "05_span_counterfactuals.py"


class TestSpanCounterfactualThermalExit(unittest.TestCase):
    def test_checkpoint_exit_returns_95(self) -> None:
        run_id = "test_v2_span_thermal_exit"
        run_root = REPO_ROOT / "runs" / run_id
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            (run_root / "v2").mkdir(parents=True, exist_ok=True)
            (run_root / "manifests").mkdir(parents=True, exist_ok=True)

            pd.DataFrame.from_records(
                [
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_uid": "u0",
                        "layer_index": 0,
                        "delta": 0.2,
                        "competitor": "B",
                        "correct_key": "A",
                    }
                ]
            ).to_parquet(run_root / "v2" / "decision_metrics.parquet", index=False)

            (run_root / "manifests" / "ccc_baseline.jsonl").write_text(
                json.dumps(
                    {
                        "prompt_uid": "u0",
                        "prompt_text": "Q?\nA) x\nB) y\nC) z\nD) w\nAnswer: ",
                        "correct_key": "A",
                        "example_id": "e0",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            cfg_path = run_root / "cfg.yaml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  - name: qwen2.5-7b-instruct",
                        "    model_id: Qwen/Qwen2.5-7B-Instruct",
                        "    revision: r0",
                        "span_counterfactual:",
                        "  mode: proxy",
                        "thermal_policy:",
                        "  enabled: true",
                        "  provider: powermetrics_thermal_pressure",
                        "  cutoff_level: serious",
                        "  cooldown_seconds: 1",
                        "  check_interval_seconds: 1",
                        "  pause_mode: checkpoint_exit",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            if str(SCRIPT_PATH.parent) not in sys.path:
                sys.path.insert(0, str(SCRIPT_PATH.parent))
            spec = importlib.util.spec_from_file_location("s05", SCRIPT_PATH)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            argv = [
                "prog",
                "--run-id",
                run_id,
                "--model-name",
                "qwen2.5-7b-instruct",
                "--config",
                str(cfg_path),
                "--counterfactual-mode",
                "proxy",
            ]
            with patch(
                "sow.thermal.thermal_governor.ThermalGovernor.maybe_cooldown",
                return_value={"checkpoint_exit": True, "cooldown_seconds": 1},
            ):
                old_argv = sys.argv
                sys.argv = argv
                try:
                    rc = module.main()
                finally:
                    sys.argv = old_argv

            self.assertEqual(rc, 95)
            report_path = run_root / "v2" / "05_span_counterfactuals.report.json"
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(bool(report.get("pass")))
            self.assertEqual(report.get("stop_reason"), "thermal_checkpoint")
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)


if __name__ == "__main__":
    unittest.main()
