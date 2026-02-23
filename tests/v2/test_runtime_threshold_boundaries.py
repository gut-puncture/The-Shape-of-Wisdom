import importlib.util
import json
import re
import shutil
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


from sow.v2.runtime_policy import choose_backend  # noqa: E402


SCRIPT00 = REPO_ROOT / "scripts" / "v2" / "00_run_experiment.py"


def _load_script00():
    if str(SCRIPT00.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT00.parent))
    spec = importlib.util.spec_from_file_location("s00_runtime_boundaries", SCRIPT00)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load script: {SCRIPT00}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRuntimeThresholdBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod00 = _load_script00()

    def test_choose_backend_exactly_36_hours_prefers_mac(self) -> None:
        self.assertEqual(choose_backend(estimated_hours_all_models=36.0, threshold_hours=36.0), "mac")
        self.assertEqual(choose_backend(estimated_hours_all_models=36.00001, threshold_hours=36.0), "gpu")

    def test_stage_row_multiplier_changes_runtime_ordering(self) -> None:
        cfg = {
            "models": [{}, {}, {}],
            "runtime_estimator": {
                "threshold_model_count": 3,
                "stage_row_multiplier": {
                    "05_span_counterfactuals.py": 8.0,
                    "07_run_tracing.py": 1.0,
                },
            },
        }
        est = self.mod00._heavy_stage_estimates(cfg=cfg, baseline_count=100, rows_per_second=10.0)
        self.assertGreater(
            float(est["05_span_counterfactuals.py"].estimated_hours_all_models),
            float(est["07_run_tracing.py"].estimated_hours_all_models),
        )

    def test_default_model_count_uses_config_models_length(self) -> None:
        cfg = {
            "models": [{}, {}],
            "runtime_estimator": {
                "stage_row_multiplier": {
                    "05_span_counterfactuals.py": 8.0,
                    "07_run_tracing.py": 1.0,
                }
            },
        }
        est = self.mod00._heavy_stage_estimates(cfg=cfg, baseline_count=10, rows_per_second=10.0)
        self.assertEqual(int(est["05_span_counterfactuals.py"].model_count), 2)
        self.assertEqual(int(est["07_run_tracing.py"].model_count), 2)

    def test_full_mode_rejects_default_fallback_when_policy_requires_measured(self) -> None:
        cfg = {
            "runtime_estimator": {
                "require_measured_rps_for_full": True,
            }
        }
        with self.assertRaises(RuntimeError):
            self.mod00._enforce_runtime_rps_policy(
                cfg=cfg,
                mode="full",
                rows_per_second=0.2,
                rps_source="default_fallback_0.2",
            )

    def test_full_mode_accepts_measured_rps_when_policy_requires_measured(self) -> None:
        cfg = {
            "runtime_estimator": {
                "require_measured_rps_for_full": True,
            }
        }
        self.mod00._enforce_runtime_rps_policy(
            cfg=cfg,
            mode="full",
            rows_per_second=3.2,
            rps_source="stage00a_report",
        )

    def test_runtime_refresh_recomputes_baseline_count_after_stage00a(self) -> None:
        src = SCRIPT00.read_text(encoding="utf-8")
        self.assertIn("baseline_prompt_count_current", src, msg="orchestrator report must expose current baseline count")
        self.assertRegex(
            src,
            re.compile(r"def _refresh_runtime_decisions\(\).*?_baseline_prompt_count\(args\.run_id\)", re.DOTALL),
            msg="refresh path must recompute baseline prompt count, not use stale startup value",
        )

    def test_runtime_rows_per_second_prefers_stage00a_measurement_over_pilot(self) -> None:
        run_id = "test_v2_runtime_stage00a_precedence"
        run_root = REPO_ROOT / "runs" / run_id
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            (run_root / "v2").mkdir(parents=True, exist_ok=True)
            (run_root / "pilot").mkdir(parents=True, exist_ok=True)
            (run_root / "v2" / "00a_generate_baseline_outputs.report.json").write_text(
                json.dumps({"rows_per_second": 3.25}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (run_root / "pilot" / "qwen_pilot_report.json").write_text(
                json.dumps({"sample_size": 100, "timing_seconds": {"inference": 10}}, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            rps, src = self.mod00._runtime_rows_per_second(run_id)
            self.assertAlmostEqual(float(rps), 3.25, places=6)
            self.assertEqual(str(src), "stage00a_report")
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)


if __name__ == "__main__":
    unittest.main()
