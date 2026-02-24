import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_00B = REPO_ROOT / "scripts" / "v2" / "00b_prompt_sweep_benchmark.py"


def _load_stage00b_module():
    if str(SCRIPT_00B.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT_00B.parent))
    spec = importlib.util.spec_from_file_location("sow_v2_stage00b", SCRIPT_00B)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module: {SCRIPT_00B}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestStage00bPromptSweepBenchmarkContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_stage00b_module()

    def test_parse_prompt_counts_rejects_nonpositive_and_duplicates(self) -> None:
        self.assertEqual(self.mod._parse_prompt_counts("5,20,50"), [5, 20, 50])
        with self.assertRaises(ValueError):
            self.mod._parse_prompt_counts("5,0,50")
        with self.assertRaises(ValueError):
            self.mod._parse_prompt_counts("5,20,5")

    def test_build_stage00a_command_uses_resume_and_thermal_wrapper(self) -> None:
        cmd = self.mod._build_stage00a_command(
            python_bin="/usr/bin/python3",
            run_id="bench_p5",
            config_path=Path("/tmp/cfg.yaml"),
            model_name="qwen2.5-7b-instruct",
            prompt_count=5,
            cooldown_seconds=900,
            done_sentinel=Path("/tmp/bench.done"),
        )
        self.assertGreater(len(cmd), 0)
        self.assertEqual(cmd[0], str(self.mod.REPO_ROOT / "scripts" / "v2" / "run_with_thermal_resume.sh"))
        self.assertIn("--done-sentinel", cmd)
        self.assertIn("--cooldown-seconds", cmd)
        self.assertIn("--resume", cmd)
        self.assertIn("--max-prompts", cmd)
        self.assertIn("5", cmd)
        self.assertIn("--model-name", cmd)
        self.assertIn("qwen2.5-7b-instruct", cmd)

    def test_run_prompt_sweep_fails_closed_when_report_missing(self) -> None:
        mod = self.mod
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            (tmp_root / "scripts" / "v2").mkdir(parents=True, exist_ok=True)
            (tmp_root / "runs").mkdir(parents=True, exist_ok=True)
            cfg = tmp_root / "cfg.yaml"
            cfg.write_text("models: []\n", encoding="utf-8")
            old_root = mod.REPO_ROOT
            mod.REPO_ROOT = tmp_root
            try:
                rc, report = mod.run_prompt_sweep(
                    run_id_prefix="bench_missing_report",
                    prompt_counts=[5],
                    config_path=cfg,
                    model_name=None,
                    cooldown_seconds=120,
                    python_bin="/usr/bin/python3",
                    exec_fn=lambda argv: 0,
                )
                self.assertEqual(rc, 2)
                self.assertFalse(bool(report.get("pass")))
                failures = report.get("failures") or []
                self.assertTrue(any("missing stage00a report" in str(x) for x in failures), msg=failures)
                out_report = tmp_root / "runs" / "bench_missing_report" / "v2" / "00b_prompt_sweep_benchmark.report.json"
                self.assertTrue(out_report.exists(), msg="benchmark sweep must emit report even on failure")
            finally:
                mod.REPO_ROOT = old_root

    def test_run_prompt_sweep_emits_scaling_summary(self) -> None:
        mod = self.mod
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            (tmp_root / "scripts" / "v2").mkdir(parents=True, exist_ok=True)
            (tmp_root / "runs").mkdir(parents=True, exist_ok=True)
            cfg = tmp_root / "cfg.yaml"
            cfg.write_text("models: []\n", encoding="utf-8")
            old_root = mod.REPO_ROOT
            mod.REPO_ROOT = tmp_root
            rps_by_count = {5: 2.0, 20: 1.5, 50: 1.2}
            model_rps = {
                5: {"Qwen/Qwen2.5-7B-Instruct": 2.0, "meta-llama/Llama-3.1-8B-Instruct": 1.6},
                20: {"Qwen/Qwen2.5-7B-Instruct": 1.5, "meta-llama/Llama-3.1-8B-Instruct": 1.1},
                50: {"Qwen/Qwen2.5-7B-Instruct": 1.2, "meta-llama/Llama-3.1-8B-Instruct": 0.8},
            }

            def _fake_exec(argv):
                rid = argv[argv.index("--run-id") + 1]
                count = int(argv[argv.index("--max-prompts") + 1])
                out_root = tmp_root / "runs" / rid / "v2"
                out_root.mkdir(parents=True, exist_ok=True)
                stats = {k: {"rows_per_second": float(v)} for k, v in model_rps[count].items()}
                report = {
                    "pass": True,
                    "rows_per_second": float(rps_by_count[count]),
                    "manifest": {"rows": int(count)},
                    "stats_per_model": stats,
                }
                (out_root / "00a_generate_baseline_outputs.report.json").write_text(
                    json.dumps(report, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                return 0

            try:
                rc, report = mod.run_prompt_sweep(
                    run_id_prefix="bench_success",
                    prompt_counts=[5, 20, 50],
                    config_path=cfg,
                    model_name=None,
                    cooldown_seconds=120,
                    python_bin="/usr/bin/python3",
                    exec_fn=_fake_exec,
                )
                self.assertEqual(rc, 0, msg=report)
                self.assertTrue(bool(report.get("pass")))
                entries = report.get("entries") or []
                self.assertEqual(len(entries), 3)
                self.assertAlmostEqual(float(entries[0]["rows_per_second"]), 2.0, places=6)
                self.assertAlmostEqual(float(entries[2]["rows_per_second"]), 1.2, places=6)
                diagnostics = report.get("diagnostics") or {}
                self.assertTrue(bool(diagnostics.get("monotonic_nonincreasing_rows_per_second")))
                self.assertAlmostEqual(float(diagnostics.get("rows_per_second_5_to_50_ratio")), 0.6, places=6)
                ll_ratio = float((diagnostics.get("per_model_rows_per_second_5_to_50_ratio") or {})["meta-llama/Llama-3.1-8B-Instruct"])
                self.assertAlmostEqual(ll_ratio, 0.5, places=6)
            finally:
                mod.REPO_ROOT = old_root


if __name__ == "__main__":
    unittest.main()
