import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sow.v2.baseline_inference import run_baseline_for_model  # noqa: E402
from sow.v2.inference_firewall import assert_inference_allowed  # noqa: E402


def _load_script(path: Path, name: str):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load script: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestInferenceFirewallContract(unittest.TestCase):
    def test_assert_inference_allowed_blocks_without_opt_in(self) -> None:
        with patch.dict(os.environ, {"SOW_ALLOW_INFERENCE": "0"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "SOW_ALLOW_INFERENCE=1"):
                assert_inference_allowed("unit_test_stage")

    def test_assert_inference_allowed_passes_with_opt_in(self) -> None:
        with patch.dict(os.environ, {"SOW_ALLOW_INFERENCE": "1"}, clear=False):
            assert_inference_allowed("unit_test_stage")

    def test_baseline_inference_function_fails_closed(self) -> None:
        with patch.dict(os.environ, {"SOW_ALLOW_INFERENCE": "0"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "SOW_ALLOW_INFERENCE=1"):
                run_baseline_for_model(
                    run_id="dummy",
                    model={"model_id": "Qwen/Qwen2.5-7B-Instruct", "revision": "r0"},
                    manifest_rows=[],
                    out_path=REPO_ROOT / "tmp" / "nonexistent.jsonl",
                    resume=False,
                    checkpoint_every_prompts=1,
                    batch_chain=[1],
                )

    def test_stage_scripts_fail_closed_before_inference(self) -> None:
        script00a = _load_script(REPO_ROOT / "scripts" / "v2" / "00a_generate_baseline_outputs.py", "stage00a_firewall")
        script07 = _load_script(REPO_ROOT / "scripts" / "v2" / "07_run_tracing.py", "stage07_firewall")
        with patch.dict(os.environ, {"SOW_ALLOW_INFERENCE": "0"}, clear=False):
            with patch.object(sys, "argv", ["prog", "--run-id", "dummy"]):
                with self.assertRaisesRegex(RuntimeError, "SOW_ALLOW_INFERENCE=1"):
                    script00a.main()
            with patch.object(sys, "argv", ["prog", "--run-id", "dummy"]):
                with self.assertRaisesRegex(RuntimeError, "SOW_ALLOW_INFERENCE=1"):
                    script07.main()

    def test_analysis_script_runs_with_firewall_off(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_json = Path(td) / "defs.json"
            env = dict(os.environ)
            env["SOW_ALLOW_INFERENCE"] = "0"
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit/extract_definitions.py",
                    "--out-json",
                    str(out_json),
                ],
                cwd=str(REPO_ROOT),
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stderr={proc.stderr}\nstdout={proc.stdout}")
            self.assertTrue(out_json.exists())


if __name__ == "__main__":
    unittest.main()

