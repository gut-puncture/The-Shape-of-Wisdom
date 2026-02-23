import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RESET_SCRIPT = REPO_ROOT / "scripts" / "v2" / "reset_runtime_state.sh"
PREFLIGHT_SCRIPT = REPO_ROOT / "scripts" / "v2" / "preflight_full_run.sh"
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_v2.yaml"


class TestRuntimeHygieneScriptsContract(unittest.TestCase):
    def test_reset_runtime_state_clears_only_runtime_dirs(self) -> None:
        runtime_dirs = [
            REPO_ROOT / "runs",
            REPO_ROOT / "artifacts" / "final_result_v2",
            REPO_ROOT / "logs",
            REPO_ROOT / "sentinels",
        ]
        runtime_markers = []
        for d in runtime_dirs:
            d.mkdir(parents=True, exist_ok=True)
            marker = d / "test_runtime_hygiene_marker.tmp"
            marker.write_text("x\n", encoding="utf-8")
            runtime_markers.append(marker)

        preserve_path = REPO_ROOT / "data" / "test_runtime_hygiene_keep.tmp"
        preserve_path.parent.mkdir(parents=True, exist_ok=True)
        preserve_path.write_text("keep\n", encoding="utf-8")
        try:
            proc = subprocess.run(
                ["bash", str(RESET_SCRIPT)],
                cwd=str(REPO_ROOT),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            for marker in runtime_markers:
                self.assertFalse(marker.exists(), msg=f"runtime marker should be deleted: {marker}")
            for d in runtime_dirs:
                self.assertTrue(d.exists(), msg=f"runtime dir should still exist: {d}")
            self.assertTrue(preserve_path.exists(), msg="non-runtime file must remain untouched")
        finally:
            if preserve_path.exists():
                preserve_path.unlink()

    def test_preflight_full_run_passes_for_repo_config(self) -> None:
        proc = subprocess.run(
            ["bash", str(PREFLIGHT_SCRIPT), "--config", str(CONFIG_PATH)],
            cwd=str(REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")

    def test_preflight_full_run_fails_on_prompt_pack_hash_mismatch(self) -> None:
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        cfg["data_scope"]["baseline_manifest_sha256"] = "deadbeef"
        with tempfile.TemporaryDirectory() as td:
            tmp_cfg = Path(td) / "cfg.yaml"
            tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            proc = subprocess.run(
                ["bash", str(PREFLIGHT_SCRIPT), "--config", str(tmp_cfg)],
                cwd=str(REPO_ROOT),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 2, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertIn("baseline_manifest_sha256 mismatch", proc.stderr + proc.stdout)

    def test_preflight_full_run_fails_when_manifest_source_key_missing(self) -> None:
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        cfg["data_scope"].pop("baseline_manifest_source", None)
        with tempfile.TemporaryDirectory() as td:
            tmp_cfg = Path(td) / "cfg.yaml"
            tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            proc = subprocess.run(
                ["bash", str(PREFLIGHT_SCRIPT), "--config", str(tmp_cfg)],
                cwd=str(REPO_ROOT),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 2, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertIn("data_scope.baseline_manifest_source missing or empty", proc.stderr + proc.stdout)

    def test_preflight_full_run_fails_on_missing_required_doc(self) -> None:
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        cfg["experiment"]["objective_doc"] = str(REPO_ROOT / "docs" / "MISSING_OBJECTIVE_DOC.md")
        with tempfile.TemporaryDirectory() as td:
            tmp_cfg = Path(td) / "cfg.yaml"
            tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            proc = subprocess.run(
                ["bash", str(PREFLIGHT_SCRIPT), "--config", str(tmp_cfg)],
                cwd=str(REPO_ROOT),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 2, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertIn("required document missing", proc.stderr + proc.stdout)

    def test_preflight_full_run_fails_when_objective_doc_key_missing(self) -> None:
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        cfg["experiment"].pop("objective_doc", None)
        with tempfile.TemporaryDirectory() as td:
            tmp_cfg = Path(td) / "cfg.yaml"
            tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            proc = subprocess.run(
                ["bash", str(PREFLIGHT_SCRIPT), "--config", str(tmp_cfg)],
                cwd=str(REPO_ROOT),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 2, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertIn("experiment.objective_doc missing or empty", proc.stderr + proc.stdout)


if __name__ == "__main__":
    unittest.main()
