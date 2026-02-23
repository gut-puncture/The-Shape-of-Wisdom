import inspect
import sys
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = REPO_ROOT / "configs" / "experiment_v2.yaml"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "v2"))


import _common  # noqa: E402,F401
import importlib.util  # noqa: E402


SCRIPT07 = REPO_ROOT / "scripts" / "v2" / "07_run_tracing.py"


def _load_stage07_module():
    spec = importlib.util.spec_from_file_location("s07_checkpoint_contract", SCRIPT07)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {SCRIPT07}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestStage07CheckpointFlushContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod07 = _load_stage07_module()

    def test_stage07_reads_checkpoint_interval_from_config(self) -> None:
        src = inspect.getsource(self.mod07.main)
        self.assertIn("stage07_checkpoint_every_prompts", src)
        self.assertIn("pending_prompts_since_checkpoint >= checkpoint_every_prompts", src)
        self.assertIn("_flush_checkpoint()", src)

    def test_stage07_default_checkpoint_interval_is_strict(self) -> None:
        cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
        execution_cfg = cfg.get("execution") or {}
        interval = int(execution_cfg.get("stage07_checkpoint_every_prompts", 0))
        self.assertLessEqual(
            interval,
            10,
            msg="stage07 checkpoint interval default must be strict to minimize crash-loss window",
        )


if __name__ == "__main__":
    unittest.main()
