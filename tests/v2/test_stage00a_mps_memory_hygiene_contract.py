from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_INFERENCE = REPO_ROOT / "src" / "sow" / "v2" / "baseline_inference.py"


class TestStage00aMpsMemoryHygieneContract(unittest.TestCase):
    def test_run_baseline_releases_mps_cache_before_and_after_model_use(self) -> None:
        text = BASELINE_INFERENCE.read_text(encoding="utf-8")
        self.assertIn(
            "torch.mps.empty_cache()",
            text,
            msg="baseline inference must call torch.mps.empty_cache for Apple Silicon unified-memory hygiene",
        )
        self.assertGreaterEqual(
            text.count("torch.mps.empty_cache()"),
            2,
            msg="baseline inference must clear MPS cache both before model load and after model teardown",
        )


if __name__ == "__main__":
    unittest.main()
