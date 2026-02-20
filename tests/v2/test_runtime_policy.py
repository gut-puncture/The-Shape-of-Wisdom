import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sow.v2.runtime_policy import choose_backend, estimate_runtime  # noqa: E402


class TestRuntimePolicy(unittest.TestCase):
    def test_estimate_and_backend_threshold(self) -> None:
        est = estimate_runtime(task_name="trace", rows_per_second=0.5, prompts_per_model=3000, model_count=3)
        self.assertGreater(est.estimated_hours_all_models, 0.0)
        self.assertEqual(choose_backend(estimated_hours_all_models=est.estimated_hours_all_models, threshold_hours=1.0), "gpu")
        self.assertEqual(choose_backend(estimated_hours_all_models=est.estimated_hours_all_models, threshold_hours=1000.0), "mac")


if __name__ == "__main__":
    unittest.main()
