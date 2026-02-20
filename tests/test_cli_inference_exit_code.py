import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sow.cli import inference_exit_code_from_model_results  # noqa: E402


class TestInferenceExitCode(unittest.TestCase):
    def test_all_pass_returns_zero(self) -> None:
        self.assertEqual(inference_exit_code_from_model_results([{"pass": True}, {"pass": True}]), 0)

    def test_thermal_checkpoint_returns_95(self) -> None:
        self.assertEqual(
            inference_exit_code_from_model_results(
                [
                    {"pass": False, "stopped_early": True, "stop_reason": "thermal_checkpoint"},
                    {"pass": False, "stopped_early": True, "stop_reason": "thermal_checkpoint"},
                ]
            ),
            95,
        )

    def test_non_thermal_failure_returns_2(self) -> None:
        self.assertEqual(
            inference_exit_code_from_model_results([{"pass": True}, {"pass": False, "stop_reason": "validation_failed"}]),
            2,
        )

    def test_mixed_thermal_and_non_thermal_returns_2(self) -> None:
        self.assertEqual(
            inference_exit_code_from_model_results(
                [
                    {"pass": False, "stopped_early": True, "stop_reason": "thermal_checkpoint"},
                    {"pass": False, "stop_reason": "validation_failed"},
                ]
            ),
            2,
        )


if __name__ == "__main__":
    unittest.main()
