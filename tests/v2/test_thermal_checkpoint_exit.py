import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sow.thermal.thermal_governor import ThermalGovernor, ThermalHygieneConfig  # noqa: E402


class TestThermalCheckpointExit(unittest.TestCase):
    def test_checkpoint_exit_mode_does_not_sleep(self) -> None:
        cfg = ThermalHygieneConfig(
            enabled=True,
            provider="powermetrics_thermal_pressure",
            cutoff_level="serious",
            cooldown_seconds=60,
            check_interval_seconds=1,
            pause_mode="checkpoint_exit",
        )
        with tempfile.TemporaryDirectory() as td:
            events = Path(td) / "thermal.jsonl"
            slept = []

            gov = ThermalGovernor(
                cfg=cfg,
                events_path=events,
                read_level_fn=lambda: "serious",
                time_fn=lambda: 0.0,
                sleep_fn=lambda dt: slept.append(float(dt)),
            )
            out = gov.maybe_cooldown(stage="stage13_baseline", model_id="m", model_revision="r")
            self.assertTrue(out.get("checkpoint_exit"))
            self.assertEqual(slept, [])


if __name__ == "__main__":
    unittest.main()
