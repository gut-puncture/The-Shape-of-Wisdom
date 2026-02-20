import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


from sow.thermal.thermal_governor import ThermalGovernor, ThermalHygieneConfig  # noqa: E402


class TestThermalGovernor(unittest.TestCase):
    def test_from_cfg_none_is_disabled_for_back_compat(self) -> None:
        cfg = ThermalHygieneConfig.from_cfg(None)
        self.assertFalse(cfg.enabled)

    def test_no_cooldown_below_cutoff(self) -> None:
        th = ThermalHygieneConfig(
            enabled=True,
            provider="powermetrics_thermal_pressure",
            cutoff_level="serious",
            cooldown_seconds=1200,
            check_interval_seconds=999,
            pause_mode="sleep",
        )
        with tempfile.TemporaryDirectory() as td:
            events = Path(td) / "events.jsonl"
            levels = ["nominal"]

            def read_level():
                return levels[0]

            t = [0.0]

            def time_fn():
                return t[0]

            sleeps = []

            def sleep_fn(dt: float) -> None:
                sleeps.append(float(dt))

            gov = ThermalGovernor(cfg=th, events_path=events, read_level_fn=read_level, time_fn=time_fn, sleep_fn=sleep_fn)
            res = gov.maybe_cooldown(stage="pilot_inference", model_id="m", model_revision="r")
            self.assertTrue(res["checked"])
            self.assertFalse(res["cooled_down"])
            self.assertEqual(sleeps, [])

            lines = events.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            obj = json.loads(lines[0])
            self.assertEqual(obj["event"], "thermal_check")
            self.assertEqual(obj["level"], "nominal")

    def test_cooldown_at_or_above_cutoff(self) -> None:
        th = ThermalHygieneConfig(
            enabled=True,
            provider="powermetrics_thermal_pressure",
            cutoff_level="serious",
            cooldown_seconds=1200,
            check_interval_seconds=1,
            pause_mode="sleep",
        )
        with tempfile.TemporaryDirectory() as td:
            events = Path(td) / "events.jsonl"

            def read_level():
                return "serious"

            t = [0.0]

            def time_fn():
                return t[0]

            sleeps = []

            def sleep_fn(dt: float) -> None:
                sleeps.append(float(dt))

            gov = ThermalGovernor(cfg=th, events_path=events, read_level_fn=read_level, time_fn=time_fn, sleep_fn=sleep_fn)
            res = gov.maybe_cooldown(stage="pilot_inference", model_id="m", model_revision="r")
            self.assertTrue(res["checked"])
            self.assertTrue(res["cooled_down"])
            self.assertEqual(sleeps, [1200.0])

            lines = events.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 3)
            objs = [json.loads(x) for x in lines]
            self.assertEqual([o["event"] for o in objs], ["thermal_check", "cooldown_start", "cooldown_end"])
            self.assertEqual(objs[0]["level"], "serious")
            self.assertEqual(objs[1]["cooldown_seconds"], 1200)

    def test_check_interval_skips_extra_checks(self) -> None:
        th = ThermalHygieneConfig(
            enabled=True,
            provider="powermetrics_thermal_pressure",
            cutoff_level="critical",
            cooldown_seconds=1200,
            check_interval_seconds=60,
            pause_mode="sleep",
        )
        with tempfile.TemporaryDirectory() as td:
            events = Path(td) / "events.jsonl"

            def read_level():
                return "nominal"

            t = [0.0]

            def time_fn():
                return t[0]

            gov = ThermalGovernor(cfg=th, events_path=events, read_level_fn=read_level, time_fn=time_fn, sleep_fn=lambda _: None)
            res1 = gov.maybe_cooldown(stage="pilot_inference", model_id="m", model_revision="r")
            self.assertTrue(res1["checked"])

            # Same timestamp: should not check again.
            res2 = gov.maybe_cooldown(stage="pilot_inference", model_id="m", model_revision="r")
            self.assertFalse(res2["checked"])

            lines = events.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
