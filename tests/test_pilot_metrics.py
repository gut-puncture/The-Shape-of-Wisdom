import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


from sow.pilot.pilot_inference import _accumulate_metrics  # noqa: E402


class TestPilotMetrics(unittest.TestCase):
    def test_accumulate_metrics_uses_nested_parser_parsed_choice(self) -> None:
        rows = [
            {"one_token_compliance": True, "parser": {"parsed_choice": "A"}, "is_correct": True},
            {"one_token_compliance": False, "parser": {"parsed_choice": None}, "is_correct": None},
            {"one_token_compliance": True, "parser": {"parsed_choice": "B"}, "is_correct": False},
        ]
        m = _accumulate_metrics(rows)
        self.assertEqual(m["n"], 3)
        self.assertAlmostEqual(m["one_token_compliance_rate"], 2 / 3)
        self.assertAlmostEqual(m["parser_resolved_rate"], 2 / 3)
        self.assertEqual(m["unresolved"], 1)
        self.assertAlmostEqual(m["accuracy_on_resolved"], 1 / 2)

