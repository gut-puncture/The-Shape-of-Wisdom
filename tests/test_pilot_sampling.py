import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


from sow.io_jsonl import write_jsonl  # noqa: E402
from sow.pilot.pilot_inference import select_pilot_rows  # noqa: E402


class TestPilotSampling(unittest.TestCase):
    def test_select_pilot_rows_stratified_and_deterministic(self) -> None:
        rows = []
        # Two domains: d1 has 10 rows, d2 has 10 rows. Sample 6 => 3 each.
        for i in range(10):
            rows.append(
                {
                    "prompt_uid": f"d1::{i}",
                    "prompt_id": f"d1::{i}",
                    "example_id": f"e1::{i}",
                    "wrapper_id": "plain_exam",
                    "coarse_domain": "d1",
                    "prompt_text": "x",
                    "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                    "correct_key": "A",
                    "manifest_sha256": "0" * 64,
                }
            )
        for i in range(10):
            rows.append(
                {
                    "prompt_uid": f"d2::{i}",
                    "prompt_id": f"d2::{i}",
                    "example_id": f"e2::{i}",
                    "wrapper_id": "plain_exam",
                    "coarse_domain": "d2",
                    "prompt_text": "x",
                    "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                    "correct_key": "A",
                    "manifest_sha256": "0" * 64,
                }
            )

        tmp = REPO_ROOT / "tests" / "_tmp_pilot_manifest.jsonl"
        try:
            write_jsonl(tmp, rows)
            s1 = select_pilot_rows(baseline_manifest_path=tmp, sample_size=6, seed=123)
            s2 = select_pilot_rows(baseline_manifest_path=tmp, sample_size=6, seed=123)
            self.assertEqual([r["prompt_uid"] for r in s1], [r["prompt_uid"] for r in s2])

            d1 = sum(1 for r in s1 if r["coarse_domain"] == "d1")
            d2 = sum(1 for r in s1 if r["coarse_domain"] == "d2")
            self.assertEqual(d1, 3)
            self.assertEqual(d2, 3)
        finally:
            if tmp.exists():
                tmp.unlink()

