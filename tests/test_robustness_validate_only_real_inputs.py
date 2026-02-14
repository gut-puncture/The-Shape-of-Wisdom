import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


from sow.manifest.canonicalize import canonicalize_robustness_manifest_v2  # noqa: E402


class TestRobustnessValidateOnlyRealInputs(unittest.TestCase):
    def test_validate_only_detects_known_issues(self) -> None:
        in_path = REPO_ROOT / "data" / "experiment_inputs" / "robustness_prompts_v2.jsonl"
        if not in_path.exists():
            self.skipTest("paid input file not present (data/ is not committed)")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            out_path = td / "robust_out.jsonl"
            meta_path = td / "robust_meta.json"
            report_path = td / "robust_report.json"

            with self.assertRaises(ValueError):
                canonicalize_robustness_manifest_v2(
                    run_id="unit_test_run",
                    input_path=in_path,
                    output_path=out_path,
                    meta_path=meta_path,
                    report_path=report_path,
                    repair_missing=False,
                )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["kind"], "robustness_manifest_v2_validate_only")
            self.assertEqual(report["total_input_rows"], 61257)
            self.assertEqual(report["duplicates_dropped"], 1258)
            self.assertEqual(report["filtered_out"], 0)
            self.assertEqual(report["unique_keys_after_filter_and_dedupe"], 59999)
            self.assertEqual(report["missing_pairs_total"], 1)
            self.assertIn(
                {"example_id": "mmlu::test::12183", "wrapper_id": "ascii_box"},
                report["missing_pairs_sample"],
            )
