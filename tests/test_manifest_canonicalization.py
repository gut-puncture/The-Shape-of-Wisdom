import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


from sow.constants import ANSWER_SUFFIX, EXPECTED_ROBUSTNESS_WRAPPER_IDS_V2  # noqa: E402
from sow.io_jsonl import iter_jsonl  # noqa: E402
from sow.manifest.canonicalize import canonicalize_robustness_manifest_v2  # noqa: E402


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")


class TestManifestCanonicalization(unittest.TestCase):
    def test_robustness_keep_last_and_repair_missing_ascii_box(self) -> None:
        ex = "mmlu::test::12183"
        wrappers = list(EXPECTED_ROBUSTNESS_WRAPPER_IDS_V2)
        self.assertIn("ascii_box", wrappers)

        options = {"A": "optA", "B": "optB", "C": "optC", "D": "optD"}
        question = "What is 2+2?"

        rows = []
        for wid in wrappers:
            if wid == "ascii_box":
                continue
            if wid == "csv_inline":
                continue
            rows.append(
                {
                    "prompt_id": f"p::{wid}",
                    "module": "unit_test",
                    "dataset": "mmlu",
                    "split": "test",
                    "example_id": ex,
                    "subject": "unit",
                    "coarse_domain": "math",
                    "question": question,
                    "options": options,
                    "correct_key": "A",
                    "wrapper_id": wid,
                    "wrapper_description": f"{wid} wrapper",
                    "prompt_text": f"{wid} prompt body\n",
                }
            )

        # Add a duplicate key where later line should win.
        rows.insert(
            0,
            {
                "prompt_id": "old_csv",
                "module": "unit_test_old",
                "dataset": "mmlu",
                "split": "test",
                "example_id": ex,
                "subject": "unit",
                "coarse_domain": "math",
                "question": question,
                "options": options,
                "correct_key": "A",
                "wrapper_id": "csv_inline",
                "wrapper_description": "old csv wrapper",
                "prompt_text": "OLD CSV BODY\n",
            },
        )
        rows.append(
            {
                "prompt_id": "new_csv",
                "module": "unit_test_new",
                "dataset": "mmlu",
                "split": "test",
                "example_id": ex,
                "subject": "unit",
                "coarse_domain": "math",
                "question": question,
                "options": options,
                "correct_key": "A",
                "wrapper_id": "csv_inline",
                "wrapper_description": "new csv wrapper",
                "prompt_text": "NEW CSV BODY\n",
            }
        )

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            in_path = td / "robust_in.jsonl"
            out_path = td / "robust_out.jsonl"
            meta_path = td / "robust_meta.json"
            report_path = td / "robust_report.json"
            _write_jsonl(in_path, rows)

            canonicalize_robustness_manifest_v2(
                run_id="unit_test_run",
                input_path=in_path,
                output_path=out_path,
                meta_path=meta_path,
                report_path=report_path,
                repair_missing=True,
            )

            out_rows = list(iter_jsonl(out_path))
            self.assertEqual(len(out_rows), 20)
            for r in out_rows:
                self.assertTrue(r["prompt_text"].endswith(ANSWER_SUFFIX))
                self.assertIn("Return only the letter", r["prompt_text"])

            csv_rows = [r for r in out_rows if r["wrapper_id"] == "csv_inline"]
            self.assertEqual(len(csv_rows), 1)
            self.assertEqual(csv_rows[0]["prompt_id"], "new_csv")
            self.assertIn("NEW CSV BODY", csv_rows[0]["prompt_text"])

            ascii_rows = [r for r in out_rows if r["wrapper_id"] == "ascii_box"]
            self.assertEqual(len(ascii_rows), 1)
            self.assertEqual(ascii_rows[0]["prompt_id"], f"robustness::ascii_box::{ex}")

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["duplicates_dropped"], 1)
            self.assertEqual(report["missing_repairs"], 1)
            self.assertEqual(report["missing_pairs_before_repair_total"], 1)
