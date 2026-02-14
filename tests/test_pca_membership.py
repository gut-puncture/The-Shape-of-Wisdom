import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


from sow.io_jsonl import iter_jsonl, write_jsonl  # noqa: E402
from sow.pca.membership import select_pca_membership  # noqa: E402


class TestPcaMembership(unittest.TestCase):
    def test_stratified_membership_is_deterministic_and_balanced(self) -> None:
        # 4 strata: (wrapper_id, coarse_domain) in {plain_exam, ascii_box} x {math, bio}
        baseline_rows = [
            {"prompt_uid": "b1", "prompt_id": "b1", "example_id": "e1", "wrapper_id": "plain_exam", "coarse_domain": "math"},
            {"prompt_uid": "b2", "prompt_id": "b2", "example_id": "e2", "wrapper_id": "plain_exam", "coarse_domain": "math"},
            {"prompt_uid": "b3", "prompt_id": "b3", "example_id": "e3", "wrapper_id": "plain_exam", "coarse_domain": "bio"},
            {"prompt_uid": "b4", "prompt_id": "b4", "example_id": "e4", "wrapper_id": "plain_exam", "coarse_domain": "bio"},
        ]
        robust_rows = [
            {"prompt_uid": "r1", "prompt_id": "r1", "example_id": "e1", "wrapper_id": "ascii_box", "coarse_domain": "math"},
            {"prompt_uid": "r2", "prompt_id": "r2", "example_id": "e2", "wrapper_id": "ascii_box", "coarse_domain": "math"},
            {"prompt_uid": "r3", "prompt_id": "r3", "example_id": "e3", "wrapper_id": "ascii_box", "coarse_domain": "bio"},
            {"prompt_uid": "r4", "prompt_id": "r4", "example_id": "e4", "wrapper_id": "ascii_box", "coarse_domain": "bio"},
        ]

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            bpath = td / "baseline.jsonl"
            rpath = td / "robust.jsonl"
            write_jsonl(bpath, baseline_rows)
            write_jsonl(rpath, robust_rows)

            s1 = select_pca_membership(baseline_manifest=bpath, robustness_manifest=rpath, sample_size=4, seed=123)
            s2 = select_pca_membership(baseline_manifest=bpath, robustness_manifest=rpath, sample_size=4, seed=123)
            self.assertEqual(s1["membership"], s2["membership"])
            self.assertEqual(s1["sample_size"], 4)
            self.assertEqual(s1["n_strata"], 4)

            # One per stratum.
            self.assertTrue(all(v == 1 for v in s1["counts_by_stratum"].values()))
            self.assertEqual(len(s1["counts_by_stratum"]), 4)

            # Output is sorted by prompt_uid.
            uids = [m["prompt_uid"] for m in s1["membership"]]
            self.assertEqual(uids, sorted(uids))

