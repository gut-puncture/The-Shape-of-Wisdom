import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


from sow.ccc.build_ccc import check_ccc_gates, compute_ccc_intersection, compute_retention_metrics  # noqa: E402


class TestCccGates(unittest.TestCase):
    def test_compute_ccc_intersection(self) -> None:
        per = {"m1": {"a", "b", "c"}, "m2": {"b", "c", "d"}}
        inter = compute_ccc_intersection({k: set(v) for k, v in per.items()})
        self.assertEqual(inter, ["b", "c"])

    def test_ccc_gate_pass(self) -> None:
        # Two domains; PCC totals are 10 each model; CCC has 8 => 0.8 overall retention.
        domain_by_example = {f"d1::{i}": "d1" for i in range(5)}
        domain_by_example.update({f"d2::{i}": "d2" for i in range(5)})
        ccc = set([f"d1::{i}" for i in range(4)] + [f"d2::{i}" for i in range(4)])

        per_model_counts = {"m1": {"d1": 5, "d2": 5}, "m2": {"d1": 5, "d2": 5}}
        retention = compute_retention_metrics(
            ccc_example_ids=ccc,
            per_model_counts_by_domain=per_model_counts,
            domain_by_example=domain_by_example,
        )
        ok, reasons = check_ccc_gates(retention_metrics=retention, min_overall=0.80, min_per_domain=0.60)
        self.assertTrue(ok)
        self.assertEqual(reasons, [])

    def test_ccc_gate_fail_per_domain(self) -> None:
        domain_by_example = {"a": "d1", "b": "d1", "c": "d2", "d": "d2"}
        ccc = {"a"}  # d2 drops to 0%
        per_model_counts = {"m1": {"d1": 2, "d2": 2}}
        retention = compute_retention_metrics(
            ccc_example_ids=ccc,
            per_model_counts_by_domain=per_model_counts,
            domain_by_example=domain_by_example,
        )
        ok, reasons = check_ccc_gates(retention_metrics=retention, min_overall=0.80, min_per_domain=0.60)
        self.assertFalse(ok)
        self.assertTrue(any("d2" in r for r in reasons))

