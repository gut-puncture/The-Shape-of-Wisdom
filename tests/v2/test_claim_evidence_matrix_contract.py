import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER = REPO_ROOT / "docs" / "audit" / "V3_REQUIREMENT_LEDGER.yaml"
MATRIX = REPO_ROOT / "docs" / "audit" / "CLAIM_EVIDENCE_MATRIX_TEMPLATE.md"
COVERAGE = REPO_ROOT / "docs" / "audit" / "V3_EDGE_CASE_COVERAGE_MAP.md"


class TestClaimEvidenceMatrixContract(unittest.TestCase):
    def test_requirement_ids_001_to_014_exist(self) -> None:
        payload = yaml.safe_load(LEDGER.read_text(encoding="utf-8")) or {}
        reqs = payload.get("requirements") or []
        ids = {str(r.get("requirement_id") or "") for r in reqs}
        expected = {f"RQ-{i:03d}" for i in range(1, 15)}
        self.assertEqual(ids, expected)

    def test_template_contains_all_requirement_rows(self) -> None:
        text = MATRIX.read_text(encoding="utf-8")
        for rid in [f"RQ-{i:03d}" for i in range(1, 15)]:
            self.assertIn(rid, text, msg=f"missing {rid} row in claim-evidence matrix template")

    def test_template_defines_status_rules(self) -> None:
        text = MATRIX.read_text(encoding="utf-8")
        self.assertIn("Status rules", text)
        for token in ["`pass`", "`fail`", "`partial`", "`gap`"]:
            self.assertIn(token, text)

    def test_edge_case_coverage_map_exists_and_has_normalized_key_column(self) -> None:
        self.assertTrue(COVERAGE.exists(), msg=f"missing edge-case coverage map: {COVERAGE}")
        text = COVERAGE.read_text(encoding="utf-8")
        self.assertIn("normalized_key", text)
        self.assertIn("edge_case_id", text)


if __name__ == "__main__":
    unittest.main()
