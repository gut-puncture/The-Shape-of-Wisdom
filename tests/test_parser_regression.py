import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


from sow.judging.deterministic_parser import parse_choice  # noqa: E402


class TestParserRegression(unittest.TestCase):
    def test_regression_cases_exact_match(self) -> None:
        reg_path = REPO_ROOT / "artifacts" / "parser_edge_case_regression" / "regression_cases.json"
        cases = json.loads(reg_path.read_text(encoding="utf-8"))

        failures = []
        for case in cases:
            got = parse_choice(
                response_text=case["response_text"],
                first_token=case.get("first_token"),
                options=case["options"],
            )
            exp = case["expected"]
            if got["parsed_choice"] != exp["choice"] or got["decision"] != exp["decision"]:
                failures.append(
                    {
                        "case_id": case["case_id"],
                        "expected": exp,
                        "got": {"choice": got["parsed_choice"], "decision": got["decision"]},
                    }
                )

        if failures:
            sample = failures[:10]
            self.fail(f"{len(failures)} regression cases failed; sample={sample}")

