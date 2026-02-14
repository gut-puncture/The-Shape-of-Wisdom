import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


from sow.manifest.canonicalize import compute_manifest_row_sha256  # noqa: E402
from sow.pcc.build_pcc import select_pcc_example_ids  # noqa: E402


def _mk_row(*, ex: str, wid: str, domain: str, pid: str, prompt_text: str) -> dict:
    row = {
        "example_id": ex,
        "wrapper_id": wid,
        "prompt_id": pid,
        "prompt_uid": pid,
        "prompt_text": prompt_text,
        "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
        "correct_key": "A",
        "coarse_domain": domain,
    }
    row["manifest_sha256"] = compute_manifest_row_sha256(row)
    return row


class TestPccSelection(unittest.TestCase):
    def test_select_pcc_example_ids_filters_and_stratifies(self) -> None:
        expected_wrappers = ["w1", "w2"]
        baseline_rows = [
            _mk_row(ex="e1", wid="plain_exam", domain="d1", pid="b1", prompt_text="x\nAnswer: "),
            _mk_row(ex="e2", wid="plain_exam", domain="d2", pid="b2", prompt_text="x\nAnswer: "),
        ]
        # Robustness: e1 is complete; e2 is missing w2 -> should be dropped.
        robustness_rows = [
            _mk_row(ex="e1", wid="w1", domain="d1", pid="r11", prompt_text="short\nAnswer: "),
            _mk_row(ex="e1", wid="w2", domain="d1", pid="r12", prompt_text="short\nAnswer: "),
            _mk_row(ex="e2", wid="w1", domain="d2", pid="r21", prompt_text="short\nAnswer: "),
        ]

        # Token counts: make e1 ok, e2 irrelevant (wrapper mismatch).
        def token_count_fn(text: str) -> int:
            return 5 if "short" in text else 5

        out = select_pcc_example_ids(
            baseline_rows=baseline_rows,
            robustness_rows=robustness_rows,
            expected_wrapper_ids=expected_wrappers,
            token_count_fn=token_count_fn,
            max_input_tokens=10,
            target_size=1,
            seed=123,
            salt="m",
        )
        self.assertEqual(out["selected_example_ids"], ["e1"])
        self.assertEqual(out["dropped_wrapper_mismatch_total"], 1)

    def test_select_pcc_example_ids_drops_too_long(self) -> None:
        expected_wrappers = ["w1", "w2"]
        baseline_rows = [
            _mk_row(ex="e1", wid="plain_exam", domain="d1", pid="b1", prompt_text="x\nAnswer: "),
            _mk_row(ex="e2", wid="plain_exam", domain="d2", pid="b2", prompt_text="x\nAnswer: "),
        ]
        robustness_rows = [
            _mk_row(ex="e1", wid="w1", domain="d1", pid="r11", prompt_text="ok\nAnswer: "),
            _mk_row(ex="e1", wid="w2", domain="d1", pid="r12", prompt_text="ok\nAnswer: "),
            _mk_row(ex="e2", wid="w1", domain="d2", pid="r21", prompt_text="ok\nAnswer: "),
            _mk_row(ex="e2", wid="w2", domain="d2", pid="r22", prompt_text="LONG\nAnswer: "),
        ]

        def token_count_fn(text: str) -> int:
            return 50 if "LONG" in text else 5

        out = select_pcc_example_ids(
            baseline_rows=baseline_rows,
            robustness_rows=robustness_rows,
            expected_wrapper_ids=expected_wrappers,
            token_count_fn=token_count_fn,
            max_input_tokens=10,
            target_size=1,
            seed=123,
            salt="m",
        )
        self.assertEqual(out["selected_example_ids"], ["e1"])
        self.assertEqual(out["dropped_too_long_total"], 1)

    def test_select_pcc_example_ids_deterministic(self) -> None:
        expected_wrappers = ["w1", "w2"]
        baseline_rows = [
            _mk_row(ex="e1", wid="plain_exam", domain="d1", pid="b1", prompt_text="x\nAnswer: "),
            _mk_row(ex="e2", wid="plain_exam", domain="d2", pid="b2", prompt_text="x\nAnswer: "),
        ]
        robustness_rows = [
            _mk_row(ex="e1", wid="w1", domain="d1", pid="r11", prompt_text="ok\nAnswer: "),
            _mk_row(ex="e1", wid="w2", domain="d1", pid="r12", prompt_text="ok\nAnswer: "),
            _mk_row(ex="e2", wid="w1", domain="d2", pid="r21", prompt_text="ok\nAnswer: "),
            _mk_row(ex="e2", wid="w2", domain="d2", pid="r22", prompt_text="ok\nAnswer: "),
        ]

        def token_count_fn(text: str) -> int:
            return 5

        o1 = select_pcc_example_ids(
            baseline_rows=baseline_rows,
            robustness_rows=robustness_rows,
            expected_wrapper_ids=expected_wrappers,
            token_count_fn=token_count_fn,
            max_input_tokens=10,
            target_size=2,
            seed=123,
            salt="m",
        )
        o2 = select_pcc_example_ids(
            baseline_rows=baseline_rows,
            robustness_rows=robustness_rows,
            expected_wrapper_ids=expected_wrappers,
            token_count_fn=token_count_fn,
            max_input_tokens=10,
            target_size=2,
            seed=123,
            salt="m",
        )
        self.assertEqual(o1["selected_example_ids"], o2["selected_example_ids"])
