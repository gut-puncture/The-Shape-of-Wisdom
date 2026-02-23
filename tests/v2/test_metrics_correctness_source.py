import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


from sow.v2.metrics import build_decision_metrics_frame  # noqa: E402


def _row(*, prompt_uid: str, final_logits: dict[str, float], baseline_is_correct: bool, parsed_choice: str | None) -> dict:
    return {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": "r0",
        "prompt_uid": prompt_uid,
        "example_id": f"e_{prompt_uid}",
        "wrapper_id": "plain_exam",
        "coarse_domain": "biology",
        "is_correct": baseline_is_correct,
        "parsed_choice": parsed_choice,
        "layerwise": [
            {
                "candidate_logits": {"A": 0.1, "B": 0.2, "C": 0.0, "D": -0.1},
                "candidate_probs": {"A": 0.2, "B": 0.5, "C": 0.2, "D": 0.1},
                "candidate_entropy": 1.0,
                "top_candidate": "B",
            },
            {
                "candidate_logits": final_logits,
                "candidate_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
                "candidate_entropy": 1.2,
                "top_candidate": max(final_logits, key=final_logits.get),
            },
        ],
    }


class TestMetricsCorrectnessSource(unittest.TestCase):
    def test_parser_unresolved_but_final_top1_correct_marks_correct(self) -> None:
        rows = [
            _row(
                prompt_uid="u0",
                final_logits={"A": 2.0, "B": 1.0, "C": 0.0, "D": -1.0},
                baseline_is_correct=False,
                parsed_choice=None,
            )
        ]
        out = build_decision_metrics_frame(rows, correct_key_by_prompt_uid={"u0": "A"})
        self.assertFalse(out.empty)
        self.assertTrue(bool(out["is_correct"].all()), msg="is_correct must follow final-layer top1 logits, not parser/baseline text")

    def test_parser_mismatch_does_not_override_final_top1(self) -> None:
        rows = [
            _row(
                prompt_uid="u1",
                final_logits={"A": 3.0, "B": 0.5, "C": 0.0, "D": -1.0},
                baseline_is_correct=False,
                parsed_choice="B",
            )
        ]
        out = build_decision_metrics_frame(rows, correct_key_by_prompt_uid={"u1": "A"})
        self.assertFalse(out.empty)
        self.assertTrue(bool(out["is_correct"].all()))

    def test_baseline_true_but_final_top1_wrong_marks_incorrect(self) -> None:
        rows = [
            _row(
                prompt_uid="u2",
                final_logits={"A": 0.1, "B": 1.4, "C": 0.0, "D": -0.2},
                baseline_is_correct=True,
                parsed_choice="A",
            )
        ]
        out = build_decision_metrics_frame(rows, correct_key_by_prompt_uid={"u2": "A"})
        self.assertFalse(out.empty)
        self.assertTrue(bool((out["is_correct"] == False).all()))  # noqa: E712


if __name__ == "__main__":
    unittest.main()
