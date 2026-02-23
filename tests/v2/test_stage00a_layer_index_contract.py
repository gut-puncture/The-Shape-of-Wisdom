import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sow.v2.baseline_inference import resume_key_for, validate_baseline_rows  # noqa: E402


class TestStage00aLayerIndexContract(unittest.TestCase):
    def _base_row(self):
        return {
            "run_id": "r",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "model_revision": "rev",
            "prompt_uid": "u0",
            "example_id": "e0",
            "wrapper_id": "plain_exam",
            "coarse_domain": "biology",
            "resume_key": resume_key_for(model_id="Qwen/Qwen2.5-7B-Instruct", prompt_uid="u0"),
            "generated_text": "A",
            "first_generated_token_text": "A",
            "parsed_choice": "A",
            "parser_status": "resolved",
            "parser_signals": {"decision": "resolved_letter_first_token"},
            "is_correct": True,
        }

    def _layer(self, layer_index: int):
        return {
            "layer_index": int(layer_index),
            "candidate_logits": {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0},
            "candidate_probs": {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1},
            "candidate_entropy": 0.8,
            "top_candidate": "A",
            "top2_margin_prob": 0.6,
            "projected_hidden_128": [0.0] * 128,
        }

    def test_validator_fails_when_layer_indices_are_non_contiguous(self) -> None:
        row = self._base_row()
        row["layerwise"] = [self._layer(0), self._layer(2)]
        report = validate_baseline_rows(
            rows=[row],
            expected_model_id="Qwen/Qwen2.5-7B-Instruct",
            expected_model_revision="rev",
        )
        self.assertFalse(bool(report.get("pass")), msg=report)
        self.assertIn("invalid_layer_index_sequence", set(report.get("errors") or []))

    def test_validator_fails_when_layer_indices_repeat(self) -> None:
        row = self._base_row()
        row["layerwise"] = [self._layer(0), self._layer(0), self._layer(1)]
        report = validate_baseline_rows(
            rows=[row],
            expected_model_id="Qwen/Qwen2.5-7B-Instruct",
            expected_model_revision="rev",
        )
        self.assertFalse(bool(report.get("pass")), msg=report)
        self.assertIn("invalid_layer_index_sequence", set(report.get("errors") or []))


if __name__ == "__main__":
    unittest.main()
