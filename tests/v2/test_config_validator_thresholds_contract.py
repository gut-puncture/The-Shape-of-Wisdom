import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = REPO_ROOT / "configs" / "experiment_v2.yaml"


class TestConfigValidatorThresholdsContract(unittest.TestCase):
    def test_explicit_thresholds_present_for_gated_stages(self) -> None:
        cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
        validators = cfg.get("validators") or {}
        data_scope = cfg.get("data_scope") or {}
        execution = cfg.get("execution") or {}
        runtime_estimator = cfg.get("runtime_estimator") or {}

        self.assertIn("baseline_manifest_source", data_scope)
        self.assertIn("baseline_manifest_sha256", data_scope)
        self.assertIn("baseline_manifest_expected_rows_full", data_scope)

        stage05 = validators.get("stage05_paraphrase") or {}
        self.assertIn("min_label_agreement", stage05)
        self.assertIn("min_span_jaccard", stage05)
        self.assertIn("sample_size_per_model", stage05)

        stage03 = validators.get("stage03_trajectory") or {}
        self.assertIn("required_types", stage03)
        self.assertIn("min_count_per_type_per_model", stage03)
        self.assertIn("tail_len", stage03)
        self.assertIn("max_late_flip_count", stage03)
        self.assertIn("min_abs_delta_tail_floor", stage03)

        stage06 = validators.get("stage06_tracing_subset") or {}
        self.assertIn("difficult_domain_top_k", stage06)
        self.assertIn("min_difficult_share", stage06)
        self.assertIn("min_domains_covered", stage06)
        self.assertIn("min_prompts_per_domain", stage06)
        self.assertIn("max_domain_share", stage06)

        stage09 = validators.get("stage09") or {}
        self.assertIn("min_ablation_rows", stage09)
        self.assertIn("min_patching_rows", stage09)

        stage10 = validators.get("stage10") or {}
        self.assertIn("min_evidence_rows", stage10)
        self.assertIn("min_distractor_rows", stage10)
        self.assertIn("alpha", stage10)
        self.assertIn("min_gap_ci_lo", stage10)
        self.assertIn("min_observed_minus_shuffled", stage10)
        self.assertIn("min_observed_minus_sign_flipped", stage10)
        self.assertIn("split_train_fraction", stage10)
        self.assertIn("min_train_rows", stage10)
        self.assertIn("min_test_rows", stage10)
        self.assertIn("require_split_direction_match", stage10)

        stage08 = validators.get("stage08_decomposition") or {}
        self.assertIn("split_train_fraction", stage08)
        self.assertIn("min_train_rows", stage08)
        self.assertIn("min_test_rows", stage08)
        self.assertIn("require_split_r2", stage08)

        self.assertIn("stage00_baseline_checkpoint_every_prompts", execution)
        self.assertIn("stage00_baseline_batch_chain_mps", execution)
        self.assertIn("stage00_baseline_batch_chain_cuda", execution)
        self.assertIn("require_measured_rps_for_full", runtime_estimator)

    def test_preregistered_hypotheses_doc_exists(self) -> None:
        prereg = REPO_ROOT / "docs" / "PREREGISTERED_HYPOTHESES_V3.md"
        self.assertTrue(prereg.exists(), msg=f"missing preregistration doc: {prereg}")


if __name__ == "__main__":
    unittest.main()
