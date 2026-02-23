import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestOrchestratorIncompleteReport(unittest.TestCase):
    def test_smoke_marks_incomplete_when_heavy_stages_skipped(self) -> None:
        run_id = "test_v2_orch_incomplete"
        run_root = REPO_ROOT / "runs" / run_id
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            (run_root / "outputs" / "Qwen__Qwen2.5-7B-Instruct").mkdir(parents=True, exist_ok=True)
            (run_root / "manifests").mkdir(parents=True, exist_ok=True)

            baseline_row = {
                "run_id": run_id,
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "model_revision": "r0",
                "prompt_uid": "u0",
                "example_id": "e0",
                "wrapper_id": "plain_exam",
                "coarse_domain": "biology",
                "is_correct": True,
                "layerwise": [
                    {
                        "layer_index": 0,
                        "candidate_logits": {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0},
                        "candidate_probs": {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1},
                        "candidate_entropy": 0.8,
                        "top_candidate": "A",
                        "top2_margin_prob": 0.6,
                        "projected_hidden_128": [0.0] * 128,
                    },
                    {
                        "layer_index": 1,
                        "candidate_logits": {"A": 1.2, "B": 0.0, "C": 0.0, "D": 0.0},
                        "candidate_probs": {"A": 0.8, "B": 0.07, "C": 0.07, "D": 0.06},
                        "candidate_entropy": 0.7,
                        "top_candidate": "A",
                        "top2_margin_prob": 0.7,
                        "projected_hidden_128": [0.1] * 128,
                    },
                ],
            }
            with (run_root / "outputs" / "Qwen__Qwen2.5-7B-Instruct" / "baseline_outputs.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps(baseline_row, sort_keys=True) + "\n")

            manifest_row = {
                "prompt_uid": "u0",
                "example_id": "e0",
                "coarse_domain": "biology",
                "correct_key": "A",
                "prompt_text": "Read.\nQuestion: 2+2?\nA) 4\nB) 3\nC) 5\nD) 6\nAnswer: ",
            }
            with (run_root / "manifests" / "ccc_baseline.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps(manifest_row, sort_keys=True) + "\n")

            cfg_path = run_root / "cfg.yaml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  - name: qwen2.5-7b-instruct",
                        "    model_id: Qwen/Qwen2.5-7B-Instruct",
                        "    revision: r0",
                        "runtime_estimator:",
                        "  threshold_hours_all_models: 36",
                        "  stage_row_multiplier:",
                        "    05_span_counterfactuals.py: 8.0",
                        "    07_run_tracing.py: 1.0",
                        "sampling:",
                        "  tracing_prompts_per_model: 1",
                        "validators:",
                        "  stage03_trajectory:",
                        "    required_types: [stable_correct]",
                        "    min_count_per_type_per_model: 1",
                        "    tail_len: 8",
                        "    max_late_flip_count: 0",
                        "    min_abs_delta_tail_floor: 0.0",
                        "  stage05_paraphrase:",
                        "    min_label_agreement: 0.0",
                        "    min_span_jaccard: 0.0",
                        "    sample_size_per_model: 1",
                        "  stage06_tracing_subset:",
                        "    difficult_domain_top_k: 1",
                        "    min_difficult_share: 0.0",
                        "    min_domains_covered: 1",
                        "    min_prompts_per_domain: 0",
                        "    max_domain_share: 1.0",
                        "  stage10:",
                        "    min_evidence_rows: 0",
                        "    min_distractor_rows: 0",
                        "    alpha: 1.0",
                        "    min_gap_ci_lo: -1.0",
                        "    min_observed_minus_shuffled: -1.0",
                        "    min_observed_minus_sign_flipped: -1.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "v2" / "00_run_experiment.py"),
                    "--run-id",
                    run_id,
                    "--mode",
                    "smoke",
                    "--model-name",
                    "qwen2.5-7b-instruct",
                    "--config",
                    str(cfg_path),
                ],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 0)

            report_path = run_root / "v2" / "00_run_experiment.report.json"
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(bool(report.get("pass")))
            self.assertTrue(any(x.get("script") == "07_run_tracing.py" for x in report.get("skipped_scripts") or []))
            self.assertTrue(any(x.get("script") == "11_generate_paper_assets.py" for x in report.get("skipped_scripts") or []))
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)


if __name__ == "__main__":
    unittest.main()
