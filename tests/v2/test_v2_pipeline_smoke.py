import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestV2PipelineSmoke(unittest.TestCase):
    def test_v2_scripts_smoke(self) -> None:
        run_id = "test_v2_smoke"
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

            scripts_and_args = [
                ("01_extract_baseline.py", []),
                ("02_compute_decision_metrics.py", []),
                ("03_classify_trajectories.py", []),
                ("04_region_analysis.py", []),
                ("05_span_counterfactuals.py", ["--counterfactual-mode", "proxy"]),
                ("10_causal_validation_tools.py", []),
            ]
            for script, extra_args in scripts_and_args:
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(REPO_ROOT / "scripts" / "v2" / script),
                        "--run-id",
                        run_id,
                        "--model-name",
                        "qwen2.5-7b-instruct",
                        *extra_args,
                    ],
                    cwd=str(REPO_ROOT),
                    check=False,
                )
                self.assertEqual(proc.returncode, 0, msg=f"failed script: {script}")

            out_root = run_root / "v2"
            self.assertTrue((out_root / "decision_metrics.parquet").exists())
            self.assertTrue((out_root / "prompt_types.parquet").exists())
            self.assertTrue((out_root / "span_labels.parquet").exists())
            self.assertTrue((out_root / "span_deletion_causal.parquet").exists())
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)


if __name__ == "__main__":
    unittest.main()
