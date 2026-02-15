import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


class TestStage14Analysis(unittest.TestCase):
    def test_analysis_writes_artifacts_and_validates(self) -> None:
        from sow.analysis.stage14 import run_stage14_analysis
        from sow.hashing import sha256_file

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            (run_dir / "sentinels").mkdir(parents=True, exist_ok=True)
            (run_dir / "manifests").mkdir(parents=True, exist_ok=True)

            cfg = {
                "models": [{"name": "fake", "model_id": "fake/model", "revision": "r0", "dtype": "float16", "device": "cpu"}],
            }

            # Small CCC manifests.
            base_manifest = run_dir / "manifests" / "ccc_baseline.jsonl"
            rob_manifest = run_dir / "manifests" / "ccc_robustness.jsonl"
            _write_jsonl(base_manifest, [{"prompt_uid": "u0"}, {"prompt_uid": "u1"}])
            _write_jsonl(rob_manifest, [{"prompt_uid": "r0"}, {"prompt_uid": "r1"}, {"prompt_uid": "r2"}, {"prompt_uid": "r3"}])

            # Matching inference outputs.
            out_base = run_dir / "outputs" / "fake__model" / "baseline_outputs.jsonl"
            out_rob = run_dir / "outputs" / "fake__model" / "robustness_outputs.jsonl"

            def mk_row(uid: str, wrapper: str):
                return {
                    "run_id": "test",
                    "model_id": "fake/model",
                    "model_revision": "r0",
                    "prompt_uid": uid,
                    "example_id": "e0",
                    "wrapper_id": wrapper,
                    "coarse_domain": "d0",
                    "first_token_is_option_letter": True,
                    "parser_status": "resolved",
                    "parsed_choice": "A",
                    "is_correct": True,
                    "flip_count": 0,
                    "commitment_layer_by_margin_threshold": {"0.1": 0},
                    "layerwise": [
                        {
                            "layer_index": 0,
                            "candidate_entropy": 0.1,
                            "top2_margin_prob": 0.9,
                            "candidate_logits": {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0},
                            "candidate_probs": {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0},
                            "top_candidate": "A",
                            "projected_hidden_128": [0.0] * 128,
                        }
                    ],
                }

            _write_jsonl(out_base, [mk_row("u0", "plain_exam"), mk_row("u1", "plain_exam")])
            _write_jsonl(out_rob, [mk_row("r0", "w0"), mk_row("r1", "w0"), mk_row("r2", "w1"), mk_row("r3", "w1")])

            # Sentinels with sha256 checks.
            (run_dir / "sentinels" / "inference_baseline.fake__model.done").write_text(
                json.dumps({"output_path": str(out_base), "output_sha256": sha256_file(out_base)}), encoding="utf-8"
            )
            (run_dir / "sentinels" / "inference_robustness.fake__model.done").write_text(
                json.dumps({"output_path": str(out_rob), "output_sha256": sha256_file(out_rob)}), encoding="utf-8"
            )

            rep = run_stage14_analysis(
                run_id="test",
                run_dir=run_dir,
                cfg=cfg,
                baseline_manifest=base_manifest,
                robustness_manifest=rob_manifest,
                baseline_wrapper_id="plain_exam",
                thresholds=[0.1],
            )
            self.assertTrue(rep["pass"])
            for k in ["per_prompt_metrics_csv", "layerwise_aggregates_csv", "robustness_deltas_csv", "commitment_hist_csv"]:
                p = Path(rep["artifacts"][k])
                self.assertTrue(p.exists())


if __name__ == "__main__":
    unittest.main()

