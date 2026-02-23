import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def _prepare_run(run_id: str, *, prompt_count: int, cfg_text: str) -> tuple[Path, Path]:
    run_root = REPO_ROOT / "runs" / run_id
    out_root = run_root / "v2"
    cfg_path = run_root / "cfg.yaml"
    if run_root.exists():
        shutil.rmtree(run_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (run_root / "manifests").mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(prompt_count):
        rows.append(
            {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "prompt_uid": f"u{i}",
                "layer_index": 0,
                "delta": 0.4,
                "competitor": "B",
                "correct_key": "A",
            }
        )
    pd.DataFrame.from_records(rows).to_parquet(out_root / "decision_metrics.parquet", index=False)

    with (run_root / "manifests" / "ccc_baseline.jsonl").open("w", encoding="utf-8") as f:
        for i in range(prompt_count):
            row = {
                "prompt_uid": f"u{i}",
                "example_id": f"e{i}",
                "coarse_domain": "biology",
                "correct_key": "A",
                "prompt_text": "Read.\nQuestion: 2+2?\nA) 4\nB) 3\nC) 5\nD) 6\nAnswer: ",
            }
            f.write(json.dumps(row, sort_keys=True) + "\n")

    cfg_path.write_text(cfg_text, encoding="utf-8")
    return run_root, cfg_path


class TestStage05SamplingContract(unittest.TestCase):
    def test_stage05_fails_when_full_mode_sampling_target_unmet(self) -> None:
        run_id = "test_v2_stage05_sampling_unmet"
        run_root, cfg_path = _prepare_run(
            run_id,
            prompt_count=3,
            cfg_text="\n".join(
                [
                    "models:",
                    "  - name: qwen2.5-7b-instruct",
                    "    model_id: Qwen/Qwen2.5-7B-Instruct",
                    "    revision: r0",
                    "sampling:",
                    "  span_counterfactual_prompts_per_model: 5",
                    "  span_counterfactual_max_prompts_per_model: 10",
                    "span_counterfactual:",
                    "  mode: proxy",
                    "validators:",
                    "  stage05_paraphrase:",
                    "    min_label_agreement: 0.0",
                    "    min_span_jaccard: 0.0",
                    "    sample_size_per_model: 1",
                ]
            )
            + "\n",
        )
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "v2" / "05_span_counterfactuals.py"),
                    "--run-id",
                    run_id,
                    "--model-name",
                    "qwen2.5-7b-instruct",
                    "--config",
                    str(cfg_path),
                    "--counterfactual-mode",
                    "proxy",
                ],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 2)
            report = json.loads((run_root / "v2" / "05_span_counterfactuals.report.json").read_text(encoding="utf-8"))
            self.assertIn("sampling_contract", report)
            self.assertIn("sampling_full_mode_prompt_count", report.get("failing_gates") or [])
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_stage05_reports_selected_prompt_counts_per_model(self) -> None:
        run_id = "test_v2_stage05_sampling_report"
        run_root, cfg_path = _prepare_run(
            run_id,
            prompt_count=4,
            cfg_text="\n".join(
                [
                    "models:",
                    "  - name: qwen2.5-7b-instruct",
                    "    model_id: Qwen/Qwen2.5-7B-Instruct",
                    "    revision: r0",
                    "sampling:",
                    "  span_counterfactual_prompts_per_model: 2",
                    "  span_counterfactual_max_prompts_per_model: 3",
                    "execution:",
                    "  deterministic_seed: 12345",
                    "span_counterfactual:",
                    "  mode: proxy",
                    "validators:",
                    "  stage05_paraphrase:",
                    "    min_label_agreement: 0.0",
                    "    min_span_jaccard: 0.0",
                    "    sample_size_per_model: 1",
                ]
            )
            + "\n",
        )
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "v2" / "05_span_counterfactuals.py"),
                    "--run-id",
                    run_id,
                    "--model-name",
                    "qwen2.5-7b-instruct",
                    "--config",
                    str(cfg_path),
                    "--counterfactual-mode",
                    "proxy",
                ],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 0)
            report = json.loads((run_root / "v2" / "05_span_counterfactuals.report.json").read_text(encoding="utf-8"))
            contract = report.get("sampling_contract") or {}
            self.assertIn("selected_prompt_counts", contract)
            self.assertEqual(int(contract["selected_prompt_counts"]["Qwen/Qwen2.5-7B-Instruct"]), 2)
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)


if __name__ == "__main__":
    unittest.main()
