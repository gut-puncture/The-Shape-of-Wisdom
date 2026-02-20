import importlib.util
import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT05 = REPO_ROOT / "scripts" / "v2" / "05_span_counterfactuals.py"


def _load_script05():
    if str(SCRIPT05.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT05.parent))
    spec = importlib.util.spec_from_file_location("s05_paraphrase_gate", SCRIPT05)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {SCRIPT05}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSpanParaphraseStabilityGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod05 = _load_script05()

    def _prepare_minimal_run(self, run_id: str, cfg_text: str) -> Path:
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        if run_root.exists():
            shutil.rmtree(run_root)
        out_root.mkdir(parents=True, exist_ok=True)
        (run_root / "manifests").mkdir(parents=True, exist_ok=True)

        pd.DataFrame.from_records(
            [
                {
                    "model_id": "Qwen/Qwen2.5-7B-Instruct",
                    "prompt_uid": "u0",
                    "layer_index": 0,
                    "delta": 0.4,
                    "competitor": "B",
                    "correct_key": "A",
                }
            ]
        ).to_parquet(out_root / "decision_metrics.parquet", index=False)

        manifest_row = {
            "prompt_uid": "u0",
            "example_id": "e0",
            "coarse_domain": "biology",
            "correct_key": "A",
            "prompt_text": "Read.\nQuestion: 2+2?\nA) 4\nB) 3\nC) 5\nD) 6\nAnswer: ",
        }
        (run_root / "manifests" / "ccc_baseline.jsonl").write_text(
            json.dumps(manifest_row, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        cfg_path = run_root / "cfg.yaml"
        cfg_path.write_text(cfg_text, encoding="utf-8")
        return cfg_path

    def test_stage05_fails_when_paraphrase_gate_is_violated(self) -> None:
        run_id = "test_v2_span_paraphrase_fail"
        cfg_path = self._prepare_minimal_run(
            run_id,
            "\n".join(
                [
                    "models:",
                    "  - name: qwen2.5-7b-instruct",
                    "    model_id: Qwen/Qwen2.5-7B-Instruct",
                    "    revision: r0",
                    "span_counterfactual:",
                    "  mode: proxy",
                    "validators:",
                    "  stage05_paraphrase:",
                    "    min_label_agreement: 0.95",
                    "    min_span_jaccard: 0.95",
                    "    sample_size_per_model: 1",
                ]
            )
            + "\n",
        )
        run_root = REPO_ROOT / "runs" / run_id

        with patch.object(
            self.mod05,
            "_deterministic_paraphrase",
            return_value="Completely different free-form text without options",
        ):
            argv = [
                "prog",
                "--run-id",
                run_id,
                "--model-name",
                "qwen2.5-7b-instruct",
                "--config",
                str(cfg_path),
                "--counterfactual-mode",
                "proxy",
            ]
            old_argv = sys.argv
            sys.argv = argv
            try:
                rc = self.mod05.main()
            finally:
                sys.argv = old_argv

        try:
            self.assertEqual(rc, 2)
            report = json.loads((run_root / "v2" / "05_span_counterfactuals.report.json").read_text(encoding="utf-8"))
            self.assertFalse(bool(report.get("pass")))
            self.assertTrue((report.get("failing_gates") or []))
            self.assertIn("paraphrase_span_jaccard", report.get("failing_gates") or [])
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_stage05_writes_paraphrase_report_fields(self) -> None:
        run_id = "test_v2_span_paraphrase_fields"
        cfg_path = self._prepare_minimal_run(
            run_id,
            "\n".join(
                [
                    "models:",
                    "  - name: qwen2.5-7b-instruct",
                    "    model_id: Qwen/Qwen2.5-7B-Instruct",
                    "    revision: r0",
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
        run_root = REPO_ROOT / "runs" / run_id
        argv = [
            "prog",
            "--run-id",
            run_id,
            "--model-name",
            "qwen2.5-7b-instruct",
            "--config",
            str(cfg_path),
            "--counterfactual-mode",
            "proxy",
        ]
        old_argv = sys.argv
        sys.argv = argv
        try:
            rc = self.mod05.main()
        finally:
            sys.argv = old_argv

        try:
            self.assertEqual(rc, 0)
            report = json.loads((run_root / "v2" / "05_span_counterfactuals.report.json").read_text(encoding="utf-8"))
            self.assertIn("paraphrase_stability", report)
            self.assertIn("gates", report)
            self.assertIn("failing_gates", report)
            self.assertTrue((run_root / "v2" / "span_paraphrase_stability.parquet").exists())
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_stage05_fails_when_expected_model_missing(self) -> None:
        run_id = "test_v2_span_paraphrase_missing_model"
        cfg_path = self._prepare_minimal_run(
            run_id,
            "\n".join(
                [
                    "models:",
                    "  - name: qwen2.5-7b-instruct",
                    "    model_id: Qwen/Qwen2.5-7B-Instruct",
                    "    revision: r0",
                    "  - name: llama-3.1-8b-instruct",
                    "    model_id: meta-llama/Llama-3.1-8B-Instruct",
                    "    revision: r1",
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
        run_root = REPO_ROOT / "runs" / run_id
        argv = [
            "prog",
            "--run-id",
            run_id,
            "--config",
            str(cfg_path),
            "--counterfactual-mode",
            "proxy",
        ]
        old_argv = sys.argv
        sys.argv = argv
        try:
            rc = self.mod05.main()
        finally:
            sys.argv = old_argv

        try:
            self.assertEqual(rc, 2)
            report = json.loads((run_root / "v2" / "05_span_counterfactuals.report.json").read_text(encoding="utf-8"))
            self.assertFalse(bool(report.get("pass")))
            self.assertIn("paraphrase_expected_models_present", report.get("failing_gates") or [])
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)


if __name__ == "__main__":
    unittest.main()
