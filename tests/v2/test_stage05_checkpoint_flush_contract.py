import importlib.util
import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT05 = REPO_ROOT / "scripts" / "v2" / "05_span_counterfactuals.py"
CFG_PATH = REPO_ROOT / "configs" / "experiment_v2.yaml"


def _load_script05():
    if str(SCRIPT05.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT05.parent))
    spec = importlib.util.spec_from_file_location("s05_checkpoint_contract", SCRIPT05)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {SCRIPT05}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestStage05CheckpointFlushContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod05 = _load_script05()

    def test_checkpoint_flush_happens_at_configured_prompt_interval(self) -> None:
        run_id = "test_v2_stage05_checkpoint_flush"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        cfg_path = run_root / "cfg.yaml"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
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
                    },
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_uid": "u1",
                        "layer_index": 0,
                        "delta": 0.5,
                        "competitor": "B",
                        "correct_key": "A",
                    },
                ]
            ).to_parquet(out_root / "decision_metrics.parquet", index=False)
            with (run_root / "manifests" / "ccc_baseline.jsonl").open("w", encoding="utf-8") as f:
                for uid in ["u0", "u1"]:
                    row = {
                        "prompt_uid": uid,
                        "example_id": f"e_{uid}",
                        "coarse_domain": "biology",
                        "correct_key": "A",
                        "prompt_text": "Read.\nQuestion: 2+2?\nA) 4\nB) 3\nC) 5\nD) 6\nAnswer: ",
                    }
                    f.write(json.dumps(row, sort_keys=True) + "\n")
            cfg_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  - name: qwen2.5-7b-instruct",
                        "    model_id: Qwen/Qwen2.5-7B-Instruct",
                        "    revision: r0",
                        "execution:",
                        "  stage05_checkpoint_every_prompts: 1",
                        "  deterministic_seed: 12345",
                        "sampling:",
                        "  span_counterfactual_prompts_per_model: 2",
                        "  span_counterfactual_max_prompts_per_model: 2",
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
                encoding="utf-8",
            )

            merge_calls = 0
            original_merge = self.mod05._merge_span_rows

            def _counting_merge(*args, **kwargs):
                nonlocal merge_calls
                merge_calls += 1
                return original_merge(*args, **kwargs)

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
            with patch.object(self.mod05, "_merge_span_rows", side_effect=_counting_merge):
                try:
                    rc = self.mod05.main()
                finally:
                    sys.argv = old_argv

            self.assertEqual(rc, 0)
            self.assertGreaterEqual(
                merge_calls,
                2,
                msg="checkpoint interval=1 with two prompts should flush at least once per processed prompt",
            )
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_stage05_default_checkpoint_interval_is_strict(self) -> None:
        cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
        execution_cfg = cfg.get("execution") or {}
        interval = int(execution_cfg.get("stage05_checkpoint_every_prompts", 0))
        self.assertLessEqual(
            interval,
            10,
            msg="stage05 checkpoint interval default must be strict to minimize crash-loss window",
        )


if __name__ == "__main__":
    unittest.main()
