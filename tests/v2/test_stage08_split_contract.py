import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_tracing_scalars(out_root: Path, *, n_prompts: int) -> None:
    rows = []
    for pi in range(int(n_prompts)):
        prompt_uid = f"u{pi}"
        for li in range(4):
            s_attn = 0.10 + 0.01 * li
            s_mlp = 0.05 + 0.01 * li
            drift = s_attn + s_mlp
            rows.append(
                {
                    "model_id": "Qwen/Qwen2.5-7B-Instruct",
                    "prompt_uid": prompt_uid,
                    "layer_index": li,
                    "drift": drift,
                    "s_attn": s_attn,
                    "s_mlp": s_mlp,
                }
            )
    pd.DataFrame.from_records(rows).to_parquet(out_root / "tracing_scalars.parquet", index=False)


class TestStage08SplitContract(unittest.TestCase):
    def test_stage08_report_includes_split_contract_fields(self) -> None:
        run_id = "test_v2_stage08_split_contract"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        cfg_path = run_root / "cfg.yaml"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            _write_tracing_scalars(out_root, n_prompts=6)
            cfg_path.write_text(
                "\n".join(
                    [
                        "causal:",
                        "  drift_decomposition_r2_min: 0.50",
                        "validators:",
                        "  stage08_decomposition:",
                        "    split_train_fraction: 0.5",
                        "    min_train_rows: 3",
                        "    min_test_rows: 3",
                        "    require_split_r2: true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "v2" / "08_attention_and_mlp_decomposition.py"),
                    "--run-id",
                    run_id,
                    "--config",
                    str(cfg_path),
                ],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 0)
            report = json.loads((out_root / "08_attention_and_mlp_decomposition.report.json").read_text(encoding="utf-8"))
            self.assertIn("split_contract", report)
            self.assertIn("gates", report)
            self.assertIn("failing_gates", report)
            self.assertTrue(bool(report.get("pass")))
            self.assertTrue(bool(report.get("gates", {}).get("split_train_rows_min")))
            self.assertTrue(bool(report.get("gates", {}).get("split_test_rows_min")))
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_stage08_fails_when_split_min_rows_unmet(self) -> None:
        run_id = "test_v2_stage08_split_rows_fail"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        cfg_path = run_root / "cfg.yaml"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            _write_tracing_scalars(out_root, n_prompts=2)
            cfg_path.write_text(
                "\n".join(
                    [
                        "causal:",
                        "  drift_decomposition_r2_min: 0.0",
                        "validators:",
                        "  stage08_decomposition:",
                        "    split_train_fraction: 0.8",
                        "    min_train_rows: 100",
                        "    min_test_rows: 100",
                        "    require_split_r2: false",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "v2" / "08_attention_and_mlp_decomposition.py"),
                    "--run-id",
                    run_id,
                    "--config",
                    str(cfg_path),
                ],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 2)
            report = json.loads((out_root / "08_attention_and_mlp_decomposition.report.json").read_text(encoding="utf-8"))
            self.assertFalse(bool(report.get("pass")))
            self.assertIn("split_train_rows_min", report.get("failing_gates") or [])
            self.assertIn("split_test_rows_min", report.get("failing_gates") or [])
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)


if __name__ == "__main__":
    unittest.main()
