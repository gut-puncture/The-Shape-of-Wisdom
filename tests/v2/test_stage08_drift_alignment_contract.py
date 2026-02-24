import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_shifted_tracing_scalars(out_root: Path, *, n_prompts: int = 12, n_layers: int = 6) -> None:
    rows = []
    for pi in range(int(n_prompts)):
        prompt_uid = f"u{pi}"
        rng = np.random.default_rng(1000 + int(pi))
        s_attn = rng.normal(loc=0.0, scale=0.2, size=int(n_layers))
        s_mlp = rng.normal(loc=0.0, scale=0.2, size=int(n_layers))
        signal = (0.9 * s_attn) + (0.4 * s_mlp)
        delta = np.cumsum(signal)
        drift_next = np.zeros_like(delta)
        if int(n_layers) > 1:
            drift_next[:-1] = delta[1:] - delta[:-1]
        drift_next[-1] = 0.0

        for li in range(int(n_layers)):
            rows.append(
                {
                    "model_id": "Qwen/Qwen2.5-7B-Instruct",
                    "prompt_uid": prompt_uid,
                    "layer_index": int(li),
                    "delta": float(delta[li]),
                    # Intentionally store forward-diff drift to emulate stage07 raw contract.
                    "drift": float(drift_next[li]),
                    "s_attn": float(s_attn[li]),
                    "s_mlp": float(s_mlp[li]),
                }
            )

    pd.DataFrame.from_records(rows).to_parquet(out_root / "tracing_scalars.parquet", index=False)


class TestStage08DriftAlignmentContract(unittest.TestCase):
    def test_stage08_aligns_drift_to_current_layer_components(self) -> None:
        run_id = "test_v2_stage08_drift_alignment"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        cfg_path = run_root / "cfg.yaml"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            _write_shifted_tracing_scalars(out_root, n_prompts=14, n_layers=6)
            cfg_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  - name: qwen2.5-7b-instruct",
                        "    model_id: Qwen/Qwen2.5-7B-Instruct",
                        "    revision: r0",
                        "causal:",
                        "  drift_decomposition_r2_min: 0.70",
                        "validators:",
                        "  stage08_decomposition:",
                        "    split_train_fraction: 0.7",
                        "    min_train_rows: 20",
                        "    min_test_rows: 20",
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
            self.assertTrue(bool(report.get("pass")))
            gates = report.get("gates") or {}
            self.assertTrue(bool(gates.get("drift_decomposition_r2")))
            self.assertTrue(bool(gates.get("split_r2_min")))
            self.assertEqual(str(report.get("drift_target_alignment")), "delta_current_minus_delta_prev_per_prompt")
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)


if __name__ == "__main__":
    unittest.main()
