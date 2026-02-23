import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestTracingSubsetDomainBalance(unittest.TestCase):
    def test_subset_report_includes_difficult_domain_balance(self) -> None:
        run_id = "test_v2_subset_domain_balance"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            (run_root / "manifests").mkdir(parents=True, exist_ok=True)

            traj_cycle = ["stable_correct", "stable_wrong", "unstable_correct", "unstable_wrong"]
            types_rows = []
            for i in range(40):
                traj = traj_cycle[i % len(traj_cycle)]
                is_correct = traj in {"stable_correct", "unstable_correct"}
                types_rows.append(
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_uid": f"u{i}",
                        "trajectory_type": traj,
                        "is_correct": is_correct,
                    }
                )
            pd.DataFrame.from_records(types_rows).to_parquet(out_root / "prompt_types.parquet", index=False)

            with (run_root / "manifests" / "ccc_baseline.jsonl").open("w", encoding="utf-8") as f:
                for i in range(40):
                    domain = "hard_domain" if i < 20 else "easy_domain"
                    row = {
                        "prompt_uid": f"u{i}",
                        "example_id": f"e{i}",
                        "coarse_domain": domain,
                        "correct_key": "A",
                        "prompt_text": "Q?\nA) a\nB) b\nC) c\nD) d\nAnswer:",
                    }
                    f.write(json.dumps(row, sort_keys=True) + "\n")

            cfg_path = run_root / "cfg.yaml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  - name: qwen2.5-7b-instruct",
                        "    model_id: Qwen/Qwen2.5-7B-Instruct",
                        "    revision: r0",
                        "sampling:",
                        "  tracing_prompts_per_model: 20",
                        "validators:",
                        "  stage06_tracing_subset:",
                        "    difficult_domain_top_k: 1",
                        "    min_difficult_share: 0.4",
                        "    min_domains_covered: 1",
                        "    min_prompts_per_domain: 0",
                        "    max_domain_share: 1.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "v2" / "06_select_tracing_subset.py"),
                    "--run-id",
                    run_id,
                    "--model-name",
                    "qwen2.5-7b-instruct",
                    "--config",
                    str(cfg_path),
                ],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 0)

            report = json.loads((out_root / "06_select_tracing_subset.report.json").read_text(encoding="utf-8"))
            model_rep = report["models"]["Qwen/Qwen2.5-7B-Instruct"]
            self.assertIn("domain_difficulty", model_rep)
            self.assertIn("difficult_domain_balance", model_rep)
            bal = model_rep["difficult_domain_balance"]
            self.assertGreaterEqual(float(bal["actual_difficult_share"]), float(bal["min_difficult_share"]))
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_subset_fails_when_min_domains_covered_unmet(self) -> None:
        run_id = "test_v2_subset_domain_coverage_fail"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            (run_root / "manifests").mkdir(parents=True, exist_ok=True)

            traj_cycle = ["stable_correct", "stable_wrong", "unstable_correct", "unstable_wrong"]
            types_rows = []
            for i in range(20):
                traj = traj_cycle[i % len(traj_cycle)]
                types_rows.append(
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_uid": f"u{i}",
                        "trajectory_type": traj,
                        "is_correct": traj in {"stable_correct", "unstable_correct"},
                    }
                )
            pd.DataFrame.from_records(types_rows).to_parquet(out_root / "prompt_types.parquet", index=False)

            with (run_root / "manifests" / "ccc_baseline.jsonl").open("w", encoding="utf-8") as f:
                for i in range(20):
                    row = {
                        "prompt_uid": f"u{i}",
                        "example_id": f"e{i}",
                        "coarse_domain": "one_domain_only",
                        "correct_key": "A",
                        "prompt_text": "Q?\nA) a\nB) b\nC) c\nD) d\nAnswer:",
                    }
                    f.write(json.dumps(row, sort_keys=True) + "\n")

            cfg_path = run_root / "cfg.yaml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  - name: qwen2.5-7b-instruct",
                        "    model_id: Qwen/Qwen2.5-7B-Instruct",
                        "    revision: r0",
                        "sampling:",
                        "  tracing_prompts_per_model: 20",
                        "validators:",
                        "  stage06_tracing_subset:",
                        "    difficult_domain_top_k: 1",
                        "    min_difficult_share: 0.3",
                        "    min_domains_covered: 2",
                        "    min_prompts_per_domain: 5",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "v2" / "06_select_tracing_subset.py"),
                    "--run-id",
                    run_id,
                    "--model-name",
                    "qwen2.5-7b-instruct",
                    "--config",
                    str(cfg_path),
                ],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 2)

            report = json.loads((out_root / "06_select_tracing_subset.report.json").read_text(encoding="utf-8"))
            model_rep = report["models"]["Qwen/Qwen2.5-7B-Instruct"]
            self.assertFalse(bool(model_rep.get("pass")))
            self.assertIn("domain_coverage", model_rep.get("failing_gates") or [])
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)

    def test_subset_fails_when_domain_share_cap_is_exceeded(self) -> None:
        run_id = "test_v2_subset_domain_share_fail"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        if run_root.exists():
            shutil.rmtree(run_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            (run_root / "manifests").mkdir(parents=True, exist_ok=True)

            traj_cycle = ["stable_correct", "stable_wrong", "unstable_correct", "unstable_wrong"]
            types_rows = []
            for i in range(40):
                traj = traj_cycle[i % len(traj_cycle)]
                types_rows.append(
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt_uid": f"u{i}",
                        "trajectory_type": traj,
                        "is_correct": traj in {"stable_correct", "unstable_correct"},
                    }
                )
            pd.DataFrame.from_records(types_rows).to_parquet(out_root / "prompt_types.parquet", index=False)

            with (run_root / "manifests" / "ccc_baseline.jsonl").open("w", encoding="utf-8") as f:
                for i in range(40):
                    domain = "domain_heavy" if i < 34 else "domain_light"
                    row = {
                        "prompt_uid": f"u{i}",
                        "example_id": f"e{i}",
                        "coarse_domain": domain,
                        "correct_key": "A",
                        "prompt_text": "Q?\nA) a\nB) b\nC) c\nD) d\nAnswer:",
                    }
                    f.write(json.dumps(row, sort_keys=True) + "\n")

            cfg_path = run_root / "cfg.yaml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  - name: qwen2.5-7b-instruct",
                        "    model_id: Qwen/Qwen2.5-7B-Instruct",
                        "    revision: r0",
                        "sampling:",
                        "  tracing_prompts_per_model: 20",
                        "validators:",
                        "  stage06_tracing_subset:",
                        "    difficult_domain_top_k: 1",
                        "    min_difficult_share: 0.0",
                        "    min_domains_covered: 1",
                        "    min_prompts_per_domain: 0",
                        "    max_domain_share: 0.6",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "v2" / "06_select_tracing_subset.py"),
                    "--run-id",
                    run_id,
                    "--model-name",
                    "qwen2.5-7b-instruct",
                    "--config",
                    str(cfg_path),
                ],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 2)

            report = json.loads((out_root / "06_select_tracing_subset.report.json").read_text(encoding="utf-8"))
            model_rep = report["models"]["Qwen/Qwen2.5-7B-Instruct"]
            self.assertFalse(bool(model_rep.get("pass")))
            self.assertIn("domain_max_share", model_rep.get("failing_gates") or [])
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)


if __name__ == "__main__":
    unittest.main()
