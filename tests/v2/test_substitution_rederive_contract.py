import importlib.util
import random
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sow.v2.causal.patching import run_activation_patching  # noqa: E402


def _load_substitution_module():
    script = REPO_ROOT / "scripts" / "audit" / "substitution_rederive.py"
    if str(script.parent) not in sys.path:
        sys.path.insert(0, str(script.parent))
    spec = importlib.util.spec_from_file_location("substitution_rederive_contract", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load substitution_rederive.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestSubstitutionRederiveContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parquet_dir = REPO_ROOT / "results" / "parquet"
        cls.audit_pairs_csv = REPO_ROOT / "results" / "audit" / "substitution_pairs_vnext.csv"
        if not (cls.parquet_dir / "tracing_scalars.parquet").exists():
            raise unittest.SkipTest("cached parquet artifacts unavailable")
        cls.tracing = pd.read_parquet(cls.parquet_dir / "tracing_scalars.parquet")
        cls.prompt_types = pd.read_parquet(cls.parquet_dir / "prompt_types.parquet")
        cls.patching = pd.read_parquet(cls.parquet_dir / "patching_results.parquet")
        cls.mod = _load_substitution_module()

    def test_legacy_patching_reproduces_cached_patching_results(self) -> None:
        merged = self.tracing.merge(
            self.prompt_types[["model_id", "prompt_uid", "trajectory_type"]],
            on=["model_id", "prompt_uid"],
            how="left",
        )
        failing = merged[merged["trajectory_type"].isin(["unstable_wrong", "stable_wrong"])]
        success = merged[merged["trajectory_type"].isin(["stable_correct"])]
        target_layers = list(range(20, 28))
        attn = run_activation_patching(failing, success, component="attention", target_layers=target_layers)
        mlp = run_activation_patching(failing, success, component="mlp", target_layers=target_layers)
        recomputed = pd.concat([attn, mlp], ignore_index=True)

        expected_mean = self.patching.groupby("component")["delta_shift"].mean().to_dict()
        got_mean = recomputed.groupby("component")["delta_shift"].mean().to_dict()
        expected_frac = self.patching.groupby("component").apply(lambda g: float((g["delta_shift"] > 0).mean())).to_dict()
        got_frac = recomputed.groupby("component").apply(lambda g: float((g["delta_shift"] > 0).mean())).to_dict()
        for comp in ["attention", "mlp"]:
            self.assertAlmostEqual(float(got_mean[comp]), float(expected_mean[comp]), places=9)
            self.assertAlmostEqual(float(got_frac[comp]), float(expected_frac[comp]), places=9)

    def test_legacy_pairing_is_row_order_sensitive(self) -> None:
        merged = self.tracing.merge(
            self.prompt_types[["model_id", "prompt_uid", "trajectory_type"]],
            on=["model_id", "prompt_uid"],
            how="left",
        )
        failing = merged[merged["trajectory_type"].isin(["unstable_wrong", "stable_wrong"])]
        success = merged[merged["trajectory_type"].isin(["stable_correct"])]
        target_layers = list(range(20, 28))

        base = run_activation_patching(failing, success, component="attention", target_layers=target_layers)
        success_shuffled = success.sample(frac=1.0, random_state=2026).reset_index(drop=True)
        perturbed = run_activation_patching(failing, success_shuffled, component="attention", target_layers=target_layers)
        base_mean = float(base["delta_shift"].mean())
        perturbed_mean = float(perturbed["delta_shift"].mean())
        self.assertGreater(abs(perturbed_mean - base_mean), 1e-3)

    def test_all_pairs_mode_is_pair_set_invariant_to_order(self) -> None:
        decision = pd.read_parquet(self.parquet_dir / "decision_metrics.parquet")
        traces, prompt_order = self.mod._build_prompt_traces(
            tracing=self.tracing,
            prompt_types=self.prompt_types,
            decision=decision,
            spans_jsonl=REPO_ROOT / "results" / "spans.jsonl",
        )
        mid = sorted({k[0] for k in traces.keys()})[0]
        prompt_uids = [u for u in prompt_order[mid] if (mid, u) in traces]
        pool = [traces[(mid, u)] for u in prompt_uids]
        success_pool = [p for p in pool if p.trajectory_type == "stable_correct"]
        fail_pool = [p for p in pool if p.trajectory_type in {"stable_wrong", "unstable_wrong"}]
        pairs_a = self.mod._build_pairs(
            pairing_mode="all_pairs_within_model",
            model_id=mid,
            success_pool=success_pool,
            fail_pool=fail_pool,
            prompt_order=prompt_order[mid],
            seed=12345,
        )
        success_pool_b = list(success_pool)
        fail_pool_b = list(fail_pool)
        random.Random(42).shuffle(success_pool_b)
        random.Random(43).shuffle(fail_pool_b)
        pairs_b = self.mod._build_pairs(
            pairing_mode="all_pairs_within_model",
            model_id=mid,
            success_pool=success_pool_b,
            fail_pool=fail_pool_b,
            prompt_order=list(reversed(prompt_order[mid])),
            seed=12345,
        )
        set_a = {(s.prompt_uid, t.prompt_uid) for s, t in pairs_a}
        set_b = {(s.prompt_uid, t.prompt_uid) for s, t in pairs_b}
        self.assertEqual(set_a, set_b)
        self.assertEqual(len(pairs_a), len(pairs_b))

    def test_attention_and_mlp_use_identical_pair_keys_per_setting(self) -> None:
        if not self.audit_pairs_csv.exists():
            raise unittest.SkipTest("substitution_pairs_vnext.csv missing")
        df = pd.read_csv(self.audit_pairs_csv)
        group_cols = ["pairing_mode", "normalization_mode", "layer_range_mode", "failing_set_mode", "model_id"]
        for key, g in df.groupby(group_cols, sort=False):
            a = set(g[g["component"] == "attention"]["pair_id"].astype(str).tolist())
            m = set(g[g["component"] == "mlp"]["pair_id"].astype(str).tolist())
            self.assertEqual(a, m, msg=f"pair-key mismatch for setting={key}")


if __name__ == "__main__":
    unittest.main()
