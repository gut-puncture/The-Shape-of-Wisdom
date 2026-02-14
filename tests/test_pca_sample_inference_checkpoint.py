import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


class _FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 0

    def __call__(self, texts, **kwargs):
        import torch

        if isinstance(texts, str):
            texts = [texts]
        return_tensors = kwargs.get("return_tensors")
        padding = bool(kwargs.get("padding", False))
        return_length = bool(kwargs.get("return_length", False))

        # Deterministic toy tokenization: 1 token per character + a terminator token.
        ids = [[(ord(c) % 250) + 1 for c in t] + [1] for t in texts]
        lengths = [len(x) for x in ids]

        if not padding:
            out = {"input_ids": ids}
            if return_length:
                out["length"] = lengths
            return out

        maxlen = max(lengths) if lengths else 0
        padded = [x + [self.pad_token_id] * (maxlen - len(x)) for x in ids]
        attn = [[1] * len(x) + [0] * (maxlen - len(x)) for x in ids]

        if return_tensors == "pt":
            out = {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
            }
        else:
            out = {"input_ids": padded, "attention_mask": attn}
        if return_length:
            out["length"] = lengths
        return out


class _FakeOut:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _FakeModel:
    def __init__(self, *, n_layers: int, hidden_dim: int, oom_batch_gt: int | None = None, crash_after_calls: int | None = None) -> None:
        self._n_layers = int(n_layers)
        self._hidden_dim = int(hidden_dim)
        self._oom_batch_gt = oom_batch_gt
        self._crash_after_calls = crash_after_calls
        self._calls = 0

    def eval(self):
        return self

    def to(self, _device: str):
        return self

    def __call__(self, *, input_ids, attention_mask, use_cache, output_hidden_states, return_dict):
        import torch

        self._calls += 1
        bsz = int(input_ids.shape[0])
        if self._oom_batch_gt is not None and bsz > int(self._oom_batch_gt):
            raise RuntimeError("out of memory")
        if self._crash_after_calls is not None and self._calls == int(self._crash_after_calls):
            raise RuntimeError("simulated crash")

        # Build (embeddings + layers) hidden_states tuple, shape (batch, seq, hidden_dim).
        base = input_ids.to(dtype=torch.float32)
        dim = torch.arange(self._hidden_dim, dtype=torch.float32, device=base.device).view(1, 1, -1) * 0.01
        hs = []
        for li in range(self._n_layers + 1):
            hs.append(base.unsqueeze(-1) + dim + (li * 0.1))
        return _FakeOut(tuple(hs))


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


class TestPCASampleInferenceCheckpointing(unittest.TestCase):
    def _setup_run_dir(self, tmp: Path) -> dict:
        run_dir = tmp / "run"
        (run_dir / "meta").mkdir(parents=True, exist_ok=True)
        (run_dir / "pca").mkdir(parents=True, exist_ok=True)

        # Minimal config snapshot files required by the stage.
        (run_dir / "run_config.yaml").write_text("run_id: test\n", encoding="utf-8")
        (run_dir / "meta" / "config_snapshot.yaml").write_text("run_id: test\n", encoding="utf-8")

        baseline = run_dir / "manifests" / "baseline_manifest.jsonl"
        robust = run_dir / "manifests" / "robustness_manifest_v2.jsonl"
        baseline.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        membership = []
        for i in range(9):
            uid = f"u{i}"
            rows.append(
                {
                    "prompt_uid": uid,
                    "prompt_id": uid,
                    "example_id": f"e{i}",
                    "wrapper_id": "baseline",
                    "prompt_text": f"Q{i}?\nA. x\nB. y\nC. z\nD. w\nAnswer: ",
                    "options": ["x", "y", "z", "w"],
                    "correct_key": "A",
                }
            )
            membership.append({"prompt_uid": uid, "wrapper_id": "baseline", "coarse_domain": "d0"})

        _write_jsonl(baseline, rows)
        robust.write_text("", encoding="utf-8")

        membership_path = run_dir / "pca" / "fake_sample_membership.json"
        membership_path.write_text(json.dumps({"seed": 123, "membership": membership}, indent=2) + "\n", encoding="utf-8")

        return {
            "run_dir": run_dir,
            "baseline": baseline,
            "robust": robust,
            "membership": membership_path,
        }

    def test_checkpoint_resume_produces_identical_hidden(self) -> None:
        from sow.pca.sample_inference import run_pca_sample_inference_for_model

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            env = self._setup_run_dir(tmp)
            run_dir = env["run_dir"]

            model_cfg = {"name": "fake", "model_id": "fake/model", "revision": "r0"}
            gen_cfg = {"do_sample": False, "max_new_tokens": 24, "temperature": 1.0, "top_p": 1.0}

            # First run: crash mid-way after progress is written.
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=_FakeTokenizer()), patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                return_value=_FakeModel(n_layers=3, hidden_dim=5, crash_after_calls=3),
            ):
                with self.assertRaises(RuntimeError):
                    run_pca_sample_inference_for_model(
                        run_id="test",
                        run_dir=run_dir,
                        model=model_cfg,
                        generation=gen_cfg,
                        baseline_manifest=env["baseline"],
                        robustness_manifest=env["robust"],
                        membership_path=env["membership"],
                        device_override="cpu",
                        batch_size="2",
                        repro_check_k=3,
                        repro_atol=1e-6,
                        thermal_hygiene_cfg=None,
                    )

            # Second run: resume and finish.
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=_FakeTokenizer()), patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                return_value=_FakeModel(n_layers=3, hidden_dim=5),
            ):
                res_resume = run_pca_sample_inference_for_model(
                    run_id="test",
                    run_dir=run_dir,
                    model=model_cfg,
                    generation=gen_cfg,
                    baseline_manifest=env["baseline"],
                    robustness_manifest=env["robust"],
                    membership_path=env["membership"],
                    device_override="cpu",
                    batch_size="2",
                    repro_check_k=3,
                    repro_atol=1e-6,
                    thermal_hygiene_cfg=None,
                )

            self.assertTrue(res_resume["report"]["pass"])

            # Reference run from scratch in a fresh run_dir: must match exactly.
            env2 = self._setup_run_dir(tmp / "run2")
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=_FakeTokenizer()), patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                return_value=_FakeModel(n_layers=3, hidden_dim=5),
            ):
                res_ref = run_pca_sample_inference_for_model(
                    run_id="test2",
                    run_dir=env2["run_dir"],
                    model=model_cfg,
                    generation=gen_cfg,
                    baseline_manifest=env2["baseline"],
                    robustness_manifest=env2["robust"],
                    membership_path=env2["membership"],
                    device_override="cpu",
                    batch_size="2",
                    repro_check_k=3,
                    repro_atol=1e-6,
                    thermal_hygiene_cfg=None,
                )

            import numpy as np

            with np.load(res_resume["hidden_path"]) as z1, np.load(res_ref["hidden_path"]) as z2:
                a = z1["hidden"]
                b = z2["hidden"]
            self.assertEqual(a.shape, b.shape)
            self.assertEqual(a.dtype, b.dtype)
            self.assertTrue(np.array_equal(a, b))

    def test_auto_batch_calibration_obeys_oom_and_passes_invariance(self) -> None:
        from sow.pca.sample_inference import run_pca_sample_inference_for_model

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            env = self._setup_run_dir(tmp)
            run_dir = env["run_dir"]

            model_cfg = {"name": "fake", "model_id": "fake/model", "revision": "r0"}
            gen_cfg = {"do_sample": False, "max_new_tokens": 24, "temperature": 1.0, "top_p": 1.0}

            with patch("transformers.AutoTokenizer.from_pretrained", return_value=_FakeTokenizer()), patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                return_value=_FakeModel(n_layers=2, hidden_dim=4, oom_batch_gt=2),
            ):
                res = run_pca_sample_inference_for_model(
                    run_id="test",
                    run_dir=run_dir,
                    model=model_cfg,
                    generation=gen_cfg,
                    baseline_manifest=env["baseline"],
                    robustness_manifest=env["robust"],
                    membership_path=env["membership"],
                    device_override="cpu",
                    batch_size="auto",
                    repro_check_k=3,
                    repro_atol=1e-6,
                    thermal_hygiene_cfg=None,
                )

            self.assertTrue(res["report"]["pass"])
            meta = json.loads(Path(res["meta_path"]).read_text(encoding="utf-8"))
            # Stage 11 does not gate on batch invariance; it is recorded informationally.
            self.assertIn("max_abs_diff", meta["batch_invariance_check"])
            # Auto calibration should never exceed oom_batch_gt=2 in steady state.
            used = meta["batching"]["batch_sizes_used"]
            self.assertTrue(all(int(x) <= 2 for x in used))


if __name__ == "__main__":
    unittest.main()
