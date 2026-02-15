import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


class _FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 0
        self.padding_side = "left"

    def __call__(self, texts, **kwargs):
        import torch

        if isinstance(texts, str):
            texts = [texts]
        padding = bool(kwargs.get("padding", False))

        # 1 token per character, deterministic; token 5 used as terminator.
        ids = [[(ord(c) % 250) + 1 for c in t] + [5] for t in texts]
        if not padding:
            return {"input_ids": ids}
        maxlen = max(len(x) for x in ids) if ids else 0
        if self.padding_side == "left":
            padded = [[self.pad_token_id] * (maxlen - len(x)) + x for x in ids]
            attn = [[0] * (maxlen - len(x)) + [1] * len(x) for x in ids]
        else:
            padded = [x + [self.pad_token_id] * (maxlen - len(x)) for x in ids]
            attn = [[1] * len(x) + [0] * (maxlen - len(x)) for x in ids]
        return {"input_ids": torch.tensor(padded, dtype=torch.long), "attention_mask": torch.tensor(attn, dtype=torch.long)}

    def decode(self, ids, **kwargs):
        # We only need a deterministic mapping for the generated new tokens.
        if isinstance(ids, int):
            ids = [ids]
        out = []
        for tid in ids:
            if int(tid) == 1:
                out.append("A")
            elif int(tid) == 2:
                out.append("B")
            elif int(tid) == 3:
                out.append("C")
            elif int(tid) == 4:
                out.append("D")
            elif int(tid) == self.pad_token_id:
                out.append("")
            else:
                out.append("x")
        return "".join(out)


class _FakeOut:
    def __init__(self, *, hidden_states):
        self.hidden_states = hidden_states


class _FakeLmHead:
    def __init__(self, *, vocab_size: int, hidden_dim: int):
        import torch

        self.weight = torch.arange(vocab_size * hidden_dim, dtype=torch.float32).view(vocab_size, hidden_dim) * 0.001
        self.bias = None


class _FakeModelInner:
    def __init__(self):
        import torch.nn as nn

        self.norm = nn.Identity()


class _FakeModel:
    def __init__(self, *, n_layers: int, hidden_dim: int, vocab_size: int = 10):
        import torch

        self._n_layers = int(n_layers)
        self._hidden_dim = int(hidden_dim)
        self.model = _FakeModelInner()
        self.lm_head = _FakeLmHead(vocab_size=vocab_size, hidden_dim=hidden_dim)
        self.config = type("Cfg", (), {"vocab_size": int(vocab_size)})
        self._device = "cpu"
        self._dtype = torch.float32

    def eval(self):
        return self

    def to(self, device):
        self._device = str(device)
        return self

    def __call__(self, *, input_ids, attention_mask, use_cache, output_hidden_states, return_dict, position_ids=None):
        import torch

        # Build (embeddings + layers) hidden_states tuple; each layer output is a simple function of input_ids.
        bsz, seqlen = int(input_ids.shape[0]), int(input_ids.shape[1])
        base = input_ids.to(dtype=torch.float32)
        dim = torch.arange(self._hidden_dim, dtype=torch.float32, device=base.device).view(1, 1, -1) * 0.01
        hs = []
        for li in range(self._n_layers + 1):
            # Make layers distinct but deterministic.
            hs.append(base.unsqueeze(-1) + dim + (li * 0.1))
        return _FakeOut(hidden_states=tuple(hs))

    def generate(
        self,
        *,
        input_ids,
        attention_mask,
        do_sample,
        max_new_tokens,
        temperature,
        top_p,
        pad_token_id,
        use_cache,
        position_ids=None,
    ):
        import torch

        bsz, seqlen = int(input_ids.shape[0]), int(input_ids.shape[1])
        # Always generate "A" token id=1 for all steps.
        gen = torch.full((bsz, int(max_new_tokens)), 1, dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, gen], dim=1)


class _OomFakeModel(_FakeModel):
    """
    Fake model that raises an OOM error when batch size exceeds `max_ok_batch`.

    This is used to verify Stage 13 adaptive batch-size logic does not skip rows
    when reducing the batch size after an OOM.
    """

    def __init__(self, *, n_layers: int, hidden_dim: int, vocab_size: int = 10, max_ok_batch: int = 2):
        super().__init__(n_layers=n_layers, hidden_dim=hidden_dim, vocab_size=vocab_size)
        self._max_ok_batch = int(max_ok_batch)

    def __call__(self, *, input_ids, attention_mask, use_cache, output_hidden_states, return_dict, position_ids=None):
        bsz = int(input_ids.shape[0])
        if bsz > int(self._max_ok_batch):
            raise RuntimeError("CUDA out of memory")
        return super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            position_ids=position_ids,
        )

    def generate(
        self,
        *,
        input_ids,
        attention_mask,
        do_sample,
        max_new_tokens,
        temperature,
        top_p,
        pad_token_id,
        use_cache,
        position_ids=None,
    ):
        bsz = int(input_ids.shape[0])
        if bsz > int(self._max_ok_batch):
            raise RuntimeError("CUDA out of memory")
        return super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_token_id,
            use_cache=use_cache,
            position_ids=position_ids,
        )


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


class TestStage13InferenceCheckpoint(unittest.TestCase):
    def test_resume_simulation_writes_full_coverage(self) -> None:
        from sow.inference.stage13 import run_stage13_inference_for_model

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            run_dir = tmp / "run"
            (run_dir / "token_buckets").mkdir(parents=True, exist_ok=True)
            (run_dir / "sentinels").mkdir(parents=True, exist_ok=True)
            (run_dir / "pca").mkdir(parents=True, exist_ok=True)

            model = {"name": "fake", "model_id": "fake/model", "revision": "r0"}
            model_key = "fake__model"

            # Token buckets: A/B/C/D are distinct token ids 1..4.
            tb = {
                "run_id": "test",
                "model_id": model["model_id"],
                "model_revision": model["revision"],
                "buckets": {"A": [1], "B": [2], "C": [3], "D": [4]},
            }
            (run_dir / "token_buckets" / f"{model_key}.json").write_text(json.dumps(tb), encoding="utf-8")

            # PCA basis + sentinel.
            mean = np.zeros((5,), dtype=np.float32)
            components = np.zeros((128, 5), dtype=np.float32)
            components[0, 0] = 1.0
            basis = run_dir / "pca" / f"{model_key}_pca_basis.npz"
            np.savez(basis, mean=mean, components=components, explained_variance_ratio=np.ones((128,), dtype=np.float64) * 0.0)
            meta = run_dir / "pca" / f"{model_key}_pca_basis.meta.json"
            meta.write_text(json.dumps({"basis_hash": "h0"}), encoding="utf-8")
            sent = run_dir / "sentinels" / f"pca_fit.{model_key}.done"

            # sha256 values are checked, so use the real file hashes.
            from sow.hashing import sha256_file

            sent.write_text(
                json.dumps(
                    {
                        "stage": "pca_fit",
                        "run_id": "test",
                        "model_id": model["model_id"],
                        "model_revision": model["revision"],
                        "basis_path": str(basis),
                        "basis_sha256": sha256_file(basis),
                        "meta_path": str(meta),
                        "meta_sha256": sha256_file(meta),
                    }
                ),
                encoding="utf-8",
            )

            # Minimal manifest (5 rows).
            manifest = run_dir / "manifests.jsonl"
            rows = []
            for i in range(5):
                rows.append(
                    {
                        "prompt_uid": f"u{i}",
                        "prompt_id": f"u{i}",
                        "example_id": f"e{i}",
                        "wrapper_id": "plain_exam",
                        "prompt_text": f"Q{i}?\\nA. x\\nB. y\\nC. z\\nD. w\\nAnswer: ",
                        "options": {"A": "x", "B": "y", "C": "z", "D": "w"},
                        "correct_key": "A",
                        "manifest_sha256": "m0",
                        "coarse_domain": "d0",
                    }
                )
            _write_jsonl(manifest, rows)

            out = run_dir / "out.jsonl"

            with patch("transformers.AutoTokenizer.from_pretrained", return_value=_FakeTokenizer()), patch(
                "transformers.AutoModelForCausalLM.from_pretrained", return_value=_FakeModel(n_layers=3, hidden_dim=5)
            ):
                # Stop early, then resume.
                res_stop = run_stage13_inference_for_model(
                    run_id="test",
                    run_dir=run_dir,
                    model=model,
                    generation={"do_sample": False, "max_new_tokens": 24, "temperature": 1.0, "top_p": 1.0},
                    manifest_path=manifest,
                    condition="baseline",
                    batch_size=2,
                    device_override="cpu",
                    output_path_override=out,
                    stop_after_rows=3,
                    commitment_margin_thresholds=[0.1],
                    thermal_hygiene_cfg=None,
                )
                self.assertTrue(res_stop["stopped_early"])

                res_resume = run_stage13_inference_for_model(
                    run_id="test",
                    run_dir=run_dir,
                    model=model,
                    generation={"do_sample": False, "max_new_tokens": 24, "temperature": 1.0, "top_p": 1.0},
                    manifest_path=manifest,
                    condition="baseline",
                    batch_size=2,
                    device_override="cpu",
                    output_path_override=out,
                    stop_after_rows=None,
                    commitment_margin_thresholds=[0.1],
                    thermal_hygiene_cfg=None,
                )
                self.assertTrue(res_resume["pass"])

            # Output must have exactly 5 rows and unique resume keys.
            rows_out = list(__import__("sow.io_jsonl", fromlist=["iter_jsonl"]).iter_jsonl(out))
            self.assertEqual(len(rows_out), 5)
            self.assertEqual(len({r["resume_key"] for r in rows_out}), 5)

    def test_oom_batch_reduction_does_not_skip_rows(self) -> None:
        from sow.inference.stage13 import run_stage13_inference_for_model

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            run_dir = tmp / "run"
            (run_dir / "token_buckets").mkdir(parents=True, exist_ok=True)
            (run_dir / "sentinels").mkdir(parents=True, exist_ok=True)
            (run_dir / "pca").mkdir(parents=True, exist_ok=True)

            model = {"name": "fake", "model_id": "fake/model", "revision": "r0"}
            model_key = "fake__model"

            tb = {
                "run_id": "test",
                "model_id": model["model_id"],
                "model_revision": model["revision"],
                "buckets": {"A": [1], "B": [2], "C": [3], "D": [4]},
            }
            (run_dir / "token_buckets" / f"{model_key}.json").write_text(json.dumps(tb), encoding="utf-8")

            mean = np.zeros((5,), dtype=np.float32)
            components = np.zeros((128, 5), dtype=np.float32)
            components[0, 0] = 1.0
            basis = run_dir / "pca" / f"{model_key}_pca_basis.npz"
            np.savez(basis, mean=mean, components=components, explained_variance_ratio=np.ones((128,), dtype=np.float64) * 0.0)
            meta = run_dir / "pca" / f"{model_key}_pca_basis.meta.json"
            meta.write_text(json.dumps({"basis_hash": "h0"}), encoding="utf-8")
            sent = run_dir / "sentinels" / f"pca_fit.{model_key}.done"

            from sow.hashing import sha256_file

            sent.write_text(
                json.dumps(
                    {
                        "stage": "pca_fit",
                        "run_id": "test",
                        "model_id": model["model_id"],
                        "model_revision": model["revision"],
                        "basis_path": str(basis),
                        "basis_sha256": sha256_file(basis),
                        "meta_path": str(meta),
                        "meta_sha256": sha256_file(meta),
                    }
                ),
                encoding="utf-8",
            )

            manifest = run_dir / "manifests.jsonl"
            rows = []
            for i in range(6):
                rows.append(
                    {
                        "prompt_uid": f"u{i}",
                        "prompt_id": f"u{i}",
                        "example_id": f"e{i}",
                        "wrapper_id": "plain_exam",
                        "prompt_text": f"Q{i}?\\nA. x\\nB. y\\nC. z\\nD. w\\nAnswer: ",
                        "options": {"A": "x", "B": "y", "C": "z", "D": "w"},
                        "correct_key": "A",
                        "manifest_sha256": "m0",
                        "coarse_domain": "d0",
                    }
                )
            _write_jsonl(manifest, rows)

            out = run_dir / "out.jsonl"

            with patch("transformers.AutoTokenizer.from_pretrained", return_value=_FakeTokenizer()), patch(
                "transformers.AutoModelForCausalLM.from_pretrained", return_value=_OomFakeModel(n_layers=3, hidden_dim=5, max_ok_batch=2)
            ):
                res = run_stage13_inference_for_model(
                    run_id="test",
                    run_dir=run_dir,
                    model=model,
                    generation={"do_sample": False, "max_new_tokens": 24, "temperature": 1.0, "top_p": 1.0},
                    manifest_path=manifest,
                    condition="baseline",
                    batch_size=4,  # will OOM, then reduce to 2
                    device_override="cpu",
                    output_path_override=out,
                    stop_after_rows=None,
                    commitment_margin_thresholds=[0.1],
                    thermal_hygiene_cfg=None,
                )
                self.assertTrue(res["pass"])

            rows_out = list(__import__("sow.io_jsonl", fromlist=["iter_jsonl"]).iter_jsonl(out))
            self.assertEqual(len(rows_out), 6)
            self.assertEqual({r["prompt_uid"] for r in rows_out}, {f"u{i}" for i in range(6)})


if __name__ == "__main__":
    unittest.main()
