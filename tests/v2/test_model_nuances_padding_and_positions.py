import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


from sow.v2.model_nuances import apply_tokenizer_nuance  # noqa: E402


def _position_ids_from_attention_mask(attn):
    attn_i = attn.long()
    pos = attn_i.cumsum(dim=1) - 1
    pos = pos.masked_fill(attn_i == 0, 0)
    return pos.long()


class _DummyTokenizer:
    def __init__(self, *, pad_token_id=None, eos_token_id=None, padding_side="right") -> None:
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.padding_side = padding_side


class TestModelNuancesPaddingAndPositions(unittest.TestCase):
    def test_apply_tokenizer_nuance_enforces_left_padding_and_eos_fallback(self) -> None:
        for model_id in [
            "Qwen/Qwen2.5-7B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ]:
            tok = _DummyTokenizer(pad_token_id=None, eos_token_id=9, padding_side="right")
            apply_tokenizer_nuance(tok, model_id=model_id)
            self.assertEqual(tok.padding_side, "left")
            self.assertEqual(tok.pad_token_id, 9)

    def test_apply_tokenizer_nuance_does_not_overwrite_existing_pad_token(self) -> None:
        tok = _DummyTokenizer(pad_token_id=3, eos_token_id=9, padding_side="right")
        apply_tokenizer_nuance(tok, model_id="Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(tok.padding_side, "left")
        self.assertEqual(tok.pad_token_id, 3)

    def test_position_ids_match_attention_mask_contract(self) -> None:
        attn = torch.tensor(
            [
                [0, 0, 1, 1, 1],
                [1, 1, 1, 0, 0],
                [0, 1, 1, 0, 1],
            ],
            dtype=torch.long,
        )
        got = _position_ids_from_attention_mask(attn)
        expected = torch.tensor(
            [
                [0, 0, 0, 1, 2],
                [0, 1, 2, 0, 0],
                [0, 0, 1, 0, 2],
            ],
            dtype=torch.long,
        )
        self.assertTrue(torch.equal(got, expected), msg=f"position ids mismatch\nexpected={expected}\ngot={got}")
        self.assertEqual(got.dtype, torch.long)


if __name__ == "__main__":
    unittest.main()
