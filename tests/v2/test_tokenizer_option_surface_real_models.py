import sys
import unittest
from pathlib import Path

import yaml
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


CONFIG_PATH = REPO_ROOT / "configs" / "experiment_v2.yaml"
REQUIRED_MODELS = {
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
}


def _fullwidth(letter: str) -> str:
    return chr(ord(letter) + 0xFEE0)


class TestTokenizerOptionSurfaceRealModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        models = cfg.get("models") or []
        cls.model_specs = [(str(m["model_id"]), str(m["revision"])) for m in models]
        cls.model_ids = {model_id for model_id, _ in cls.model_specs}
        cls.tokenizers = {}
        for model_id, revision in cls.model_specs:
            cls.tokenizers[model_id] = AutoTokenizer.from_pretrained(
                model_id,
                revision=revision,
                use_fast=True,
                trust_remote_code=False,
                local_files_only=True,
            )

    def test_required_models_present_in_config(self) -> None:
        self.assertEqual(self.model_ids, REQUIRED_MODELS)

    def test_practical_answer_variants_are_tokenizable(self) -> None:
        templates = [
            "{L}",
            "{l}",
            " {L}",
            "{L} ",
            "{L}.",
            "{L})",
            "({L})",
            "[{L}]",
            "{L}:",
            "{L},",
            "Answer: {L}",
            "Option {L}",
            "- {L}",
            "{L}\n",
            "{L}\t",
        ]
        for model_id, tok in self.tokenizers.items():
            for letter in ["A", "B", "C", "D"]:
                variants = [tpl.format(L=letter, l=letter.lower()) for tpl in templates]
                variants.append(_fullwidth(letter))
                for variant in variants:
                    with self.subTest(model=model_id, letter=letter, variant=variant):
                        ids = tok.encode(variant, add_special_tokens=False)
                        self.assertIsInstance(ids, list)
                        self.assertGreater(
                            len(ids),
                            0,
                            msg=f"variant must be tokenizable: model={model_id} variant={variant!r}",
                        )

    def test_option_labels_are_multi_token_in_context(self) -> None:
        context_templates = ["({L})", "[{L}]", "Answer: {L}", "Option {L}", "- {L}", "{L}."]
        for model_id, tok in self.tokenizers.items():
            for letter in ["A", "B", "C", "D"]:
                lengths = [len(tok.encode(tpl.format(L=letter), add_special_tokens=False)) for tpl in context_templates]
                with self.subTest(model=model_id, letter=letter):
                    self.assertTrue(
                        any(length > 1 for length in lengths),
                        msg=f"expected at least one multi-token context for {letter} in {model_id}; lengths={lengths}",
                    )

    def test_tokenizer_specific_behavior_differs_across_models(self) -> None:
        enc_a = {
            model_id: tuple(tok.encode("A", add_special_tokens=False))
            for model_id, tok in self.tokenizers.items()
        }
        unique_sequences = set(enc_a.values())
        self.assertGreaterEqual(
            len(unique_sequences),
            2,
            msg=f"expected tokenizer-specific behavior across models for 'A'; got={enc_a}",
        )


if __name__ == "__main__":
    unittest.main()
