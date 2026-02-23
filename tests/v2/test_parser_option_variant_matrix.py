import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


from sow.judging.deterministic_parser import parse_choice  # noqa: E402


def _fullwidth(letter: str) -> str:
    return chr(ord(letter) + 0xFEE0)


def _options() -> dict[str, str]:
    return {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"}


class TestParserOptionVariantMatrix(unittest.TestCase):
    def test_full_text_variant_matrix_resolves_letters(self) -> None:
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
        for letter in ["A", "B", "C", "D"]:
            lower = letter.lower()
            variants = [tpl.format(L=letter, l=lower) for tpl in templates]
            variants.append(_fullwidth(letter))
            for variant in variants:
                with self.subTest(letter=letter, variant=variant):
                    out = parse_choice(response_text=variant, first_token=None, options=_options())
                    self.assertEqual(
                        out["parsed_choice"],
                        letter,
                        msg=f"full-text parser failed for variant={variant!r}; out={out}",
                    )
                    self.assertEqual(out["decision"], "resolved_letter")

    def test_first_token_path_handles_practical_token_forms(self) -> None:
        token_templates = [
            "{L}",
            "({L}",
            "[{L}",
            "\n{L}",
            " {L}",
            "{L}.",
            "Ġ{L}",
            "▁{L}",
            "Ġ{L}.",
            "▁({L}",
        ]
        for letter in ["A", "B", "C", "D"]:
            tokens = [tpl.format(L=letter) for tpl in token_templates]
            tokens.append(_fullwidth(letter))
            for token in tokens:
                with self.subTest(letter=letter, first_token=token):
                    out = parse_choice(response_text="", first_token=token, options=_options())
                    self.assertEqual(
                        out["parsed_choice"],
                        letter,
                        msg=f"first-token parser failed for token={token!r}; out={out}",
                    )
                    self.assertEqual(out["decision"], "resolved_letter_first_token")

    def test_first_token_wins_over_conflicting_later_mentions(self) -> None:
        for letter, conflicting in [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]:
            with self.subTest(letter=letter, conflicting=conflicting):
                out = parse_choice(
                    response_text=f"Answer: {conflicting}. Actually option {conflicting} is correct.",
                    first_token=f" {letter}",
                    options=_options(),
                )
                self.assertEqual(out["parsed_choice"], letter, msg=f"first-token precedence failed; out={out}")
                self.assertEqual(out["decision"], "resolved_letter_first_token")

    def test_ambiguous_in_word_text_stays_unresolved(self) -> None:
        for text in ["Aardvark", "planB", "catC", "Dynamo"]:
            with self.subTest(text=text):
                out = parse_choice(response_text=text, first_token=None, options=_options())
                self.assertIsNone(out["parsed_choice"], msg=f"expected unresolved for in-word signal text={text!r}; out={out}")
                self.assertEqual(out["decision"], "unresolved_no_signal")


if __name__ == "__main__":
    unittest.main()
