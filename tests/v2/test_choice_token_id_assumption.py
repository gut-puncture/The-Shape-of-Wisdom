import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT05 = REPO_ROOT / "scripts" / "v2" / "05_span_counterfactuals.py"
SCRIPT07 = REPO_ROOT / "scripts" / "v2" / "07_run_tracing.py"


def _load_script(module_name: str, script_path: Path):
    if str(script_path.parent) not in sys.path:
        sys.path.insert(0, str(script_path.parent))
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _MultiTokenTokenizer:
    def __init__(self) -> None:
        self._map = {
            "A": [101, 102],
            "B": [201, 202],
            "C": [301, 302],
            "D": [401, 402],
        }

    def encode(self, text, add_special_tokens=False):
        _ = add_special_tokens
        return list(self._map.get(str(text), []))


class _SingleTokenTokenizer:
    def __init__(self) -> None:
        self._map = {"A": [11], "B": [12], "C": [13], "D": [14]}

    def encode(self, text, add_special_tokens=False):
        _ = add_special_tokens
        return list(self._map.get(str(text), []))


class TestChoiceTokenIdAssumption(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod05 = _load_script("s05_choice_ids", SCRIPT05)
        cls.mod07 = _load_script("s07_choice_ids", SCRIPT07)

    def test_multi_token_option_labels_are_not_silently_truncated(self) -> None:
        tok = _MultiTokenTokenizer()
        for name, module in [("05_span_counterfactuals", self.mod05), ("07_run_tracing", self.mod07)]:
            with self.subTest(module=name):
                got = module._choice_token_ids(tok)
                for letter in ["A", "B", "C", "D"]:
                    self.assertIsNone(
                        got[letter],
                        msg=f"{name} should not silently truncate multi-token label {letter}; got={got[letter]}",
                    )

    def test_single_token_option_labels_remain_supported(self) -> None:
        tok = _SingleTokenTokenizer()
        for name, module in [("05_span_counterfactuals", self.mod05), ("07_run_tracing", self.mod07)]:
            with self.subTest(module=name):
                got = module._choice_token_ids(tok)
                self.assertEqual(got, {"A": 11, "B": 12, "C": 13, "D": 14})

    def test_choice_token_validation_rejects_missing_single_token_labels(self) -> None:
        bad = {"A": None, "B": 12, "C": 13, "D": 14}
        for name, module in [("05_span_counterfactuals", self.mod05), ("07_run_tracing", self.mod07)]:
            with self.subTest(module=name):
                with self.assertRaises(ValueError):
                    module._validate_choice_token_ids(bad)

    def test_choice_token_validation_accepts_complete_single_token_labels(self) -> None:
        good = {"A": 11, "B": 12, "C": 13, "D": 14}
        for name, module in [("05_span_counterfactuals", self.mod05), ("07_run_tracing", self.mod07)]:
            with self.subTest(module=name):
                module._validate_choice_token_ids(good)


if __name__ == "__main__":
    unittest.main()
