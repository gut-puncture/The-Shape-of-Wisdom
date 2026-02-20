import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


from sow.v2.model_nuances import assert_transformers_version_floor  # noqa: E402


class TestModelNuancesVersionFloor(unittest.TestCase):
    def test_accepts_equal_or_higher_version(self) -> None:
        assert_transformers_version_floor("Qwen/Qwen2.5-7B-Instruct", "4.37.0")
        assert_transformers_version_floor("meta-llama/Llama-3.1-8B-Instruct", "4.50.1")

    def test_rejects_lower_version(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_transformers_version_floor("meta-llama/Llama-3.1-8B-Instruct", "4.42.9")


if __name__ == "__main__":
    unittest.main()
