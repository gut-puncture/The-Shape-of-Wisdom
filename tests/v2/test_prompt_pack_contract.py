import hashlib
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PACK = REPO_ROOT / "prompt_packs" / "ccc_baseline_v1_3000.jsonl"
EXPECTED_SHA256 = "bfe2557316cb1e0eae6a684eb8de84885f74e446ac72f1847a7da80baf2de56c"


class TestPromptPackContract(unittest.TestCase):
    def test_prompt_pack_exists_and_hash_matches(self) -> None:
        self.assertTrue(PROMPT_PACK.exists(), msg=f"missing canonical prompt pack: {PROMPT_PACK}")
        h = hashlib.sha256()
        with PROMPT_PACK.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        self.assertEqual(h.hexdigest(), EXPECTED_SHA256)

    def test_prompt_pack_row_count_is_3000(self) -> None:
        with PROMPT_PACK.open("r", encoding="utf-8") as f:
            rows = sum(1 for _ in f)
        self.assertEqual(rows, 3000)


if __name__ == "__main__":
    unittest.main()
