import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT05 = REPO_ROOT / "scripts" / "v2" / "05_span_counterfactuals.py"


def _load_script(module_name: str, script_path: Path):
    if str(script_path.parent) not in sys.path:
        sys.path.insert(0, str(script_path.parent))
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSpanCounterfactualResumeMerge(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod05 = _load_script("s05_resume_merge", SCRIPT05)

    def test_resume_merge_prefers_new_rows_for_duplicate_span_keys(self) -> None:
        existing = [
            {
                "model_id": "m",
                "prompt_uid": "u",
                "span_id": "s1",
                "span_role": "old_role",
                "start_char": 1,
                "end_char": 2,
            }
        ]
        new = [
            {
                "model_id": "m",
                "prompt_uid": "u",
                "span_id": "s1",
                "span_role": "new_role",
                "start_char": 10,
                "end_char": 20,
            }
        ]

        merged = self.mod05._merge_span_rows(existing_rows=existing, new_rows=new)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["span_role"], "new_role")
        self.assertEqual(merged[0]["start_char"], 10)
        self.assertEqual(merged[0]["end_char"], 20)


if __name__ == "__main__":
    unittest.main()
