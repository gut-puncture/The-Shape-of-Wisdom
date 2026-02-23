import importlib.util
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_COMMON = REPO_ROOT / "scripts" / "v2" / "_common.py"
SCRIPT00 = REPO_ROOT / "scripts" / "v2" / "00_run_experiment.py"
SCRIPT05 = REPO_ROOT / "scripts" / "v2" / "05_span_counterfactuals.py"
SCRIPT06 = REPO_ROOT / "scripts" / "v2" / "06_select_tracing_subset.py"
SCRIPT07 = REPO_ROOT / "scripts" / "v2" / "07_run_tracing.py"
SCRIPT11 = REPO_ROOT / "scripts" / "v2" / "11_generate_paper_assets.py"


def _load_common_module():
    if str(SCRIPT_COMMON.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT_COMMON.parent))
    spec = importlib.util.spec_from_file_location("sow_v2_common_atomic", SCRIPT_COMMON)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {SCRIPT_COMMON}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestAtomicWriteContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_common_module()

    def test_write_json_uses_atomic_replace(self) -> None:
        src = inspect.getsource(self.mod._atomic_write_bytes)
        self.assertIn("os.replace", src, msg="atomic writer must commit via os.replace")
        write_json_src = inspect.getsource(self.mod.write_json)
        self.assertIn("_atomic_write_bytes", write_json_src)
        self.assertNotIn("path.write_text", write_json_src, msg="write_json must not write directly to target path")

    def test_write_parquet_writes_to_temp_path_before_commit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "artifact.parquet"
            df = pd.DataFrame.from_records([{"x": 1}])
            original = pd.DataFrame.to_parquet
            seen_paths: list[Path] = []

            def _spy(self_df, path, *args, **kwargs):
                seen_paths.append(Path(path))
                return original(self_df, path, *args, **kwargs)

            with patch("pandas.DataFrame.to_parquet", autospec=True, side_effect=_spy):
                self.mod.write_parquet(out, df)

            self.assertTrue(out.exists())
            self.assertTrue(seen_paths, msg="expected parquet writer call")
            self.assertNotEqual(
                seen_paths[0].resolve(),
                out.resolve(),
                msg="write_parquet must write to a temp file before atomic replace",
            )

    def test_write_text_atomic_exists(self) -> None:
        self.assertTrue(hasattr(self.mod, "write_text_atomic"))
        src = inspect.getsource(self.mod.write_text_atomic)
        self.assertIn("_atomic_write_bytes", src)

    def test_stage_scripts_use_atomic_writer_for_reports(self) -> None:
        s00 = SCRIPT00.read_text(encoding="utf-8")
        s05 = SCRIPT05.read_text(encoding="utf-8")
        s06 = SCRIPT06.read_text(encoding="utf-8")
        s07 = SCRIPT07.read_text(encoding="utf-8")
        s11 = SCRIPT11.read_text(encoding="utf-8")
        self.assertIn("write_json(", s00)
        self.assertNotIn(".write_text(", s00, msg="stage00 should avoid direct report writes")
        self.assertIn("write_text_atomic(", s05)
        self.assertNotIn(".write_text(", s05, msg="stage05 should avoid direct done-sentinel writes")
        self.assertIn("write_json(", s06)
        self.assertNotIn(".write_text(", s06, msg="stage06 should avoid direct subset/report writes")
        self.assertIn("write_text_atomic(", s07)
        self.assertNotIn(".write_text(", s07, msg="stage07 should avoid direct done-sentinel writes")
        self.assertIn("write_json(", s11)
        self.assertNotIn(".write_text(", s11, msg="stage11 should avoid direct final-report writes")


if __name__ == "__main__":
    unittest.main()
