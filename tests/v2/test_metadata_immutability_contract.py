import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT11 = REPO_ROOT / "scripts" / "v2" / "11_generate_paper_assets.py"


def _load_script11():
    if str(SCRIPT11.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT11.parent))
    spec = importlib.util.spec_from_file_location("s11_immutability", SCRIPT11)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {SCRIPT11}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestMetadataImmutabilityContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod11 = _load_script11()

    def test_immutability_check_passes_when_hashes_match(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "doc.md"
            src.write_text("alpha\n", encoding="utf-8")
            snap = root / "snapshot.json"
            rec = {"path": str(src), "sha256": self.mod11._sha256(src)}
            snap.write_text(json.dumps({"files": [rec]}, sort_keys=True) + "\n", encoding="utf-8")

            out = self.mod11._evaluate_metadata_immutability(snap)
            self.assertTrue(bool(out.get("pass")))
            self.assertEqual(out.get("changed_files") or [], [])
            self.assertEqual(out.get("missing_files") or [], [])

    def test_immutability_check_fails_on_hash_change(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "doc.md"
            src.write_text("alpha\n", encoding="utf-8")
            snap = root / "snapshot.json"
            rec = {"path": str(src), "sha256": self.mod11._sha256(src)}
            snap.write_text(json.dumps({"files": [rec]}, sort_keys=True) + "\n", encoding="utf-8")
            src.write_text("beta\n", encoding="utf-8")

            out = self.mod11._evaluate_metadata_immutability(snap)
            self.assertFalse(bool(out.get("pass")))
            self.assertTrue((out.get("changed_files") or []))

    def test_stage11_report_contains_metadata_immutability_gate(self) -> None:
        run_id = "test_v2_stage11_metadata_gate"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        final_root = REPO_ROOT / "artifacts" / "final_result_v2" / run_id
        if run_root.exists():
            shutil.rmtree(run_root)
        if final_root.exists():
            shutil.rmtree(final_root)

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            # satisfy stage11 startup precondition
            import pandas as pd

            pd.DataFrame.from_records(
                [{"model_id": "Qwen/Qwen2.5-7B-Instruct", "prompt_uid": "u0", "layer_index": 0, "delta": 0.1, "drift": 0.0, "boundary": 0.1, "is_correct": True, "entropy": 0.5}]
            ).to_parquet(out_root / "decision_metrics.parquet", index=False)
            pd.DataFrame.from_records(
                [{"model_id": "Qwen/Qwen2.5-7B-Instruct", "prompt_uid": "u0", "trajectory_type": "stable_correct", "is_correct": True}]
            ).to_parquet(out_root / "prompt_types.parquet", index=False)

            proc = __import__("subprocess").run(
                [sys.executable, str(SCRIPT11), "--run-id", run_id],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 2)
            rep = json.loads((final_root / "final_report.json").read_text(encoding="utf-8"))
            self.assertIn("metadata_immutability", rep)
            self.assertIn("metadata_immutability", rep.get("failing_gates") or [])
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)
            if final_root.exists():
                shutil.rmtree(final_root)

    def test_stage11_fails_closed_when_only_metadata_gate_fails(self) -> None:
        run_id = "test_v2_stage11_metadata_only_fail"
        run_root = REPO_ROOT / "runs" / run_id
        out_root = run_root / "v2"
        final_root = REPO_ROOT / "artifacts" / "final_result_v2" / run_id
        if run_root.exists():
            shutil.rmtree(run_root)
        if final_root.exists():
            shutil.rmtree(final_root)

        original_required_files = list(self.mod11.REQUIRED_FILES)
        original_required_metadata_files = list(self.mod11.REQUIRED_METADATA_FILES)
        original_argv = list(sys.argv)
        try:
            out_root.mkdir(parents=True, exist_ok=True)
            import pandas as pd

            pd.DataFrame.from_records(
                [{"model_id": "Qwen/Qwen2.5-7B-Instruct", "prompt_uid": "u0", "layer_index": 0, "delta": 0.1, "drift": 0.0, "boundary": 0.1, "is_correct": True, "entropy": 0.5}]
            ).to_parquet(out_root / "decision_metrics.parquet", index=False)
            pd.DataFrame.from_records(
                [{"model_id": "Qwen/Qwen2.5-7B-Instruct", "prompt_uid": "u0", "trajectory_type": "stable_correct", "is_correct": True}]
            ).to_parquet(out_root / "prompt_types.parquet", index=False)

            # Keep required-file contract minimal so this test isolates metadata immutability behavior.
            self.mod11.REQUIRED_FILES = ["decision_metrics.parquet", "prompt_types.parquet"]
            self.mod11.REQUIRED_METADATA_FILES = []

            tracked = out_root / "meta" / "tracked_metadata.md"
            tracked.parent.mkdir(parents=True, exist_ok=True)
            tracked.write_text("alpha\n", encoding="utf-8")
            snapshot = out_root / "meta" / "run_start_metadata_snapshot.json"
            snapshot.write_text(
                json.dumps({"files": [{"path": str(tracked), "sha256": "0" * 64}]}, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            sys.argv = [str(SCRIPT11), "--run-id", run_id]
            rc = int(self.mod11.main())
            self.assertEqual(rc, 2)

            rep = json.loads((final_root / "final_report.json").read_text(encoding="utf-8"))
            gates = rep.get("gates") or {}
            self.assertTrue(bool(gates.get("required_files_present")))
            self.assertFalse(bool(gates.get("metadata_immutability")))
            self.assertIn("metadata_immutability", rep.get("failing_gates") or [])
        finally:
            self.mod11.REQUIRED_FILES = original_required_files
            self.mod11.REQUIRED_METADATA_FILES = original_required_metadata_files
            sys.argv = original_argv
            if run_root.exists():
                shutil.rmtree(run_root)
            if final_root.exists():
                shutil.rmtree(final_root)


if __name__ == "__main__":
    unittest.main()
