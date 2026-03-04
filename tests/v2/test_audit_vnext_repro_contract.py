import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestAuditVnextReproContract(unittest.TestCase):
    def test_reproduce_vnext_fails_loudly_when_required_artifacts_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            missing_parquet = Path(td) / "parquet"
            missing_parquet.mkdir(parents=True, exist_ok=True)
            out_json = Path(td) / "manifest.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit/reproduce_vnext.py",
                    "--parquet-dir",
                    str(missing_parquet),
                    "--out-json",
                    str(out_json),
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            msg = (proc.stderr or "") + "\n" + (proc.stdout or "")
            self.assertIn("missing required artifact", msg.lower())


if __name__ == "__main__":
    unittest.main()

