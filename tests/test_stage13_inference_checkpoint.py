import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "v2" / "run_with_thermal_resume.sh"


class TestStage13InferenceCheckpoint(unittest.TestCase):
    def test_loops_on_95_then_exits_on_done_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            done = root / "done.json"
            marker = root / "count.txt"
            cmd = (
                "c=$(cat \"$1\" 2>/dev/null || echo 0); "
                "c=$((c+1)); "
                "echo \"$c\" > \"$1\"; "
                "if [ \"$c\" -eq 1 ]; then exit 95; fi; "
                "echo '{\"pass\":true}' > \"$2\"; "
                "exit 0"
            )
            proc = subprocess.run(
                [
                    str(SCRIPT),
                    "--done-sentinel",
                    str(done),
                    "--cooldown-seconds",
                    "0",
                    "--",
                    "sh",
                    "-c",
                    cmd,
                    "_",
                    str(marker),
                    str(done),
                ],
                cwd=str(REPO_ROOT),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertTrue(done.exists())
            self.assertEqual((marker.read_text(encoding="utf-8").strip()), "2")
            payload = json.loads(done.read_text(encoding="utf-8"))
            self.assertTrue(bool(payload.get("pass")))

    def test_non_95_failure_propagates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            done = root / "done.json"
            proc = subprocess.run(
                [
                    str(SCRIPT),
                    "--done-sentinel",
                    str(done),
                    "--cooldown-seconds",
                    "0",
                    "--",
                    "sh",
                    "-c",
                    "exit 7",
                ],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 7)
            self.assertFalse(done.exists())

    def test_existing_done_sentinel_short_circuits(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            done = root / "done.json"
            done.write_text("{\"pass\":true}\n", encoding="utf-8")
            marker = root / "marker.txt"
            proc = subprocess.run(
                [
                    str(SCRIPT),
                    "--done-sentinel",
                    str(done),
                    "--cooldown-seconds",
                    "0",
                    "--",
                    "sh",
                    "-c",
                    f"echo should-not-run > {marker}",
                ],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 0)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
