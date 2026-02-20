import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


from sow.v2.assets import write_sha_manifest  # noqa: E402


class TestArtifactHashManifestContract(unittest.TestCase):
    def test_manifest_entries_are_complete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "bundle"
            root.mkdir(parents=True, exist_ok=True)
            (root / "a.txt").write_text("alpha\n", encoding="utf-8")
            (root / "b.txt").write_text("beta\n", encoding="utf-8")
            out_path = root / "sha256_manifest.json"

            write_sha_manifest(root_dir=root, out_path=out_path)
            rows = json.loads(out_path.read_text(encoding="utf-8"))

            self.assertGreaterEqual(len(rows), 2)
            for row in rows:
                self.assertIn("path", row)
                self.assertIn("sha256", row)
                self.assertEqual(len(str(row["sha256"])), 64)
            row_paths = {str(r["path"]) for r in rows}
            self.assertIn(str(root / "a.txt"), row_paths)
            self.assertIn(str(root / "b.txt"), row_paths)

    def test_sha_manifest_is_idempotent_across_reruns(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "bundle"
            root.mkdir(parents=True, exist_ok=True)
            (root / "a.txt").write_text("alpha\n", encoding="utf-8")
            (root / "b.txt").write_text("beta\n", encoding="utf-8")
            out_path = root / "sha256_manifest.json"

            write_sha_manifest(root_dir=root, out_path=out_path)
            first = out_path.read_text(encoding="utf-8")
            write_sha_manifest(root_dir=root, out_path=out_path)
            second = out_path.read_text(encoding="utf-8")

            self.assertEqual(first, second, msg="sha256 manifest should be byte-stable across reruns")


if __name__ == "__main__":
    unittest.main()
