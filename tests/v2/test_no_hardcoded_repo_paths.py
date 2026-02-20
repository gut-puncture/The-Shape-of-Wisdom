import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HARD_CODED_REPO = "/Users/shaileshrana/shape-of-wisdom"


class TestNoHardcodedRepoPaths(unittest.TestCase):
    def test_active_code_has_no_hardcoded_repo_path(self) -> None:
        roots = [REPO_ROOT / "scripts" / "v2", REPO_ROOT / "src" / "sow"]
        offenders = []
        for root in roots:
            for path in root.rglob("*.py"):
                text = path.read_text(encoding="utf-8")
                if HARD_CODED_REPO in text:
                    offenders.append(str(path))

        self.assertFalse(
            offenders,
            msg=(
                "absolute workspace paths break reproducibility/portability; found hardcoded repo path in: "
                f"{offenders}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
