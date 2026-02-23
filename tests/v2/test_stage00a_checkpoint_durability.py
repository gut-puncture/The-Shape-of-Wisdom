import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sow.io_jsonl import iter_jsonl  # noqa: E402
from sow.v2.baseline_inference import (  # noqa: E402
    append_jsonl_rows,
    load_completed_resume_keys,
    repair_trailing_partial_line,
    resume_key_for,
)


class TestStage00aCheckpointDurability(unittest.TestCase):
    def test_partial_line_is_repaired_before_resume_scan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "baseline_outputs.jsonl"
            p.write_text(
                json.dumps({"prompt_uid": "u0", "resume_key": resume_key_for(model_id="m", prompt_uid="u0")}, sort_keys=True)
                + "\n"
                + '{"prompt_uid":"u1","resume_key":"broken"',
                encoding="utf-8",
            )
            repair_trailing_partial_line(p)
            keys = load_completed_resume_keys(p)
            self.assertEqual(len(keys), 1)

    def test_checkpoint_append_preserves_committed_rows_after_repair(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "baseline_outputs.jsonl"
            r0 = {"prompt_uid": "u0", "resume_key": resume_key_for(model_id="m", prompt_uid="u0")}
            r1 = {"prompt_uid": "u1", "resume_key": resume_key_for(model_id="m", prompt_uid="u1")}
            r2 = {"prompt_uid": "u2", "resume_key": resume_key_for(model_id="m", prompt_uid="u2")}
            append_jsonl_rows(p, [r0, r1])

            with p.open("ab") as f:
                f.write(b'{"prompt_uid":"partial"')

            repair_trailing_partial_line(p)
            append_jsonl_rows(p, [r2])
            rows = list(iter_jsonl(p))
            self.assertEqual([str(r.get("prompt_uid")) for r in rows], ["u0", "u1", "u2"])


if __name__ == "__main__":
    unittest.main()
