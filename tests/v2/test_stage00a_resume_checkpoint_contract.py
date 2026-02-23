import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


from sow.v2.baseline_inference import (  # noqa: E402
    checkpoint_flush_required,
    resume_key_for,
    select_pending_manifest_rows,
)


class TestStage00aResumeCheckpointContract(unittest.TestCase):
    def test_resume_mode_deduplicates_completed_prompt_keys(self) -> None:
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        manifest_rows = [
            {"prompt_uid": "u0"},
            {"prompt_uid": "u1"},
            {"prompt_uid": ""},
            {"prompt_uid": "u2"},
        ]
        completed = {resume_key_for(model_id=model_id, prompt_uid="u1")}

        pending, stats = select_pending_manifest_rows(
            manifest_rows=manifest_rows,
            model_id=model_id,
            completed_keys=completed,
        )

        self.assertEqual([str(r.get("prompt_uid")) for r in pending], ["u0", "u2"])
        self.assertEqual(int(stats.get("resume_rows_total", -1)), 3)
        self.assertEqual(int(stats.get("resume_rows_skipped", -1)), 1)
        self.assertEqual(int(stats.get("pending_rows", -1)), 2)

    def test_checkpoint_flush_frequency_obeys_contract(self) -> None:
        rows_since_checkpoint = 0
        flush_points = []
        for idx in range(1, 6):
            rows_since_checkpoint += 1
            if checkpoint_flush_required(rows_since_checkpoint=rows_since_checkpoint, checkpoint_every=2):
                flush_points.append(idx)
                rows_since_checkpoint = 0

        self.assertEqual(flush_points, [2, 4])


if __name__ == "__main__":
    unittest.main()
