from __future__ import annotations

import json

from failure_registry import FailureRecord, FailureRegistry


def test_archive_existing_resets_current_log_and_preserves_history(tmp_path) -> None:
    path = tmp_path / "failures.jsonl"
    registry = FailureRegistry(path)
    registry.record(
        FailureRecord(
            stage="run_controls",
            unit_key="unit-1",
            severity="error",
            reason_code="example_failed",
            message="example failed",
            metadata={"model_name": "toy"},
        )
    )

    archive_path = registry.archive_existing(run_label="smoke")

    assert archive_path is not None
    assert archive_path.exists()
    archived_rows = [json.loads(line) for line in archive_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(archived_rows) == 1
    assert archived_rows[0]["reason_code"] == "example_failed"
    assert path.exists()
    assert path.read_text(encoding="utf-8") == ""
    assert registry.records == []
