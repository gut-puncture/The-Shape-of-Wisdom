from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from io_utils import write_jsonl_atomic


@dataclass(frozen=True)
class FailureRecord:
    stage: str
    unit_key: str
    severity: str
    reason_code: str
    message: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FailureRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._records: list[FailureRecord] = []
        if path.exists():
            import json

            self._records = [
                FailureRecord(**json.loads(line))
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    @property
    def records(self) -> list[FailureRecord]:
        return list(self._records)

    def archive_existing(self, *, run_label: str) -> Path | None:
        if not self._records:
            write_jsonl_atomic(self.path, [])
            return None
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_dir = self.path.parent / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{self.path.stem}_{run_label}_{timestamp}{self.path.suffix}"
        write_jsonl_atomic(archive_path, [item.to_dict() for item in self._records])
        self._records = []
        write_jsonl_atomic(self.path, [])
        return archive_path

    def record(self, record: FailureRecord) -> None:
        self._records.append(record)
        write_jsonl_atomic(self.path, [item.to_dict() for item in self._records])
