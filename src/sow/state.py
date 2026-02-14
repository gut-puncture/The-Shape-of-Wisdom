from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Tuple


@dataclass(frozen=True)
class HashedPath:
    path: str
    sha256: str


def _fmt_hashed(items: Iterable[HashedPath]) -> str:
    out = []
    for hp in items:
        out.append(f"  - {hp.path}: {hp.sha256}")
    return "\n".join(out) if out else "  - (none)"


def append_state_entry(
    *,
    state_path: Path,
    stage: str,
    status: str,
    command: str,
    inputs: Iterable[HashedPath],
    outputs: Iterable[HashedPath],
    validators: Iterable[HashedPath],
    notes: Optional[str],
    next_step: str,
) -> None:
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M (local)")
    lines = []
    lines.append(f"\n### {ts} - {stage} - {status}")
    lines.append(f"- command: {command}")
    lines.append("- inputs (paths + SHA-256):")
    lines.append(_fmt_hashed(inputs))
    lines.append("- outputs (paths + SHA-256):")
    lines.append(_fmt_hashed(outputs))
    lines.append("- validators (paths + PASS/FAIL):")
    lines.append(_fmt_hashed(validators))
    if notes:
        lines.append(f"- notes: {notes}")
    lines.append(f"- next: {next_step}")
    prev = state_path.read_text(encoding="utf-8") if state_path.exists() else ""
    state_path.write_text(prev + "\n".join(lines) + "\n", encoding="utf-8")
