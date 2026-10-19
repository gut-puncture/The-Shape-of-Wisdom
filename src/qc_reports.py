from __future__ import annotations

from collections import Counter
from typing import Any


def build_qc_summary(
    *,
    canonical_examples: list[dict[str, Any]],
    token_audits: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    paired_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    example_flags = Counter()
    exclusions = Counter()
    for example in canonical_examples:
        for flag in example.get("anomaly_flags", []):
            example_flags[flag] += 1
        if example.get("exclusion_reason"):
            exclusions[example["exclusion_reason"]] += 1
    token_flags = Counter()
    for audit in token_audits:
        for flag in audit.get("suspicious_flags", []):
            token_flags[flag] += 1
    failure_counts = Counter(record.get("reason_code", "unknown") for record in failures)
    return {
        "example_flag_counts": dict(sorted(example_flags.items())),
        "exclusion_counts": dict(sorted(exclusions.items())),
        "token_flag_counts": dict(sorted(token_flags.items())),
        "failure_counts": dict(sorted(failure_counts.items())),
        "paired_coverage": paired_coverage or {},
        "total_examples": len(canonical_examples),
        "total_token_audits": len(token_audits),
        "total_failures": len(failures),
    }


def render_qc_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# QC Summary",
        "",
        f"- Total examples: {summary.get('total_examples', 0)}",
        f"- Total token audits: {summary.get('total_token_audits', 0)}",
        f"- Total failures: {summary.get('total_failures', 0)}",
        "",
        "## Example Flags",
    ]
    for key, value in (summary.get("example_flag_counts") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Exclusions"])
    for key, value in (summary.get("exclusion_counts") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Token Flags"])
    for key, value in (summary.get("token_flag_counts") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Failures"])
    for key, value in (summary.get("failure_counts") or {}).items():
        lines.append(f"- {key}: {value}")
    if summary.get("paired_coverage"):
        lines.extend(["", "## Paired Coverage"])
        for key, value in summary["paired_coverage"].items():
            lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"

