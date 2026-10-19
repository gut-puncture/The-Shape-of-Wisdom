from __future__ import annotations

from collections import Counter
from typing import Any

from dataset_loader import CanonicalExample


def longest_shared_prefix_length(strings: list[str]) -> int:
    if not strings:
        return 0
    prefix = strings[0]
    for value in strings[1:]:
        limit = min(len(prefix), len(value))
        idx = 0
        while idx < limit and prefix[idx] == value[idx]:
            idx += 1
        prefix = prefix[:idx]
        if not prefix:
            return 0
    return len(prefix)


def confound_flags_for_example(example: CanonicalExample) -> list[str]:
    texts = [option.raw_text for option in example.options]
    flags: list[str] = []
    if any(text.lower() in {"all of the above", "none of the above"} for text in texts):
        flags.append("all_none_option")
    lowered = [text.lower() for text in texts]
    if len(set(lowered)) < len(lowered):
        flags.append("case_only_or_duplicate_option")
    if any(text.startswith(("A.", "B.", "C.", "D.", "(A)", "(B)", "(C)", "(D)")) for text in texts):
        flags.append("option_contains_embedded_label")
    if longest_shared_prefix_length(texts) > 10:
        flags.append("long_shared_prefix")
    counts = Counter(text.strip() for text in texts)
    if any(count > 1 for count in counts.values()):
        flags.append("duplicate_or_near_duplicate")
    return sorted(set(flags))


def paired_coverage_summary(rows: list[dict[str, Any]], *, required_fields: list[str]) -> dict[str, Any]:
    total = len(rows)
    paired = [
        row for row in rows
        if all(bool(row.get(field)) for field in required_fields)
    ]
    return {
        "total_rows": total,
        "paired_rows": len(paired),
        "unpaired_rows": total - len(paired),
        "required_fields": required_fields,
    }
