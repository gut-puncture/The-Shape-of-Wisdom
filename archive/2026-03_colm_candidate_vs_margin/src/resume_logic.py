from __future__ import annotations


def unresolved_failure_unit_keys(failure_rows: list[dict], success_rows: list[dict]) -> list[str]:
    succeeded = {str(row.get("unit_key")) for row in success_rows if row.get("unit_key")}
    unresolved = []
    seen = set()
    for row in failure_rows:
        unit_key = str(row.get("unit_key") or "")
        if not unit_key or unit_key in succeeded or unit_key in seen:
            continue
        seen.add(unit_key)
        unresolved.append(unit_key)
    return unresolved
