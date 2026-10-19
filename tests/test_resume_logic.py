from __future__ import annotations

from resume_logic import unresolved_failure_unit_keys


def test_unresolved_failure_unit_keys_excludes_already_succeeded_units() -> None:
    failures = [
        {"unit_key": "u1"},
        {"unit_key": "u2"},
        {"unit_key": "u2"},
    ]
    successes = [
        {"unit_key": "u1"},
    ]
    unresolved = unresolved_failure_unit_keys(failures, successes)
    assert unresolved == ["u2"]
