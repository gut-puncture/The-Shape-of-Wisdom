from __future__ import annotations

from pathlib import Path

from shape_of_wisdom_release.build_assets import DERIVED_PATH, FORBIDDEN_TEXT_PATTERNS, MECHANISTIC_PATH, load_jsonl
from shape_of_wisdom_release.verify_derived import verify_derived


ROOT = Path(__file__).resolve().parents[1]


def test_included_derived_data_is_clean_and_complete() -> None:
    summary = verify_derived()
    assert summary["rows"] == 8901
    assert summary["mechanistic_rows"] >= 400


def test_derived_data_has_no_absolute_cache_paths() -> None:
    rows = load_jsonl(DERIVED_PATH)
    assert rows
    assert all("cache_relpath" not in row for row in rows)
    mech_rows = load_jsonl(MECHANISTIC_PATH)
    assert mech_rows


def test_paper_source_avoids_stale_or_internal_language() -> None:
    text = "\n".join(path.read_text() for path in (ROOT / "paper").rglob("*.tex"))
    for pattern in FORBIDDEN_TEXT_PATTERNS:
        assert pattern.lower().replace("\\b", "") not in text.lower()


def test_tiny_dataset_is_absent_from_paper_facing_artifacts() -> None:
    checked_roots = [
        ROOT / "paper",
        ROOT / "docs",
        ROOT / "data" / "tables",
    ]
    checked_files = [ROOT / "README.md", ROOT / "REPRODUCE.md", ROOT / "LICENSES_AND_SOURCES.md"]
    checked_files.extend(path for base in checked_roots for path in base.rglob("*") if path.is_file())
    forbidden = ["mmlu_abstract_algebra", "MMLU Abstract Algebra", "Abstract Algebra"]
    for path in checked_files:
        text = path.read_text(errors="ignore")
        for needle in forbidden:
            assert needle not in text, f"{needle} leaked into {path.relative_to(ROOT)}"


def test_plot_legends_are_kept_outside_axes() -> None:
    source = (ROOT / "src" / "shape_of_wisdom_release" / "build_assets.py").read_text()
    assert 'legend(loc="lower right"' not in source
    assert 'legend(loc="upper right"' not in source
    assert 'legend(loc="best"' not in source

    legend_blocks = source.split(".legend(")[1:]
    assert legend_blocks
    for block in legend_blocks:
        call = block.split(")", 1)[0]
        assert "bbox_to_anchor" in call
