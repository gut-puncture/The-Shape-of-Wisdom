from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_required_cached_artifacts_present_and_well_formed():
    required = [
        "decision_metrics.parquet",
        "prompt_types.parquet",
        "tracing_scalars.parquet",
        "span_deletion_causal.parquet",
        "negative_controls.parquet",
        "patching_results.parquet",
    ]
    for name in required:
        path = ROOT / "data" / "cached" / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
        assert not pd.read_parquet(path).empty, name


def test_core_counts_and_layer_completeness():
    decision = pd.read_parquet(ROOT / "data" / "cached" / "decision_metrics.parquet")
    prompt_types = pd.read_parquet(ROOT / "data" / "cached" / "prompt_types.parquet")
    tracing = pd.read_parquet(ROOT / "data" / "cached" / "tracing_scalars.parquet")
    assert prompt_types.shape[0] == 9000
    assert decision.shape[0] == 276000
    assert tracing.groupby("model_id")["prompt_uid"].nunique().to_dict()
    assert prompt_types.duplicated(["model_id", "prompt_uid"]).sum() == 0
    assert decision.duplicated(["model_id", "prompt_uid", "layer_index"]).sum() == 0
    for _, group in decision.groupby(["model_id", "prompt_uid"]):
        layers = group["layer_index"].tolist()
        assert layers == sorted(layers)


def test_audit_pass_and_claim_matrix_present():
    audit = json.loads((ROOT / "data" / "audit" / "artifact_integrity.json").read_text())
    assert audit["pass"] is True
    matrix = (ROOT / "docs" / "claim_evidence_matrix.md").read_text()
    for phrase in [
        "depthwise trajectories",
        "Attention/MLP",
        "Span deletion",
        "protocol-sensitive",
    ]:
        assert phrase in matrix


def test_forbidden_scope_absent_from_paper_facing_text():
    forbidden = [
        "ARC-Challenge",
        "CommonsenseQA",
        "mmlu_abstract_algebra",
        "Abstract Algebra",
        "nine-model",
        "9-model",
        "tiny dataset",
    ]
    text_paths = list((ROOT / "paper").rglob("*.tex"))
    text_paths += [ROOT / "README.md", ROOT / "REPRODUCE.md"]
    text_paths += list((ROOT / "docs").rglob("*.md"))
    for path in text_paths:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{token} in {path}"


def test_figures_exist_and_are_nonempty():
    figures = sorted((ROOT / "paper" / "figures").glob("figure*.pdf"))
    assert len(figures) >= 5
    for fig in figures:
        assert fig.stat().st_size > 10_000, fig


def test_arxiv_source_bundle_is_clean():
    source = ROOT / "arxiv_source"
    for name in ["main.tex", "references.bib", "main.bbl"]:
        path = source / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    for name in ["main.aux", "main.blg", "main.log", "main.out", "main.pdf"]:
        assert not (source / name).exists(), name
