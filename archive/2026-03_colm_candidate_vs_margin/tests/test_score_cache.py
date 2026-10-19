from __future__ import annotations

from pathlib import Path

from path_manager import ExperimentPaths
from score_cache import ScoreCache, ScoreCacheKey


def _make_key(**overrides: str) -> ScoreCacheKey:
    payload = {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "tokenizer_revision": "tokrev",
        "model_revision": "modrev",
        "dataset_revision": "datarev",
        "prompt_template_hash": "prompt_hash",
        "readout_id": "semantic_exact",
        "permutation_id": "identity",
        "scoring_config_hash": "score_hash",
        "unit_id": "dataset/example_001",
        "scorer_engine": "reference_exact",
        "scorer_feature_hash": "features_v1",
        "dtype": "float32",
        "tokenization_version": "contextual_v1",
    }
    payload.update(overrides)
    return ScoreCacheKey(**payload)


def test_cache_path_changes_when_key_changes(tmp_path: Path) -> None:
    workspace = tmp_path / "v2" / "semantic_candidate_vs_winner_isolation"
    paths = ExperimentPaths.from_workspace_root(workspace)
    paths.ensure_required_dirs()
    cache = ScoreCache(paths)
    key_a = _make_key()
    key_b = _make_key(prompt_template_hash="different")
    assert cache.cache_path_for(key_a) != cache.cache_path_for(key_b)


def test_cache_path_changes_when_scorer_engine_changes(tmp_path: Path) -> None:
    workspace = tmp_path / "v2" / "semantic_candidate_vs_winner_isolation"
    paths = ExperimentPaths.from_workspace_root(workspace)
    paths.ensure_required_dirs()
    cache = ScoreCache(paths)
    key_a = _make_key(scorer_engine="reference_exact")
    key_b = _make_key(scorer_engine="optimized_exact")
    assert cache.cache_path_for(key_a) != cache.cache_path_for(key_b)


def test_cache_save_and_load_bundle(tmp_path: Path) -> None:
    workspace = tmp_path / "v2" / "semantic_candidate_vs_winner_isolation"
    paths = ExperimentPaths.from_workspace_root(workspace)
    paths.ensure_required_dirs()
    cache = ScoreCache(paths)
    key = _make_key()
    bundle = {"final_winner": "A", "layers": [0, 1]}
    cache.save_bundle(key, bundle)
    assert cache.exists(key)
    assert cache.load_bundle(key) == bundle
