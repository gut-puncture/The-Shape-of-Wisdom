from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from io_utils import read_json, write_json_atomic
from path_manager import ExperimentPaths


def _stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ScoreCacheKey:
    model_id: str
    tokenizer_revision: str
    model_revision: str
    dataset_revision: str
    prompt_template_hash: str
    readout_id: str
    permutation_id: str
    scoring_config_hash: str
    unit_id: str
    prompt_render_hash: str = ""
    prompt_config_hash: str = ""
    readout_config_hash: str = ""
    permutation_config_hash: str = ""
    scorer_engine: str = "reference_exact"
    scorer_feature_hash: str = ""
    dtype: str = "float32"
    tokenization_version: str = "contextual_v1"

    def digest(self) -> str:
        return _stable_hash(asdict(self))


class ScoreCache:
    def __init__(self, paths: ExperimentPaths, namespace: str = "semantic_scores") -> None:
        self.paths = paths
        self.namespace = namespace
        self.root = self.paths.resolve_under_workspace("outputs", "caches", namespace)
        self.root.mkdir(parents=True, exist_ok=True)

    def cache_path_for(self, key: ScoreCacheKey) -> Path:
        digest = key.digest()
        return self.root / digest[:2] / f"{digest}.json"

    def exists(self, key: ScoreCacheKey) -> bool:
        return self.cache_path_for(key).exists()

    def save_bundle(self, key: ScoreCacheKey, bundle: dict[str, Any]) -> Path:
        path = self.cache_path_for(key)
        payload = {"cache_key": asdict(key), "bundle": bundle}
        write_json_atomic(path, payload)
        return path

    def load_bundle(self, key: ScoreCacheKey) -> dict[str, Any]:
        payload = read_json(self.cache_path_for(key))
        return dict(payload["bundle"])
