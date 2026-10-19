from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from io_utils import read_yaml


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    size: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    prompt_mode: str
    enabled: bool
    provisional: bool
    trust_remote_code: bool
    family_ready: bool
    local_files_only: bool
    hidden_state_projection_mode: str
    system_prompt_handling: str


def load_model_specs(config_path: Path) -> list[ModelSpec]:
    cfg = read_yaml(config_path)
    specs = []
    for item in cfg.get("models", []):
        specs.append(
            ModelSpec(
                name=str(item["name"]),
                family=str(item["family"]),
                size=str(item["size"]),
                model_id=str(item["model_id"]),
                model_revision=str(item["model_revision"]),
                tokenizer_revision=str(item.get("tokenizer_revision") or item["model_revision"]),
                prompt_mode=str(item["prompt_mode"]),
                enabled=bool(item.get("enabled", True)),
                provisional=bool(item.get("provisional", False)),
                trust_remote_code=bool(item.get("trust_remote_code", False)),
                family_ready=bool(item.get("family_ready", False)),
                local_files_only=bool(item.get("local_files_only", False)),
                hidden_state_projection_mode=str(item.get("hidden_state_projection_mode", "apply_final_norm_except_last")),
                system_prompt_handling=str(item.get("system_prompt_handling", "system_role")),
            )
        )
    return specs


def resolve_enabled_models(specs: list[ModelSpec], *, require_ready: bool = True) -> list[ModelSpec]:
    out = []
    for spec in specs:
        if not spec.enabled:
            continue
        if require_ready and not spec.family_ready:
            continue
        out.append(spec)
    return out
