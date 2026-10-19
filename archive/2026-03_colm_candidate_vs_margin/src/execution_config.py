from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dataset_loader import CanonicalOption
from io_utils import read_yaml


def _stable_hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class PromptTemplateSpec:
    template_id: str
    prompt_mode: str
    answer_prefix: str
    system_prompt: str | None

    def config_hash(self) -> str:
        return _stable_hash(asdict(self))


@dataclass(frozen=True)
class ReadoutSpec:
    id: str
    kind: str
    normalization: str
    primary: bool
    template: str | None = None

    def config_hash(self) -> str:
        return _stable_hash(asdict(self))


@dataclass(frozen=True)
class PermutationSpec:
    name: str
    strategy: str

    def config_hash(self) -> str:
        return _stable_hash(asdict(self))


@dataclass(frozen=True)
class ControlsConfig:
    permutations: dict[str, PermutationSpec]
    paired_comparison_required: bool


def load_prompt_templates(config_path: Path) -> dict[str, PromptTemplateSpec]:
    cfg = read_yaml(config_path)
    templates = {}
    for name, payload in (cfg.get("templates") or {}).items():
        templates[str(name)] = PromptTemplateSpec(
            template_id=str(payload["template_id"]),
            prompt_mode=str(payload["prompt_mode"]),
            answer_prefix=str(payload["answer_prefix"]),
            system_prompt=(None if payload.get("system_prompt") in [None, ""] else str(payload.get("system_prompt"))),
        )
    return templates


def load_readout_specs(config_path: Path) -> dict[str, ReadoutSpec]:
    cfg = read_yaml(config_path)
    out = {}
    for payload in cfg.get("readouts", []):
        spec = ReadoutSpec(
            id=str(payload["id"]),
            kind=str(payload["kind"]),
            normalization=str(payload["normalization"]),
            primary=bool(payload.get("primary", False)),
            template=(None if payload.get("template") in [None, ""] else str(payload.get("template"))),
        )
        out[spec.id] = spec
    return out


def load_controls_config(config_path: Path) -> ControlsConfig:
    cfg = read_yaml(config_path)
    controls = cfg.get("controls") or {}
    permutations: dict[str, PermutationSpec] = {}
    raw_permutations = controls.get("permutations") or {}
    for name, payload in raw_permutations.items():
        if isinstance(payload, dict):
            strategy = str(payload["strategy"])
        else:
            strategy = str(payload)
        permutations[str(name)] = PermutationSpec(name=str(name), strategy=strategy)
    return ControlsConfig(
        permutations=permutations,
        paired_comparison_required=bool(controls.get("paired_comparison_required", True)),
    )


def resolve_prompt_template_for_mode(templates: dict[str, PromptTemplateSpec], *, prompt_mode: str) -> PromptTemplateSpec:
    matches = [template for template in templates.values() if template.prompt_mode == prompt_mode]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one prompt template for prompt_mode={prompt_mode}, found {len(matches)}")
    return matches[0]


def resolve_readout_specs(readout_names: list[str], readout_specs: dict[str, ReadoutSpec]) -> list[ReadoutSpec]:
    resolved = []
    for name in readout_names:
        spec = readout_specs.get(str(name))
        if spec is None:
            raise ValueError(f"unknown readout_id={name}")
        resolved.append(spec)
    return resolved


def resolve_permutation_specs(permutation_names: list[str], controls_cfg: ControlsConfig) -> list[PermutationSpec]:
    resolved = []
    for name in permutation_names:
        spec = controls_cfg.permutations.get(str(name))
        if spec is None:
            raise ValueError(f"unknown permutation={name}")
        resolved.append(spec)
    return resolved


def build_option_surfaces(readout_spec: ReadoutSpec, options: list[CanonicalOption]) -> dict[str, str]:
    if readout_spec.kind == "semantic_option_text":
        return {option.option_id: option.raw_text for option in options}
    if readout_spec.kind == "templated_semantic":
        if readout_spec.template is None:
            raise ValueError(f"templated readout {readout_spec.id} requires a template")
        return {
            option.option_id: readout_spec.template.format(option_text=option.raw_text, label=option.option_id)
            for option in options
        }
    if readout_spec.kind == "letter_label":
        template = readout_spec.template or "{label}"
        return {
            option.option_id: template.format(option_text=option.raw_text, label=option.option_id)
            for option in options
        }
    raise ValueError(f"unsupported readout kind={readout_spec.kind}")


def permutation_for_spec(spec: PermutationSpec, *, option_count: int) -> list[int]:
    if spec.strategy == "identity":
        return list(range(option_count))
    if spec.strategy == "reverse":
        return list(reversed(range(option_count)))
    raise ValueError(f"unsupported permutation strategy={spec.strategy}")
