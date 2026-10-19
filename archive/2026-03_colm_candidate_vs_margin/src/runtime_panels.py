from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, TypeVar

from io_utils import read_yaml


T = TypeVar("T")


def load_runtime_config(config_path: Path) -> dict[str, Any]:
    cfg = read_yaml(config_path)
    runtime_cfg = cfg.get("runtime")
    if not isinstance(runtime_cfg, dict):
        raise ValueError(f"runtime config missing at {config_path}")
    return runtime_cfg


def resolve_panel_name(runtime_cfg: dict[str, Any], *, smoke: bool, explicit_panel: str | None) -> str:
    if smoke:
        if explicit_panel is not None and explicit_panel != "smoke":
            raise ValueError("cannot combine --smoke with a non-smoke panel override")
        return "smoke"
    return str(explicit_panel or runtime_cfg.get("default_panel") or "paper")


def get_panel_config(runtime_cfg: dict[str, Any], panel_name: str) -> dict[str, Any]:
    panel_cfg = runtime_cfg.get(panel_name)
    if not isinstance(panel_cfg, dict):
        raise ValueError(f"unknown runtime panel={panel_name}")
    return panel_cfg


def resolve_named_specs(
    specs: Iterable[T],
    names: Iterable[str],
    *,
    require_enabled: bool = True,
    require_family_ready: bool = False,
    label: str,
) -> list[T]:
    spec_by_name = {str(getattr(spec, "name")): spec for spec in specs}
    missing: list[str] = []
    disabled: list[str] = []
    unready: list[str] = []
    resolved: list[T] = []
    for name in names:
        spec = spec_by_name.get(str(name))
        if spec is None:
            missing.append(str(name))
            continue
        if require_enabled and not bool(getattr(spec, "enabled", True)):
            disabled.append(str(name))
            continue
        if require_family_ready and not bool(getattr(spec, "family_ready", False)):
            unready.append(str(name))
            continue
        resolved.append(spec)
    problems = []
    if missing:
        problems.append(f"missing={missing}")
    if disabled:
        problems.append(f"disabled={disabled}")
    if unready:
        problems.append(f"not_family_ready={unready}")
    if problems:
        joined = " ".join(problems)
        raise ValueError(f"{label} panel resolution failed: {joined}")
    return resolved
