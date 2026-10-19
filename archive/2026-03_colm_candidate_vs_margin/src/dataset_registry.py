from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from io_utils import read_yaml


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_id: str
    config_name: str | None
    revision: str
    split: str
    enabled: bool
    smoke_examples: int
    note: str


def load_dataset_specs(config_path: Path) -> list[DatasetSpec]:
    cfg = read_yaml(config_path)
    specs = []
    for item in cfg.get("datasets", []):
        specs.append(
            DatasetSpec(
                name=str(item["name"]),
                dataset_id=str(item["dataset_id"]),
                config_name=(None if item.get("config_name") in [None, ""] else str(item["config_name"])),
                revision=str(item["revision"]),
                split=str(item["split"]),
                enabled=bool(item.get("enabled", True)),
                smoke_examples=int(item.get("smoke_examples", 1)),
                note=str(item.get("note", "")),
            )
        )
    return specs
