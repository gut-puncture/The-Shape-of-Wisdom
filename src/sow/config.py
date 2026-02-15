from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

from sow.constants import EXPECTED_ROBUSTNESS_WRAPPER_IDS_V2


def default_run_config(*, run_id: str, random_seed: int) -> Dict[str, Any]:
    import platform

    is_macos = platform.system() == "Darwin"
    return {
        "run_id": run_id,
        "random_seed": int(random_seed),
        "models": [
            {
                "name": "qwen2.5-7b-instruct",
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
                "dtype": "float16",
                "device": "auto",
            },
            {
                "name": "llama-3.1-8b-instruct",
                "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
                "dtype": "float16",
                "device": "auto",
            },
            {
                "name": "mistral-7b-instruct-v0.3",
                "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
                "revision": "c170c708c41dac9275d15a8fff4eca08d52bab71",
                "dtype": "float16",
                "device": "auto",
            },
        ],
        "generation": {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_new_tokens": 24,
        },
        # Operational safety for local Apple Silicon runs.
        # Uses macOS "thermal pressure" signals (via powermetrics) to trigger cool-downs.
        "thermal_hygiene": {
            "enabled": bool(is_macos),
            "provider": "powermetrics_thermal_pressure",
            # Approximate "start cooling around ~80-90C" by triggering before "Critical".
            "cutoff_level": "serious",
            "cooldown_seconds": 20 * 60,
            "check_interval_seconds": 30,
        },
        "prompting": {
            "baseline_wrapper_id": "plain_exam",
            "robustness_wrapper_ids_v2": list(EXPECTED_ROBUSTNESS_WRAPPER_IDS_V2),
            "required_answer_suffix": "Answer: ",
        },
        "pca": {
            "sample_size": 1000,
            "n_components": 128,
            "sampling_policy": "stratified(wrapper_id, coarse_domain) uniform-over-strata",
        },
        "inference": {
            # Commitment is computed for multiple fixed margin thresholds (prob top1 - prob top2).
            # These are scientific decisions and must be frozen for the run.
            "commitment_margin_thresholds": [0.05, 0.1, 0.2],
            # Batch-consistency gate tolerance for candidate logits (float16 -> float32).
            "batch_consistency_atol_candidate_logits": 1e-3,
        },
        "resume_key": {
            "definition": "sha256(model_id + ':' + prompt_uid)",
        },
    }


def write_yaml(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(obj, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )


def read_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_run_config(cfg: Dict[str, Any]) -> None:
    if not isinstance(cfg, dict):
        raise ValueError("run_config must be a mapping")
    for k in ["run_id", "random_seed", "models", "generation", "prompting", "pca", "resume_key"]:
        if k not in cfg:
            raise ValueError(f"run_config missing required key: {k}")
    if not isinstance(cfg["models"], list) or len(cfg["models"]) != 3:
        raise ValueError("run_config.models must be a list of 3 models")
    for m in cfg["models"]:
        for k in ["name", "model_id", "revision", "dtype", "device"]:
            if k not in m:
                raise ValueError(f"model entry missing key: {k}")
    gen = cfg["generation"]
    if gen.get("do_sample") is not False:
        raise ValueError("Milestone 1 requires do_sample=false (greedy decoding) for determinism")
    if int(gen.get("max_new_tokens", -1)) != 24:
        raise ValueError("Milestone 1 requires max_new_tokens=24")
    wrappers = cfg["prompting"].get("robustness_wrapper_ids_v2")
    if wrappers != EXPECTED_ROBUSTNESS_WRAPPER_IDS_V2:
        raise ValueError("robustness wrapper list must match the expected v2 wrapper_id list exactly")

    # Optional: inference gates/metrics settings (Stage 13/14).
    inf = cfg.get("inference")
    if inf is not None:
        if not isinstance(inf, dict):
            raise ValueError("inference must be a mapping")
        thrs = inf.get("commitment_margin_thresholds")
        if thrs is not None:
            if not isinstance(thrs, list) or (not thrs):
                raise ValueError("inference.commitment_margin_thresholds must be a non-empty list")
            vals = [float(x) for x in thrs]
            if any((x <= 0.0 or x >= 1.0) for x in vals):
                raise ValueError("inference.commitment_margin_thresholds must be in (0,1)")
        atol = inf.get("batch_consistency_atol_candidate_logits")
        if atol is not None and float(atol) <= 0.0:
            raise ValueError("inference.batch_consistency_atol_candidate_logits must be positive")

    # Optional (back-compat): thermal hygiene block.
    th = cfg.get("thermal_hygiene")
    if th is not None:
        if not isinstance(th, dict):
            raise ValueError("thermal_hygiene must be a mapping")
        provider = str(th.get("provider", "powermetrics_thermal_pressure"))
        if provider != "powermetrics_thermal_pressure":
            raise ValueError("thermal_hygiene.provider must be powermetrics_thermal_pressure")
        cutoff = str(th.get("cutoff_level", "serious")).strip().lower()
        if cutoff not in ("nominal", "fair", "serious", "critical"):
            raise ValueError("thermal_hygiene.cutoff_level must be one of: nominal/fair/serious/critical")
        cooldown_seconds = int(th.get("cooldown_seconds", 20 * 60))
        if cooldown_seconds <= 0:
            raise ValueError("thermal_hygiene.cooldown_seconds must be positive")
        interval_seconds = int(th.get("check_interval_seconds", 30))
        if interval_seconds <= 0:
            raise ValueError("thermal_hygiene.check_interval_seconds must be positive")
