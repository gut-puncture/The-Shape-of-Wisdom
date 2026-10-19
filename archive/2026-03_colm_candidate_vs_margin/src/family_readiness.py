from __future__ import annotations

import hashlib
from typing import Any

from gpu_preflight import verify_prompt_template_compatibility
from layerwise_extractor import _find_final_norm, assert_final_layer_equivalence
from model_loader import load_model_bundle


def _check_metadata(spec) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi  # noqa: PLC0415

        info = HfApi().model_info(spec.model_id, revision=spec.model_revision)
        return {
            "passed": True,
            "gated": getattr(info, "gated", None),
            "sha": getattr(info, "sha", None),
            "private": getattr(info, "private", None),
        }
    except Exception as exc:
        return {"passed": False, "error": str(exc)}


def _check_tokenizer(spec) -> tuple[dict[str, Any], Any | None]:
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415

        tokenizer = AutoTokenizer.from_pretrained(
            spec.model_id,
            revision=spec.tokenizer_revision,
            trust_remote_code=spec.trust_remote_code,
            local_files_only=spec.local_files_only,
        )
        chat_template = getattr(tokenizer, "chat_template", None)
        if spec.prompt_mode == "chat" and not chat_template:
            return ({"passed": False, "error": "chat_template_missing"}, None)
        template_hash = (
            hashlib.sha256(str(chat_template).encode("utf-8")).hexdigest()
            if chat_template is not None
            else None
        )
        return (
            {
                "passed": True,
                "chat_template_present": bool(chat_template),
                "chat_template_hash": template_hash,
                "vocab_size": getattr(tokenizer, "vocab_size", None),
            },
            tokenizer,
        )
    except Exception as exc:
        return ({"passed": False, "error": str(exc)}, None)


def _check_prompt_compatibility(spec, *, tokenizer: Any | None, prompt_template: Any | None) -> dict[str, Any]:
    if tokenizer is None:
        return {"passed": False, "error": "tokenizer_unavailable"}
    if prompt_template is None:
        return {"passed": False, "error": "prompt_template_unavailable"}
    sanity_options = ["A. one", "B. two", "C. three"]
    return verify_prompt_template_compatibility(
        tokenizer=tokenizer,
        prompt_mode=spec.prompt_mode,
        question_text="Which option is correct?",
        options=sanity_options,
        answer_prefix=prompt_template.answer_prefix,
        system_prompt=prompt_template.system_prompt,
        system_prompt_handling=spec.system_prompt_handling,
    )


def _check_model_load_and_equivalence(spec, *, device: str, dtype: str) -> dict[str, Any]:
    try:
        bundle = load_model_bundle(spec, device=device, dtype=dtype)
    except Exception as exc:
        return {"passed": False, "error": str(exc), "stage": "model_load"}
    try:
        tokenizer = bundle.tokenizer
        prompt_ids = tokenizer.encode("Sanity check.\nFinal answer:", add_special_tokens=False)
        if not prompt_ids:
            raise ValueError("sanity prompt tokenized to zero length")
        import torch  # noqa: PLC0415

        with torch.no_grad():
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=bundle.device)
            outputs = bundle.model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        assert_final_layer_equivalence(
            outputs=outputs,
            output_embedding=bundle.model.get_output_embeddings(),
            final_norm=_find_final_norm(bundle.model),
            projection_mode=bundle.hidden_state_projection_mode,
            model_config=getattr(bundle.model, "config", None),
        )
        return {
            "passed": True,
            "device": bundle.device,
            "dtype": bundle.dtype,
            "projection_mode": bundle.hidden_state_projection_mode,
            "stage": "equivalence",
        }
    except Exception as exc:
        return {
            "passed": False,
            "error": str(exc),
            "device": bundle.device,
            "dtype": bundle.dtype,
            "projection_mode": bundle.hidden_state_projection_mode,
            "stage": "equivalence",
        }


def check_model_readiness(spec, *, device: str = "cpu", dtype: str = "auto", prompt_template: Any | None = None) -> dict[str, Any]:
    metadata = _check_metadata(spec)
    tokenizer_check, tokenizer = _check_tokenizer(spec)
    prompt_check = _check_prompt_compatibility(spec, tokenizer=tokenizer, prompt_template=prompt_template)
    model_check = _check_model_load_and_equivalence(spec, device=device, dtype=dtype)
    checks = {
        "metadata_access": metadata,
        "tokenizer_audit": tokenizer_check,
        "prompt_compatibility": prompt_check,
        "model_load_and_equivalence": model_check,
    }
    return {
        "model_name": spec.name,
        "model_family": spec.family,
        "model_size": spec.size,
        "model_id": spec.model_id,
        "model_revision": spec.model_revision,
        "tokenizer_revision": spec.tokenizer_revision,
        "checks": checks,
        "overall_passed": all(bool(check.get("passed")) for check in checks.values()),
    }


def summarize_family_readiness(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault(str(row["model_family"]), []).append(row)
    summary = []
    for family, rows in sorted(grouped.items()):
        passed_rows = [row for row in rows if row.get("overall_passed") is True]
        summary.append(
            {
                "model_family": family,
                "models_checked": sorted(str(row["model_name"]) for row in rows),
                "models_passed": sorted(str(row["model_name"]) for row in passed_rows),
                "sizes_passed": sorted(str(row["model_size"]) for row in passed_rows),
                "passed_model_count": len(passed_rows),
                "at_least_two_sizes_passed": len({str(row["model_size"]) for row in passed_rows}) >= 2,
                "all_checked_models_passed": len(passed_rows) == len(rows),
            }
        )
    return summary


def timed_out_readiness_result(spec, *, timeout_seconds: int) -> dict[str, Any]:
    return {
        "model_name": spec.name,
        "model_family": spec.family,
        "model_size": spec.size,
        "model_id": spec.model_id,
        "model_revision": spec.model_revision,
        "tokenizer_revision": spec.tokenizer_revision,
        "checks": {
            "metadata_access": {"passed": False, "error": "timed_out_before_completion"},
            "tokenizer_audit": {"passed": False, "error": "timed_out_before_completion"},
            "prompt_compatibility": {"passed": False, "error": "timed_out_before_completion"},
            "model_load_and_equivalence": {
                "passed": False,
                "error": f"timed_out_after_{timeout_seconds}s",
                "stage": "timeout",
            },
        },
        "overall_passed": False,
        "timed_out": True,
    }
