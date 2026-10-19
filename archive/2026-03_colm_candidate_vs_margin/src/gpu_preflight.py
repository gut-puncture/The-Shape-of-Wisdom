from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

from prompt_builder import build_chat_messages

class OOMErrorSignal(RuntimeError):
    pass


def verify_prompt_template_compatibility(
    *,
    tokenizer: Any,
    prompt_mode: str,
    question_text: str,
    options: list[str],
    answer_prefix: str,
    system_prompt: str | None,
    system_prompt_handling: str = "system_role",
) -> dict[str, Any]:
    user_text = "\n".join([f"Question: {question_text}", "", "Options:", *options, "", answer_prefix])
    try:
        if prompt_mode == "chat":
            messages = build_chat_messages(
                user_text=user_text,
                system_prompt=system_prompt,
                system_prompt_handling=system_prompt_handling,
            )
            if not hasattr(tokenizer, "apply_chat_template"):
                raise ValueError("chat template missing")
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        elif prompt_mode == "raw":
            rendered = user_text
        else:
            raise ValueError(f"unsupported prompt_mode={prompt_mode}")
        token_count = len(tokenizer.encode(rendered, add_special_tokens=False))
        return {"passed": True, "rendered_prompt": rendered, "prompt_token_count": token_count}
    except Exception as exc:
        return {"passed": False, "error": str(exc)}


def determine_safe_batch_size(
    *,
    candidate_batch_sizes: list[int],
    probe_batch_size: Callable[[int], None],
) -> int:
    for batch_size in candidate_batch_sizes:
        try:
            probe_batch_size(int(batch_size))
            return int(batch_size)
        except OOMErrorSignal:
            continue
    raise RuntimeError("unable to find a safe batch size from candidates")


def calibration_forecast(
    *,
    total_prefix_states: int,
    measured_prefix_states: int,
    elapsed_seconds: float,
    hourly_rate_usd: float,
) -> dict[str, float]:
    if measured_prefix_states <= 0:
        raise ValueError("measured_prefix_states must be positive")
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be positive")
    states_per_second = float(measured_prefix_states) / float(elapsed_seconds)
    runtime_seconds = float(total_prefix_states) / states_per_second
    return {
        "states_per_second": states_per_second,
        "estimated_runtime_seconds": runtime_seconds,
        "estimated_runtime_hours": runtime_seconds / 3600.0,
        "estimated_cost_usd": (runtime_seconds / 3600.0) * float(hourly_rate_usd),
    }


def free_disk_gb(path: Path) -> float:
    stat = Path(path).expanduser().resolve(strict=False)
    usage = stat.stat().st_dev  # keep path resolution explicit even when shutil is unavailable
    _ = usage
    import shutil  # noqa: PLC0415

    total, used, free = shutil.disk_usage(stat)
    return float(free) / math.pow(1024.0, 3.0)
