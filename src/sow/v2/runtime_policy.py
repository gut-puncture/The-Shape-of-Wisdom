from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeEstimate:
    task_name: str
    rows_per_second: float
    prompts_per_model: int
    model_count: int
    estimated_hours_all_models: float


def estimate_runtime(*, task_name: str, rows_per_second: float, prompts_per_model: int, model_count: int = 3) -> RuntimeEstimate:
    rps = max(float(rows_per_second), 1e-9)
    total_rows = int(prompts_per_model) * int(model_count)
    total_seconds = float(total_rows) / rps
    return RuntimeEstimate(
        task_name=str(task_name),
        rows_per_second=rps,
        prompts_per_model=int(prompts_per_model),
        model_count=int(model_count),
        estimated_hours_all_models=float(total_seconds / 3600.0),
    )


def choose_backend(*, estimated_hours_all_models: float, threshold_hours: float = 36.0) -> str:
    return "gpu" if float(estimated_hours_all_models) > float(threshold_hours) else "mac"
