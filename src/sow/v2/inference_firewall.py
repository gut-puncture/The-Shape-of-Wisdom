from __future__ import annotations

import os


def assert_inference_allowed(stage_name: str) -> None:
    """Raise unless explicit opt-in for inference is present.

    Contract:
      - inference is allowed only when `SOW_ALLOW_INFERENCE=1`
      - any other value (or unset) fails closed with an actionable message
    """

    flag = str(os.environ.get("SOW_ALLOW_INFERENCE", "")).strip()
    if flag == "1":
        return
    stage = str(stage_name or "unknown_stage")
    got = flag if flag else "<unset>"
    raise RuntimeError(
        "Inference firewall blocked execution for "
        f"{stage}: expected SOW_ALLOW_INFERENCE=1, got {got}. "
        "This command appears to load a model/tokenizer or run forward passes. "
        "If you intentionally want inference, re-run with SOW_ALLOW_INFERENCE=1."
    )

