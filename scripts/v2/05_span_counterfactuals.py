#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from _common import base_parser, baseline_manifest_path, load_experiment_config, resolve_models, run_v2_root_for, write_json, write_parquet
from sow.io_jsonl import iter_jsonl
from sow.thermal.thermal_governor import ThermalGovernor, ThermalHygieneConfig
from sow.v2.model_nuances import apply_tokenizer_nuance, pick_torch_dtype
from sow.v2.span_counterfactuals import completed_span_keys_for_mode, compute_span_effect, delete_span, label_span_effects
from sow.v2.span_parser import parse_prompt_spans


def _tail_decision_info(decision_metrics: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, str | float]]:
    if decision_metrics.empty:
        return {}
    tail = decision_metrics.sort_values("layer_index").groupby(["model_id", "prompt_uid"], as_index=False).tail(1)
    out: Dict[Tuple[str, str], Dict[str, str | float]] = {}
    for _, row in tail.iterrows():
        model_id = str(row["model_id"])
        prompt_uid = str(row["prompt_uid"])
        out[(model_id, prompt_uid)] = {
            "full_delta": float(row.get("delta", 0.0)),
            "competitor": str(row.get("competitor") or "B"),
            "correct_key": str(row.get("correct_key") or "A").strip().upper(),
        }
    return out


def _proxy_mutated_delta(*, full_delta: float, span_role: str, span_len: int, prompt_len: int, correct_key: str) -> float:
    ratio = float(span_len) / max(1.0, float(prompt_len))
    if span_role == "question_stem":
        effect = 0.30 * ratio + 0.02
    elif span_role.startswith("option_") and span_role.endswith(str(correct_key)):
        effect = 0.35 * ratio + 0.02
    elif span_role.startswith("option_"):
        effect = -0.20 * ratio
    elif span_role == "instruction":
        effect = 0.05 * ratio
    else:
        effect = -0.03 * ratio
    return float(full_delta - effect)


def _position_ids(attn):
    p = attn.long().cumsum(dim=-1) - 1
    return p.masked_fill(attn == 0, 0)


def _choice_token_ids(tokenizer) -> Dict[str, int | None]:
    out: Dict[str, int | None] = {}
    for key in ["A", "B", "C", "D"]:
        ids = tokenizer.encode(key, add_special_tokens=False)
        out[key] = int(ids[0]) if ids else None
    return out


def _model_mutated_delta(
    *,
    model_obj,
    tokenizer,
    prompt_text: str,
    correct_key: str,
    competitor_key: str,
    choice_token_ids: Dict[str, int | None],
    device: str,
) -> float:
    import torch  # noqa: PLC0415

    ck = str(correct_key).strip().upper()
    kp = str(competitor_key).strip().upper()

    enc = tokenizer(prompt_text, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"]
    attn = enc.get("attention_mask")
    if attn is None:
        raise RuntimeError("tokenizer missing attention_mask")
    if device != "cpu":
        input_ids = input_ids.to(device)
        attn = attn.to(device)

    with torch.inference_mode():
        out = model_obj(
            input_ids=input_ids,
            attention_mask=attn,
            position_ids=_position_ids(attn),
            use_cache=False,
            return_dict=True,
        )
        logits = out.logits[0, -1, :]

    c_tid = choice_token_ids.get(ck)
    k_tid = choice_token_ids.get(kp)
    if c_tid is None:
        return 0.0

    if k_tid is None:
        # Fallback: strongest non-correct option token.
        best_key = None
        best_val = None
        for key in ["A", "B", "C", "D"]:
            if key == ck:
                continue
            tid = choice_token_ids.get(key)
            if tid is None:
                continue
            val = float(logits[int(tid)].item())
            if best_val is None or val > float(best_val):
                best_val = val
                best_key = key
        if best_key is None:
            return 0.0
        k_tid = choice_token_ids.get(best_key)

    if k_tid is None:
        return 0.0
    return float(logits[int(c_tid)].item() - logits[int(k_tid)].item())


def _write_spans_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    ap = base_parser("V2: parse spans and compute counterfactual span effects")
    ap.add_argument("--counterfactual-mode", choices=["model", "proxy"], default=None)
    ap.add_argument("--allow-proxy-fallback", action="store_true")
    args = ap.parse_args()

    cfg = load_experiment_config(Path(args.config))
    models = resolve_models(cfg, model_name=args.model_name)

    span_cfg = cfg.get("span_counterfactual") or {}
    mode = str(args.counterfactual_mode or span_cfg.get("mode") or "proxy")
    allow_proxy_fallback = bool(args.allow_proxy_fallback or span_cfg.get("allow_proxy_fallback", False))

    out_root = run_v2_root_for(args.run_id)
    thermal_cfg_obj = ThermalHygieneConfig.from_cfg(cfg.get("thermal_policy"))
    thermal_events = out_root / "meta" / "thermal_events_span_counterfactuals.jsonl"
    governor = ThermalGovernor(cfg=thermal_cfg_obj, events_path=thermal_events) if thermal_cfg_obj.enabled else None

    decision_path = out_root / "decision_metrics.parquet"
    if not decision_path.exists():
        raise SystemExit(f"missing input: {decision_path}")
    decision = pd.read_parquet(decision_path)

    info_by_model_prompt = _tail_decision_info(decision)
    manifest = baseline_manifest_path(args.run_id)
    if not manifest.exists():
        raise SystemExit(f"missing baseline manifest: {manifest}")
    manifest_rows = {str(row.get("prompt_uid") or ""): row for row in iter_jsonl(manifest)}

    spans_path = out_root / "spans.jsonl"
    effects_path = out_root / "span_effects.parquet"

    existing_effects = pd.read_parquet(effects_path) if (args.resume and effects_path.exists()) else pd.DataFrame()
    done_keys = completed_span_keys_for_mode(existing_effects, mode=mode)

    new_spans_rows: List[Dict[str, object]] = []
    effects_rows: List[Dict[str, object]] = []

    max_prompts = int(args.max_prompts)
    prompts_processed = 0
    fallback_count = 0
    thermal_stop_action = None

    for model in models:
        model_id = str(model["model_id"])
        revision = str(model["revision"])

        model_prompt_uids = sorted(
            {
                prompt_uid
                for (mid, prompt_uid), _ in info_by_model_prompt.items()
                if str(mid) == model_id and prompt_uid in manifest_rows
            }
        )
        if max_prompts > 0:
            model_prompt_uids = model_prompt_uids[:max_prompts]

        model_obj = None
        tokenizer = None
        choice_token_ids: Dict[str, int | None] | None = None
        device = "cpu"

        if mode == "model":
            import torch  # noqa: PLC0415
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True, trust_remote_code=False)
            apply_tokenizer_nuance(tokenizer, model_id=model_id)
            model_obj = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype=pick_torch_dtype(device=device),
                low_cpu_mem_usage=True,
                trust_remote_code=False,
            )
            model_obj.eval()
            if device != "cpu":
                model_obj.to(device)
            choice_token_ids = _choice_token_ids(tokenizer)

        for prompt_uid in model_prompt_uids:
            if governor is not None:
                action = governor.maybe_cooldown(
                    stage="v2_span_counterfactuals",
                    model_id=model_id,
                    model_revision=revision,
                )
                if bool(action.get("checkpoint_exit")):
                    thermal_stop_action = {
                        "model_id": model_id,
                        "model_revision": revision,
                        "prompt_uid": prompt_uid,
                        "action": action,
                    }
                    break

            info = info_by_model_prompt.get((model_id, prompt_uid))
            if info is None:
                continue
            manifest_row = manifest_rows.get(prompt_uid)
            if not manifest_row:
                continue

            prompt_text = str(manifest_row.get("prompt_text") or "")
            correct_key = str(info.get("correct_key") or manifest_row.get("correct_key") or "A").strip().upper()
            competitor_key = str(info.get("competitor") or "B").strip().upper()
            full_delta = float(info.get("full_delta") or 0.0)

            spans = parse_prompt_spans(prompt_text)
            for span in spans:
                key = (model_id, prompt_uid, str(span.span_id))
                if key in done_keys:
                    continue

                mutated_text = delete_span(prompt_text, start_char=int(span.start_char), end_char=int(span.end_char))
                source = "proxy_v1"

                if mode == "model" and model_obj is not None and tokenizer is not None and choice_token_ids is not None:
                    try:
                        mutated_delta = _model_mutated_delta(
                            model_obj=model_obj,
                            tokenizer=tokenizer,
                            prompt_text=mutated_text,
                            correct_key=correct_key,
                            competitor_key=competitor_key,
                            choice_token_ids=choice_token_ids,
                            device=device,
                        )
                        source = "model_counterfactual"
                    except Exception:
                        if not allow_proxy_fallback:
                            raise
                        fallback_count += 1
                        mutated_delta = _proxy_mutated_delta(
                            full_delta=full_delta,
                            span_role=str(span.label),
                            span_len=int(span.end_char - span.start_char),
                            prompt_len=len(prompt_text),
                            correct_key=correct_key,
                        )
                else:
                    mutated_delta = _proxy_mutated_delta(
                        full_delta=full_delta,
                        span_role=str(span.label),
                        span_len=int(span.end_char - span.start_char),
                        prompt_len=len(prompt_text),
                        correct_key=correct_key,
                    )

                new_spans_rows.append(
                    {
                        "model_id": model_id,
                        "prompt_uid": prompt_uid,
                        "span_id": str(span.span_id),
                        "span_role": str(span.label),
                        "start_char": int(span.start_char),
                        "end_char": int(span.end_char),
                    }
                )

                effects_rows.append(
                    {
                        "model_id": model_id,
                        "prompt_uid": prompt_uid,
                        "correct_key": correct_key,
                        "competitor_key": competitor_key,
                        "span_id": str(span.span_id),
                        "span_role": str(span.label),
                        "full_delta": float(full_delta),
                        "mutated_delta": float(mutated_delta),
                        "effect_delta": float(compute_span_effect(full_delta=full_delta, mutated_delta=mutated_delta)),
                        "effect_source": source,
                        "counterfactual_mode": mode,
                    }
                )
                done_keys.add(key)

            prompts_processed += 1

        if mode == "model" and model_obj is not None:
            try:
                del model_obj
            except Exception:
                pass
            if device == "cuda":
                import torch  # noqa: PLC0415

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if device == "mps":
                import torch  # noqa: PLC0415

                if hasattr(torch, "mps"):
                    torch.mps.empty_cache()

        if thermal_stop_action is not None:
            break

    effects_new = pd.DataFrame.from_records(effects_rows)
    if existing_effects.empty:
        effects = effects_new
    elif effects_new.empty:
        effects = existing_effects
    else:
        effects = pd.concat([existing_effects, effects_new], ignore_index=True)
    if not effects.empty:
        effects = effects.drop_duplicates(subset=["model_id", "prompt_uid", "span_id"], keep="last")

    labels = label_span_effects(effects, output_col="span_label")

    spans_rows = new_spans_rows
    if args.resume and spans_path.exists():
        for row in iter_jsonl(spans_path):
            spans_rows.append(row)
    if spans_rows:
        # Keep last write for each key.
        tmp = pd.DataFrame.from_records(spans_rows)
        tmp = tmp.drop_duplicates(subset=["model_id", "prompt_uid", "span_id"], keep="last")
        spans_rows = tmp.to_dict(orient="records")

    _write_spans_jsonl(spans_path, spans_rows)
    write_parquet(effects_path, effects)
    write_parquet(out_root / "span_labels.parquet", labels)

    label_counts = labels["span_label"].value_counts().to_dict() if (not labels.empty and "span_label" in labels.columns) else {}

    done_sentinel = out_root / "sentinels" / "05_span_counterfactuals.done"
    if thermal_stop_action is None:
        done_sentinel.parent.mkdir(parents=True, exist_ok=True)
        done_sentinel.write_text(json.dumps({"stage": "05_span_counterfactuals", "pass": True}, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    write_json(
        out_root / "05_span_counterfactuals.report.json",
        {
            "pass": thermal_stop_action is None,
            "counterfactual_mode": mode,
            "allow_proxy_fallback": bool(allow_proxy_fallback),
            "proxy_fallback_count": int(fallback_count),
            "prompts_processed": int(prompts_processed),
            "spans": int(len(spans_rows)),
            "effects": int(effects.shape[0]),
            "span_labels": label_counts,
            "thermal_policy": {
                "enabled": bool(thermal_cfg_obj.enabled),
                "pause_mode": str(thermal_cfg_obj.pause_mode),
                "events_path": str(thermal_events),
            },
            "stopped_early": bool(thermal_stop_action is not None),
            "stop_reason": "thermal_checkpoint" if thermal_stop_action is not None else None,
            "thermal_action": thermal_stop_action,
            "done_sentinel": str(done_sentinel) if thermal_stop_action is None else None,
        },
    )
    print(str(out_root / "span_labels.parquet"))
    if thermal_stop_action is not None:
        return 95
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
