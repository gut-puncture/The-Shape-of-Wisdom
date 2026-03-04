#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from _common import (
    base_parser,
    baseline_manifest_path,
    load_experiment_config,
    resolve_models,
    run_v2_root_for,
    write_json,
    write_jsonl,
    write_parquet,
    write_text_atomic,
)
from sow.io_jsonl import iter_jsonl
from sow.thermal.thermal_governor import ThermalGovernor, ThermalHygieneConfig
from sow.v2.inference_firewall import assert_inference_allowed
from sow.v2.model_nuances import apply_tokenizer_nuance, assert_transformers_version_floor, pick_torch_dtype
from sow.v2.span_counterfactuals import completed_span_keys_for_mode, compute_span_effect, delete_span, label_span_effects
from sow.v2.span_paraphrase_stability import deterministic_paraphrase, proxy_mutated_delta, score_prompt_paraphrase
from sow.v2.span_parser import parse_prompt_spans

_deterministic_paraphrase = deterministic_paraphrase


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


def _position_ids(attn):
    p = attn.long().cumsum(dim=-1) - 1
    return p.masked_fill(attn == 0, 0)


def _choice_token_ids(tokenizer) -> Dict[str, int | None]:
    out: Dict[str, int | None] = {}
    for key in ["A", "B", "C", "D"]:
        ids = tokenizer.encode(key, add_special_tokens=False)
        # Multi-token labels are not safe to collapse to one token id for delta readout.
        out[key] = int(ids[0]) if len(ids) == 1 else None
    return out


def _validate_choice_token_ids(choice_token_ids: Dict[str, int | None]) -> None:
    missing = [k for k in ["A", "B", "C", "D"] if choice_token_ids.get(k) is None]
    if missing:
        raise ValueError(
            "single-token option label ids are required for model counterfactual mode; "
            f"missing labels={missing}"
        )


def _merge_span_rows(*, existing_rows: List[Dict[str, object]], new_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = [*existing_rows, *new_rows]
    if not rows:
        return []
    tmp = pd.DataFrame.from_records(rows)
    if tmp.empty:
        return []
    key_cols = ["model_id", "prompt_uid", "span_id"]
    if all(c in tmp.columns for c in key_cols):
        tmp = tmp.drop_duplicates(subset=key_cols, keep="last")
    return tmp.to_dict(orient="records")


def _merge_effect_rows(*, existing_df: pd.DataFrame, new_rows: List[Dict[str, object]]) -> pd.DataFrame:
    if not new_rows:
        return existing_df
    new_df = pd.DataFrame.from_records(new_rows)
    if existing_df.empty:
        merged = new_df
    elif new_df.empty:
        merged = existing_df
    else:
        merged = pd.concat([existing_df, new_df], ignore_index=True)
    if not merged.empty:
        merged = merged.drop_duplicates(subset=["model_id", "prompt_uid", "span_id"], keep="last")
    return merged


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
        raise ValueError(f"missing single-token id for correct_key={ck!r}")

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
            raise ValueError("unable to find competitor token id for A/B/C/D")
        k_tid = choice_token_ids.get(best_key)

    if k_tid is None:
        raise ValueError("competitor token id resolved to None")
    return float(logits[int(c_tid)].item() - logits[int(k_tid)].item())


def _write_spans_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    write_jsonl(path, rows)


def _compute_paraphrase_stability(
    *,
    model_prompt_uids_by_model: Dict[str, List[str]],
    manifest_rows: Dict[str, Dict[str, object]],
    info_by_model_prompt: Dict[Tuple[str, str], Dict[str, str | float]],
    sample_size_per_model: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(int(seed))
    rows: List[Dict[str, object]] = []
    for model_id in sorted(model_prompt_uids_by_model.keys()):
        candidates = [u for u in sorted(model_prompt_uids_by_model.get(model_id) or []) if (model_id, u) in info_by_model_prompt]
        if sample_size_per_model > 0 and len(candidates) > sample_size_per_model:
            sampled_uids = sorted(rng.sample(candidates, int(sample_size_per_model)))
        else:
            sampled_uids = candidates
        for prompt_uid in sampled_uids:
            info = info_by_model_prompt.get((model_id, prompt_uid)) or {}
            manifest_row = manifest_rows.get(prompt_uid) or {}
            prompt_text = str(manifest_row.get("prompt_text") or "")
            full_delta = float(info.get("full_delta") or 0.0)
            correct_key = str(info.get("correct_key") or manifest_row.get("correct_key") or "A").strip().upper()
            paraphrased = _deterministic_paraphrase(prompt_text)
            score = score_prompt_paraphrase(
                prompt_text=prompt_text,
                full_delta=full_delta,
                correct_key=correct_key,
                paraphrased_text=paraphrased,
            )
            rows.append(
                {
                    "model_id": str(model_id),
                    "prompt_uid": str(prompt_uid),
                    "label_agreement": float(score["label_agreement"]),
                    "span_jaccard": float(score["span_jaccard"]),
                    "n_common_labels": int(score["n_common_labels"]),
                    "n_original_labels": int(score["n_original_labels"]),
                    "n_paraphrased_labels": int(score["n_paraphrased_labels"]),
                }
            )
    return pd.DataFrame.from_records(rows)


def main() -> int:
    ap = base_parser("V2: parse spans and compute counterfactual span effects")
    ap.add_argument("--counterfactual-mode", choices=["model", "proxy"], default=None)
    ap.add_argument("--allow-proxy-fallback", action="store_true")
    args = ap.parse_args()

    cfg = load_experiment_config(Path(args.config))
    models = resolve_models(cfg, model_name=args.model_name)
    expected_model_ids = [str(m["model_id"]) for m in models]
    execution_cfg = cfg.get("execution") or {}
    validators_cfg = cfg.get("validators") or {}
    stage05_cfg = validators_cfg.get("stage05_paraphrase") or {}
    paraphrase_min_label_agreement = float(stage05_cfg.get("min_label_agreement", 0.85))
    paraphrase_min_span_jaccard = float(stage05_cfg.get("min_span_jaccard", 0.80))
    paraphrase_sample_size = int(stage05_cfg.get("sample_size_per_model", 50))
    sampling_cfg = cfg.get("sampling") or {}
    span_counterfactual_prompts_per_model = int(sampling_cfg.get("span_counterfactual_prompts_per_model", 0))
    span_counterfactual_max_prompts_per_model = int(sampling_cfg.get("span_counterfactual_max_prompts_per_model", 0))
    deterministic_seed = int(execution_cfg.get("deterministic_seed", 12345))
    checkpoint_every_prompts = max(1, int(execution_cfg.get("stage05_checkpoint_every_prompts", 25)))

    span_cfg = cfg.get("span_counterfactual") or {}
    mode = str(args.counterfactual_mode or span_cfg.get("mode") or "proxy")
    allow_proxy_fallback = bool(args.allow_proxy_fallback or span_cfg.get("allow_proxy_fallback", False))
    if mode == "model":
        assert_inference_allowed("scripts/v2/05_span_counterfactuals.py")

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
    existing_span_rows: List[Dict[str, object]] = []
    if args.resume and spans_path.exists():
        existing_span_rows = [dict(r) for r in iter_jsonl(spans_path)]

    new_spans_rows: List[Dict[str, object]] = []
    effects_rows: List[Dict[str, object]] = []
    pending_prompts_since_checkpoint = 0

    max_prompts = int(args.max_prompts)
    prompts_processed = 0
    fallback_count = 0
    thermal_stop_action = None
    model_level_proxy_fallback: Dict[str, str] = {}
    model_prompt_uids_by_model: Dict[str, List[str]] = {}
    available_prompt_counts: Dict[str, int] = {}
    requested_prompt_counts: Dict[str, int] = {}
    selected_prompt_counts: Dict[str, int] = {}

    def _flush_checkpoint() -> None:
        nonlocal existing_effects, existing_span_rows, new_spans_rows, effects_rows, pending_prompts_since_checkpoint
        if not new_spans_rows and not effects_rows:
            pending_prompts_since_checkpoint = 0
            return
        existing_span_rows = _merge_span_rows(existing_rows=existing_span_rows, new_rows=new_spans_rows)
        existing_effects = _merge_effect_rows(existing_df=existing_effects, new_rows=effects_rows)
        _write_spans_jsonl(spans_path, existing_span_rows)
        write_parquet(effects_path, existing_effects)
        labels_ckpt = label_span_effects(existing_effects, output_col="span_label")
        write_parquet(out_root / "span_labels.parquet", labels_ckpt)
        new_spans_rows = []
        effects_rows = []
        pending_prompts_since_checkpoint = 0

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
        available_prompt_counts[model_id] = int(len(model_prompt_uids))
        requested = int(span_counterfactual_prompts_per_model) if int(span_counterfactual_prompts_per_model) > 0 else int(len(model_prompt_uids))
        if int(span_counterfactual_max_prompts_per_model) > 0:
            requested = min(int(requested), int(span_counterfactual_max_prompts_per_model))
        if max_prompts > 0:
            requested = min(int(requested), int(max_prompts))
        requested = max(0, int(requested))
        requested_prompt_counts[model_id] = int(requested)
        if int(len(model_prompt_uids)) > int(requested):
            seed_offset = sum(ord(ch) for ch in model_id)
            rng = random.Random(int(deterministic_seed) + int(seed_offset))
            model_prompt_uids = sorted(rng.sample(model_prompt_uids, int(requested)))
        model_prompt_uids_by_model[model_id] = list(model_prompt_uids)
        selected_prompt_counts[model_id] = int(len(model_prompt_uids))

        model_obj = None
        tokenizer = None
        choice_token_ids: Dict[str, int | None] | None = None
        device = "cpu"
        model_counterfactual_enabled = mode == "model"

        if mode == "model":
            import torch  # noqa: PLC0415
            import transformers  # noqa: PLC0415
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

            assert_transformers_version_floor(model_id, str(transformers.__version__))
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True, trust_remote_code=False)
            apply_tokenizer_nuance(tokenizer, model_id=model_id)
            choice_token_ids = _choice_token_ids(tokenizer)
            try:
                _validate_choice_token_ids(choice_token_ids)
            except ValueError as exc:
                if not allow_proxy_fallback:
                    raise SystemExit(f"{model_id}: {exc}") from exc
                model_counterfactual_enabled = False
                model_level_proxy_fallback[model_id] = str(exc)

            if model_counterfactual_enabled:
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

                if model_counterfactual_enabled and model_obj is not None and tokenizer is not None and choice_token_ids is not None:
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
                        mutated_delta = proxy_mutated_delta(
                            full_delta=full_delta,
                            span_role=str(span.label),
                            span_len=int(span.end_char - span.start_char),
                            prompt_len=len(prompt_text),
                            correct_key=correct_key,
                        )
                else:
                    mutated_delta = proxy_mutated_delta(
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
            pending_prompts_since_checkpoint += 1
            if pending_prompts_since_checkpoint >= checkpoint_every_prompts:
                _flush_checkpoint()

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

    _flush_checkpoint()
    effects = existing_effects
    labels = label_span_effects(effects, output_col="span_label")
    spans_rows = existing_span_rows

    _write_spans_jsonl(spans_path, spans_rows)
    write_parquet(effects_path, effects)
    write_parquet(out_root / "span_labels.parquet", labels)

    stability = _compute_paraphrase_stability(
        model_prompt_uids_by_model=model_prompt_uids_by_model,
        manifest_rows=manifest_rows,
        info_by_model_prompt=info_by_model_prompt,
        sample_size_per_model=max(0, int(paraphrase_sample_size)),
        seed=deterministic_seed,
    )
    stability_path = out_root / "span_paraphrase_stability.parquet"
    write_parquet(stability_path, stability)
    label_agreement_mean = float(stability["label_agreement"].mean()) if (not stability.empty and "label_agreement" in stability.columns) else 0.0
    span_jaccard_mean = float(stability["span_jaccard"].mean()) if (not stability.empty and "span_jaccard" in stability.columns) else 0.0
    per_model_stability: Dict[str, Dict[str, float | int]] = {}
    for model_id in expected_model_ids:
        if stability.empty or "model_id" not in stability.columns:
            sub = pd.DataFrame()
        else:
            sub = stability[stability["model_id"].astype(str) == str(model_id)]
        per_model_stability[str(model_id)] = {
            "rows": int(sub.shape[0]),
            "label_agreement_mean": float(sub["label_agreement"].mean()) if (not sub.empty and "label_agreement" in sub.columns) else 0.0,
            "span_jaccard_mean": float(sub["span_jaccard"].mean()) if (not sub.empty and "span_jaccard" in sub.columns) else 0.0,
        }
    expected_models_present = bool(expected_model_ids) and all(int(per_model_stability[mid]["rows"]) > 0 for mid in expected_model_ids)
    per_model_label_ok = bool(expected_models_present) and all(
        float(per_model_stability[mid]["label_agreement_mean"]) >= float(paraphrase_min_label_agreement) for mid in expected_model_ids
    )
    per_model_span_ok = bool(expected_models_present) and all(
        float(per_model_stability[mid]["span_jaccard_mean"]) >= float(paraphrase_min_span_jaccard) for mid in expected_model_ids
    )
    paraphrase_gates = {
        "paraphrase_sample_nonempty": int(stability.shape[0]) > 0,
        "paraphrase_label_agreement": float(label_agreement_mean) >= float(paraphrase_min_label_agreement),
        "paraphrase_span_jaccard": float(span_jaccard_mean) >= float(paraphrase_min_span_jaccard),
        "paraphrase_expected_models_present": bool(expected_models_present),
        "paraphrase_label_agreement_per_model": bool(per_model_label_ok),
        "paraphrase_span_jaccard_per_model": bool(per_model_span_ok),
    }
    is_full_mode = int(max_prompts) == 0
    sampling_full_mode_prompt_count = True
    if is_full_mode and int(span_counterfactual_prompts_per_model) > 0:
        sampling_full_mode_prompt_count = all(
            int(selected_prompt_counts.get(mid, 0)) >= int(span_counterfactual_prompts_per_model) for mid in expected_model_ids
        )
    gates = {
        **paraphrase_gates,
        "sampling_full_mode_prompt_count": bool(sampling_full_mode_prompt_count),
    }
    failing_gates = sorted([k for k, v in gates.items() if not bool(v)])
    gates_pass = len(failing_gates) == 0

    label_counts = labels["span_label"].value_counts().to_dict() if (not labels.empty and "span_label" in labels.columns) else {}

    done_sentinel = out_root / "sentinels" / "05_span_counterfactuals.done"
    pass_flag = (thermal_stop_action is None) and bool(gates_pass)
    if pass_flag:
        done_sentinel.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(
            done_sentinel,
            json.dumps({"stage": "05_span_counterfactuals", "pass": True}, ensure_ascii=False, sort_keys=True) + "\n",
        )

    write_json(
        out_root / "05_span_counterfactuals.report.json",
        {
            "pass": bool(pass_flag),
            "counterfactual_mode": mode,
            "allow_proxy_fallback": bool(allow_proxy_fallback),
            "proxy_fallback_count": int(fallback_count),
            "model_level_proxy_fallback": model_level_proxy_fallback,
            "prompts_processed": int(prompts_processed),
            "spans": int(len(spans_rows)),
            "effects": int(effects.shape[0]),
            "span_labels": label_counts,
            "thermal_policy": {
                "enabled": bool(thermal_cfg_obj.enabled),
                "pause_mode": str(thermal_cfg_obj.pause_mode),
                "events_path": str(thermal_events),
            },
            "paraphrase_stability": {
                "sample_size_per_model": int(paraphrase_sample_size),
                "expected_models": expected_model_ids,
                "rows": int(stability.shape[0]),
                "label_agreement_mean": float(label_agreement_mean),
                "span_jaccard_mean": float(span_jaccard_mean),
                "per_model": per_model_stability,
                "out_path": str(stability_path),
            },
            "sampling_contract": {
                "is_full_mode": bool(is_full_mode),
                "span_counterfactual_prompts_per_model": int(span_counterfactual_prompts_per_model),
                "span_counterfactual_max_prompts_per_model": int(span_counterfactual_max_prompts_per_model),
                "available_prompt_counts": {str(k): int(v) for k, v in available_prompt_counts.items()},
                "requested_prompt_counts": {str(k): int(v) for k, v in requested_prompt_counts.items()},
                "selected_prompt_counts": {str(k): int(v) for k, v in selected_prompt_counts.items()},
            },
            "gates": gates,
            "failing_gates": failing_gates,
            "stopped_early": bool(thermal_stop_action is not None),
            "stop_reason": "thermal_checkpoint" if thermal_stop_action is not None else None,
            "thermal_action": thermal_stop_action,
            "done_sentinel": str(done_sentinel) if pass_flag else None,
        },
    )
    print(str(out_root / "span_labels.parquet"))
    if thermal_stop_action is not None:
        return 95
    if not gates_pass:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
