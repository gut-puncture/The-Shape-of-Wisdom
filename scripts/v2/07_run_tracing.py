#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from _common import (
    base_parser,
    load_experiment_config,
    resolve_models,
    run_v2_root_for,
    write_json,
    write_parquet,
    write_text_atomic,
)
from sow.thermal.thermal_governor import ThermalGovernor, ThermalHygieneConfig
from sow.v2.inference_firewall import assert_inference_allowed
from sow.v2.model_nuances import apply_tokenizer_nuance, assert_transformers_version_floor, get_model_nuance, pick_torch_dtype
from sow.v2.span_parser import parse_prompt_spans
from sow.v2.tracing.decomposition import attention_mass_by_span_per_layer, drift_series_from_deltas
from sow.v2.tracing.hooks import capture_component_outputs


def _position_ids(attn: torch.Tensor) -> torch.Tensor:
    p = attn.long().cumsum(dim=-1) - 1
    return p.masked_fill(attn == 0, 0)


def _set_deterministic_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


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
            "single-token option label ids are required for tracing delta readout; "
            f"missing labels={missing}"
        )


def _logits_from_hidden(model_obj, hidden_vec: torch.Tensor) -> torch.Tensor:
    h = hidden_vec.unsqueeze(0)
    core = getattr(model_obj, "model", None)
    norm = getattr(core, "norm", None)
    if norm is not None:
        h = norm(h)
    logits = model_obj.lm_head(h)[0]
    return logits


def _competitor_key(*, logits: torch.Tensor, choice_token_ids: Dict[str, int | None], correct_key: str) -> str:
    ck = str(correct_key).strip().upper()
    candidates: List[Tuple[str, float]] = []
    for key in ["A", "B", "C", "D"]:
        if key == ck:
            continue
        tid = choice_token_ids.get(key)
        val = float(logits[int(tid)].item()) if tid is not None else -1e9
        candidates.append((key, val))
    if not candidates:
        return "A"
    candidates.sort(key=lambda x: x[1], reverse=True)
    return str(candidates[0][0])


def _delta_from_logits(*, logits: torch.Tensor, correct_tid: int | None, competitor_tid: int | None) -> float:
    if correct_tid is None or competitor_tid is None:
        return 0.0
    return float(logits[int(correct_tid)].item() - logits[int(competitor_tid)].item())


def _span_token_indices(prompt_text: str, offsets: List[List[int]]) -> Dict[str, List[int]]:
    spans = parse_prompt_spans(prompt_text)
    out: Dict[str, List[int]] = {s.label: [] for s in spans}
    for i, (start, end) in enumerate(offsets):
        for s in spans:
            if int(end) <= int(s.start_char) or int(start) >= int(s.end_char):
                continue
            out[s.label].append(int(i))
    return out


def _load_existing(path: Path, *, key_cols: List[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return df
    return df.drop_duplicates(subset=key_cols, keep="last")


def _merge_resume(existing: pd.DataFrame, new_df: pd.DataFrame, *, key_cols: List[str]) -> pd.DataFrame:
    if existing.empty:
        return new_df.drop_duplicates(subset=key_cols, keep="last") if not new_df.empty else new_df
    if new_df.empty:
        return existing
    merged = pd.concat([existing, new_df], ignore_index=True)
    return merged.drop_duplicates(subset=key_cols, keep="last")


def main() -> int:
    ap = base_parser("V2: run tracing for attention routing and MLP injection")
    args = ap.parse_args()
    assert_inference_allowed("scripts/v2/07_run_tracing.py")

    cfg = load_experiment_config(Path(args.config))
    models = resolve_models(cfg, model_name=args.model_name)
    out_root = run_v2_root_for(args.run_id)

    execution_cfg = cfg.get("execution") or {}
    seed = int(execution_cfg.get("deterministic_seed", 12345))
    checkpoint_every_prompts = max(1, int(execution_cfg.get("stage07_checkpoint_every_prompts", 25)))
    _set_deterministic_seed(seed)

    thermal_cfg_obj = ThermalHygieneConfig.from_cfg(cfg.get("thermal_policy"))
    thermal_events = out_root / "meta" / "thermal_events_tracing.jsonl"
    governor = ThermalGovernor(cfg=thermal_cfg_obj, events_path=thermal_events) if thermal_cfg_obj.enabled else None

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    max_prompts = int(args.max_prompts)

    scalar_path = out_root / "tracing_scalars.parquet"
    mass_path = out_root / "attention_mass_by_span.parquet"
    contrib_path = out_root / "attention_contrib_by_span.parquet"

    scalar_key = ["model_id", "prompt_uid", "layer_index"]
    mass_key = ["model_id", "prompt_uid", "layer_index", "span_label"]

    scalars_existing = _load_existing(scalar_path, key_cols=scalar_key) if args.resume else pd.DataFrame()
    mass_existing = _load_existing(mass_path, key_cols=mass_key) if args.resume else pd.DataFrame()
    contrib_existing = _load_existing(contrib_path, key_cols=mass_key) if args.resume else pd.DataFrame()

    done_prompt_keys = set()
    if not scalars_existing.empty:
        done_prompt_keys = {
            (str(r["model_id"]), str(r["prompt_uid"]))
            for _, r in scalars_existing[["model_id", "prompt_uid"]].drop_duplicates().iterrows()
        }

    scalar_rows = []
    mass_rows = []
    contrib_rows = []
    pending_prompts_since_checkpoint = 0

    thermal_stop_action = None
    models_completed = 0
    prompts_attempted = 0

    def _flush_checkpoint() -> None:
        nonlocal scalars_existing, mass_existing, contrib_existing
        nonlocal scalar_rows, mass_rows, contrib_rows, pending_prompts_since_checkpoint
        if not scalar_rows and not mass_rows and not contrib_rows:
            pending_prompts_since_checkpoint = 0
            return
        scalars_new = pd.DataFrame.from_records(scalar_rows)
        mass_new = pd.DataFrame.from_records(mass_rows)
        contrib_new = pd.DataFrame.from_records(contrib_rows)
        scalars_existing = _merge_resume(scalars_existing, scalars_new, key_cols=scalar_key)
        mass_existing = _merge_resume(mass_existing, mass_new, key_cols=mass_key)
        contrib_existing = _merge_resume(contrib_existing, contrib_new, key_cols=mass_key)
        write_parquet(scalar_path, scalars_existing)
        write_parquet(mass_path, mass_existing)
        write_parquet(contrib_path, contrib_existing)
        scalar_rows = []
        mass_rows = []
        contrib_rows = []
        pending_prompts_since_checkpoint = 0

    for model in models:
        model_id = str(model["model_id"])
        revision = str(model["revision"])

        subset_path = out_root / f"tracing_subset_{model_id.replace('/', '__')}.json"
        if not subset_path.exists():
            raise SystemExit(f"missing tracing subset: {subset_path}")
        subset = json.loads(subset_path.read_text(encoding="utf-8"))
        if max_prompts > 0:
            subset = subset[:max_prompts]

        nuance = get_model_nuance(model_id)
        assert_transformers_version_floor(model_id, str(transformers.__version__))

        tok = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True, trust_remote_code=False)
        apply_tokenizer_nuance(tok, model_id=model_id)
        kwargs = {
            "revision": revision,
            "torch_dtype": pick_torch_dtype(device=device),
            "low_cpu_mem_usage": True,
            "trust_remote_code": False,
        }
        load_kwargs = dict(kwargs)
        if nuance.force_eager_attention_for_tracing:
            load_kwargs["attn_implementation"] = "eager"
        try:
            mdl = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        except TypeError:
            mdl = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        mdl.eval()
        if device != "cpu":
            mdl.to(device)

        choice_token_ids = _choice_token_ids(tok)
        try:
            _validate_choice_token_ids(choice_token_ids)
        except ValueError as exc:
            raise SystemExit(f"{model_id}: {exc}") from exc

        with torch.inference_mode():
            for rec in subset:
                prompt_uid = str(rec.get("prompt_uid") or "")
                if not prompt_uid:
                    continue
                if (model_id, prompt_uid) in done_prompt_keys:
                    continue

                if governor is not None:
                    action = governor.maybe_cooldown(
                        stage="v2_tracing",
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

                prompts_attempted += 1
                prompt_text = str(rec.get("prompt_text") or "")
                correct_key = str(rec.get("correct_key") or "A").strip().upper()

                enc = tok(prompt_text, return_tensors="pt", return_offsets_mapping=True, padding=True)
                offsets = enc.pop("offset_mapping")[0].tolist() if "offset_mapping" in enc else []
                input_ids = enc["input_ids"]
                attn = enc.get("attention_mask")
                if attn is None:
                    raise RuntimeError("tokenizer missing attention_mask")
                if device != "cpu":
                    input_ids = input_ids.to(device)
                    attn = attn.to(device)
                pos = _position_ids(attn)

                traced = capture_component_outputs(
                    model_obj=mdl,
                    input_ids=input_ids,
                    attention_mask=attn,
                    position_ids=pos,
                    output_attentions=True,
                )
                out = traced["model_output"]
                hidden_states = out.hidden_states
                if hidden_states is None or len(hidden_states) < 2:
                    continue

                attn_outs = traced["attn_outputs"]
                mlp_outs = traced["mlp_outputs"]
                n_layers = min(len(hidden_states) - 1, len(attn_outs), len(mlp_outs))
                if n_layers <= 0:
                    continue

                span_tokens = _span_token_indices(prompt_text, offsets)
                layer_masses = attention_mass_by_span_per_layer(out.attentions, span_token_indices=span_tokens)

                # Align deltas/drifts to transformer-layer outputs (same convention as baseline metrics):
                # delta(l) from hidden_states[l+1], drift(l)=delta(l+1)-delta(l), terminal drift=0.
                layer_deltas: List[float] = []
                layer_comp_keys: List[str] = []
                for li in range(n_layers):
                    h_layer_out = hidden_states[li + 1][0, -1, :]
                    logits_layer = _logits_from_hidden(mdl, h_layer_out)
                    comp_key = _competitor_key(
                        logits=logits_layer,
                        choice_token_ids=choice_token_ids,
                        correct_key=correct_key,
                    )
                    c_tid_layer = choice_token_ids.get(correct_key)
                    k_tid_layer = choice_token_ids.get(comp_key)
                    delta_layer = _delta_from_logits(
                        logits=logits_layer,
                        correct_tid=c_tid_layer,
                        competitor_tid=k_tid_layer,
                    )
                    layer_deltas.append(float(delta_layer))
                    layer_comp_keys.append(str(comp_key))
                drifts = drift_series_from_deltas(np.asarray(layer_deltas, dtype=np.float64))

                for li in range(n_layers):
                    h_in = hidden_states[li][0, -1, :]
                    comp_key = layer_comp_keys[li]
                    c_tid = choice_token_ids.get(correct_key)
                    k_tid = choice_token_ids.get(comp_key)

                    delta = float(layer_deltas[li])
                    drift = float(drifts[li]) if li < len(drifts) else 0.0

                    attn_vec = attn_outs[li][0, -1, :]
                    mlp_vec = mlp_outs[li][0, -1, :]

                    logits_in = _logits_from_hidden(mdl, h_in)
                    delta_in = _delta_from_logits(logits=logits_in, correct_tid=c_tid, competitor_tid=k_tid)
                    logits_after_attn = _logits_from_hidden(mdl, h_in + attn_vec)
                    delta_after_attn = _delta_from_logits(logits=logits_after_attn, correct_tid=c_tid, competitor_tid=k_tid)
                    s_attn = float(delta_after_attn - delta_in)

                    logits_after_mlp = _logits_from_hidden(mdl, h_in + attn_vec + mlp_vec)
                    delta_after_mlp = _delta_from_logits(logits=logits_after_mlp, correct_tid=c_tid, competitor_tid=k_tid)
                    s_mlp = float(delta_after_mlp - delta_after_attn)

                    scalar_rows.append(
                        {
                            "model_id": model_id,
                            "model_revision": revision,
                            "prompt_uid": prompt_uid,
                            "layer_index": int(li),
                            "correct_key": correct_key,
                            "competitor_key": comp_key,
                            "delta": float(delta),
                            "drift": float(drift),
                            "s_attn": float(s_attn),
                            "s_mlp": float(s_mlp),
                        }
                    )

                    layer_mass = layer_masses[li] if li < len(layer_masses) else {}
                    for span_label, mass in layer_mass.items():
                        mass_rows.append(
                            {
                                "model_id": model_id,
                                "model_revision": revision,
                                "prompt_uid": prompt_uid,
                                "layer_index": int(li),
                                "span_label": str(span_label),
                                "attention_mass": float(mass),
                            }
                        )
                        contrib_rows.append(
                            {
                                "model_id": model_id,
                                "model_revision": revision,
                                "prompt_uid": prompt_uid,
                                "layer_index": int(li),
                                "span_label": str(span_label),
                                "attention_contribution": float(float(mass) * float(s_attn)),
                            }
                        )

                done_prompt_keys.add((model_id, prompt_uid))
                pending_prompts_since_checkpoint += 1
                if pending_prompts_since_checkpoint >= checkpoint_every_prompts:
                    _flush_checkpoint()

        try:
            del mdl
        except Exception:
            pass
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

        if thermal_stop_action is not None:
            break

        models_completed += 1

    _flush_checkpoint()
    scalars = scalars_existing
    mass = mass_existing
    contrib = contrib_existing

    report = {
        "pass": thermal_stop_action is None,
        "run_id": args.run_id,
        "device": device,
        "seed": seed,
        "models_requested": [str(m["model_id"]) for m in models],
        "models_completed": int(models_completed),
        "prompts_attempted": int(prompts_attempted),
        "rows": {
            "tracing_scalars": int(scalars.shape[0]),
            "attention_mass_by_span": int(mass.shape[0]),
            "attention_contrib_by_span": int(contrib.shape[0]),
        },
        "thermal_policy": {
            "enabled": bool(thermal_cfg_obj.enabled),
            "pause_mode": str(thermal_cfg_obj.pause_mode),
            "events_path": str(thermal_events),
        },
    }
    done_sentinel = out_root / "sentinels" / "07_run_tracing.done"
    if thermal_stop_action is not None:
        report.update(
            {
                "stopped_early": True,
                "stop_reason": "thermal_checkpoint",
                "thermal_action": thermal_stop_action,
            }
        )
    else:
        done_sentinel.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(
            done_sentinel,
            json.dumps({"stage": "07_run_tracing", "pass": True}, ensure_ascii=False, sort_keys=True) + "\n",
        )
        report["done_sentinel"] = str(done_sentinel)

    write_json(out_root / "07_run_tracing.report.json", report)
    print(str(scalar_path))

    if thermal_stop_action is not None:
        return 95
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
