#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import append_jsonl, iter_jsonl, read_jsonl
from option_token_buckets import build_option_token_buckets, buckets_sha256, normalize_piece_to_option


def detect_device(force: Optional[str] = None) -> str:
    if force and force != "auto":
        return force
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _compute_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Build pad-aware position ids so batched and unbatched runs align."""
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
    return position_ids


def _model_forward_with_optional_position_ids(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    output_hidden_states: bool = False,
    use_cache: bool = True,
    return_dict: bool = True,
    past_key_values=None,
    position_ids: Optional[torch.Tensor] = None,
):
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_hidden_states": output_hidden_states,
        "use_cache": use_cache,
        "return_dict": return_dict,
    }
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
    if position_ids is not None:
        kwargs["position_ids"] = position_ids

    try:
        return model(**kwargs)
    except TypeError as exc:
        # Some architectures may not expose position_ids in forward kwargs.
        if position_ids is None:
            raise
        msg = str(exc)
        if "position_ids" not in msg and "unexpected keyword argument" not in msg:
            raise
        kwargs.pop("position_ids", None)
        return model(**kwargs)


def get_final_norm_module(model):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    if hasattr(model, "base_model") and hasattr(model.base_model, "norm"):
        return model.base_model.norm
    return None


def extract_choice(decoded: str) -> Optional[str]:
    # Option-token invariance: decoded may be "A", " a", "(A)", "Ａ", etc.
    return normalize_piece_to_option(decoded)


def entropy(probs: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.sum(probs * np.log(probs + eps)))


def compute_commitment_layers(argmax_letters: List[str], margins: List[float], thresholds: Sequence[float]) -> Dict[str, Optional[int]]:
    out: Dict[str, Optional[int]] = {}
    final_letter = argmax_letters[-1]
    n = len(argmax_letters)
    for m in thresholds:
        found = None
        for i in range(n):
            if argmax_letters[i] != final_letter:
                continue
            if margins[i] < m:
                continue
            stable = all(argmax_letters[j] == final_letter for j in range(i, n))
            if stable:
                found = i + 1  # 1-indexed layer
                break
        out[str(m)] = found
    return out


def reversal_count(argmax_letters: List[str]) -> int:
    if not argmax_letters:
        return 0
    c = 0
    for i in range(1, len(argmax_letters)):
        if argmax_letters[i] != argmax_letters[i - 1]:
            c += 1
    return c


def classify_error_regimes(
    argmax_letters: List[str],
    margins: List[float],
    correct_key: str,
) -> Tuple[bool, bool, float]:
    n = len(argmax_letters)
    final_letter = argmax_letters[-1]
    rev = reversal_count(argmax_letters)

    # Early wrong lock-in: wrong final choice, low reversals, and stable from first quarter onward.
    q1 = max(1, n // 4)
    early_wrong_lock_in = False
    if final_letter != correct_key:
        stable_from_q1 = all(letter == final_letter for letter in argmax_letters[q1 - 1 :])
        strong_margin_early = margins[q1 - 1] >= 0.2
        early_wrong_lock_in = stable_from_q1 and strong_margin_early and rev <= 1

    # Late correction: final correct, but wrong for most early layers and switch late.
    late_correction = False
    if final_letter == correct_key:
        wrong_prefix = sum(1 for x in argmax_letters[: max(1, (3 * n) // 4)] if x != correct_key)
        switched_late = argmax_letters[-2] != correct_key if n >= 2 else False
        late_correction = wrong_prefix >= max(1, n // 2) and switched_late

    max_wrong_prob = 0.0
    return early_wrong_lock_in, late_correction, max_wrong_prob


def max_wrong_prob(layer_probs: List[List[float]], correct_key: str) -> float:
    idx = {"A": 0, "B": 1, "C": 2, "D": 3}[correct_key]
    m = 0.0
    for p in layer_probs:
        for j, v in enumerate(p):
            if j != idx:
                m = max(m, float(v))
    return m


def maybe_token_index_from_char_index(token_pieces: List[str], char_index: Optional[int]) -> Optional[int]:
    if char_index is None or char_index < 0:
        return None
    total = 0
    for i, p in enumerate(token_pieces, start=1):
        total += len(p)
        if total > char_index:
            return i
    return None


def response_position_diagnostics(response_text: str, token_pieces: List[str]) -> Dict:
    text = response_text or ""
    upper = text.upper()

    first_non_ws_char_idx = None
    first_non_ws_char = None
    m0 = re.search(r"\S", text)
    if m0:
        first_non_ws_char_idx = int(m0.start())
        first_non_ws_char = text[first_non_ws_char_idx]

    first_non_ws_char_is_option_letter = False
    if first_non_ws_char is not None:
        first_non_ws_char_is_option_letter = bool(normalize_piece_to_option(first_non_ws_char) in {"A", "B", "C", "D"})

    # broad "option mention" detector.
    option_match = None
    option_pattern = re.compile(
        r"\b(?:OPTION|CHOICE|ANSWER|FINAL\s+ANSWER|CORRECT\s+ANSWER|CORRECT\s+OPTION|BEST\s+ANSWER)\b"
        r"\s*(?:IS|:|=)?\s*[\(\[\{<\s]*([ABCD])\b"
        r"|\b([ABCD])\b\s*(?:IS\s*)?(?:THE\s*)?(?:CORRECT|RIGHT|BEST)\b"
        r"|^\s*[\(\[\{<\"']*\s*([ABCD])\b",
        flags=re.IGNORECASE,
    )
    m = option_pattern.search(upper)
    if m:
        option_match = int(m.start())

    return {
        "response_chars": len(text),
        "response_tokens_generated": len(token_pieces),
        "first_non_ws_char": first_non_ws_char,
        "first_non_ws_char_index": first_non_ws_char_idx,
        "first_non_ws_char_is_option_letter": first_non_ws_char_is_option_letter,
        "option_mention_char_index": option_match,
        "option_mention_token_index_estimate": maybe_token_index_from_char_index(token_pieces, option_match),
    }


def _prepare_bucket_projection(model, device: str, option_buckets: Dict[str, List[int]]):
    """Precompute the sliced lm_head projection for option-token invariance.

    Returns:
      union_token_ids (List[int])
      union_weight (Tensor[union, hidden])
      union_bias (Tensor[union] or None)
      option_union_indices (Dict[str, Tensor[k]]) mapping A/B/C/D to indices into the union dimension.

    Note: keeps tensors on `device`.
    """

    for opt in ["A", "B", "C", "D"]:
        if not option_buckets.get(opt):
            raise RuntimeError(f"Empty option bucket for {opt}. Tokenizer may be incompatible.")

    union_token_ids = sorted(set(sum((option_buckets[o] for o in ["A", "B", "C", "D"]), [])))
    union_ids_t = torch.tensor(union_token_ids, dtype=torch.long, device=device)

    union_index = {tid: i for i, tid in enumerate(union_token_ids)}
    option_union_indices = {
        opt: torch.tensor([union_index[int(tid)] for tid in option_buckets[opt]], dtype=torch.long, device=device)
        for opt in ["A", "B", "C", "D"]
    }

    lm_head = model.get_output_embeddings()
    if lm_head is None or not hasattr(lm_head, "weight"):
        raise RuntimeError("Model does not expose output embeddings (lm_head weight)")

    # Materialize sliced weights/bias on device (cheap: union<=~200).
    W = lm_head.weight
    W_union = W.index_select(0, union_ids_t).contiguous()
    b_union = None
    if getattr(lm_head, "bias", None) is not None:
        b_union = lm_head.bias.index_select(0, union_ids_t).contiguous()

    return union_token_ids, W_union, b_union, option_union_indices


def run_forward(
    model,
    tokenizer,
    final_norm,
    option_buckets: Dict[str, List[int]],
    union_token_ids: List[int],
    union_W: torch.Tensor,
    union_b: Optional[torch.Tensor],
    option_union_indices: Dict[str, torch.Tensor],
    prompt: str,
    device: str,
    top20: bool,
    response_max_new_tokens: int,
):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    position_ids = _compute_position_ids(attention_mask)
    prompt_len = input_ids.shape[1]

    # --- Pass 1: Forward with hidden states + KV-cache ---
    # use_cache=True so we can reuse KV states for generation (avoids re-encoding prompt).
    with torch.no_grad():
        outputs = _model_forward_with_optional_position_ids(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states[1:]  # exclude embedding layer
    n_layers = len(hidden_states)
    p = prompt_len - 1  # position of last input token

    # Final-token prediction from standard next-token logits.
    final_logits = outputs.logits[0, -1, :]
    first_token_id = int(torch.argmax(final_logits).item())
    first_token_decoded = tokenizer.decode([first_token_id], skip_special_tokens=True)
    chosen_letter = extract_choice(first_token_decoded)

    # Save KV-cache for generation reuse, then free the large outputs object.
    past_key_values = outputs.past_key_values
    del outputs

    # --- Batched per-layer computation ---
    # Stack all layer hidden states at the final token position: [n_layers, d]
    all_h = torch.stack([hidden_states[l][0, p, :] for l in range(n_layers)], dim=0)
    # Free hidden states immediately (large: n_layers × seq_len × hidden_dim).
    del hidden_states

    # Apply layer norm in batch: [n_layers, d] → [n_layers, d]
    if final_norm is not None:
        all_h_norm = final_norm(all_h)
    else:
        all_h_norm = all_h

    # Batched projection: [n_layers, d] @ [d, U] = [n_layers, U]
    all_union_logits = torch.matmul(all_h_norm, union_W.t())
    if union_b is not None:
        all_union_logits = all_union_logits + union_b  # broadcast [U] over [n_layers, U]

    # Compute bucket scores per layer (lightweight loop — all data already on GPU).
    logits_by_layer: List[List[float]] = []
    probs_by_layer: List[List[float]] = []
    margins: List[float] = []
    entropies: List[float] = []
    argmax_letters: List[str] = []
    top20_by_layer: List[List[str]] = []

    for l in range(n_layers):
        union_logits = all_union_logits[l]  # [U]

        # Bucket scores: logsumexp over token variants within each option.
        bucket_logits = []
        for opt in ["A", "B", "C", "D"]:
            idxs = option_union_indices[opt]
            bucket_logits.append(torch.logsumexp(union_logits.index_select(0, idxs), dim=0))
        bucket_logits_t = torch.stack(bucket_logits, dim=0).to(torch.float32)
        bucket_probs_t = torch.softmax(bucket_logits_t, dim=0)

        cand_logits = bucket_logits_t.detach().cpu().numpy().astype(np.float64)
        cand_probs = bucket_probs_t.detach().cpu().numpy().astype(np.float64)

        order = np.argsort(cand_probs)[::-1]
        top = cand_probs[order[0]]
        second = cand_probs[order[1]]
        margin = float(top - second)
        ent = entropy(cand_probs)
        top_letter = ["A", "B", "C", "D"][int(order[0])]

        logits_by_layer.append(cand_logits.tolist())
        probs_by_layer.append(cand_probs.tolist())
        margins.append(margin)
        entropies.append(ent)
        argmax_letters.append(top_letter)

        if top20:
            # Full projection only for sanity/debug.
            h_norm_l = all_h_norm[l:l+1]  # [1,d]
            vocab_logits_full = model.get_output_embeddings()(h_norm_l).squeeze(0)
            vals, ids = torch.topk(vocab_logits_full, k=20)
            toks = tokenizer.batch_decode(ids.unsqueeze(-1), skip_special_tokens=True)
            top20_by_layer.append([t.strip() for t in toks])

    # Single CPU transfer for all PCA source vectors: [n_layers, d] → numpy
    vecs = all_h.detach().float().cpu().numpy()
    del all_h, all_h_norm, all_union_logits

    # --- Pass 2: Generation (reusing KV-cache from Pass 1) ---
    # Manual autoregressive loop instead of model.generate() for KV-cache compatibility.
    # This avoids re-encoding the full prompt — we continue from the cached state.
    remaining_tokens = response_max_new_tokens - 1  # Already have first token
    all_new_ids = [first_token_id]

    if remaining_tokens > 0:
        cur_id = torch.tensor([[first_token_id]], dtype=torch.long, device=device)
        cur_mask = torch.cat([
            attention_mask,
            torch.ones((1, 1), dtype=attention_mask.dtype, device=device),
        ], dim=1)
        kv = past_key_values

        with torch.no_grad():
            for _ in range(remaining_tokens):
                step_position_ids = _compute_position_ids(cur_mask)[:, -1:]
                step_out = _model_forward_with_optional_position_ids(
                    model,
                    input_ids=cur_id,
                    attention_mask=cur_mask,
                    past_key_values=kv,
                    position_ids=step_position_ids,
                    use_cache=True,
                    return_dict=True,
                )
                next_id = int(torch.argmax(step_out.logits[0, -1, :]).item())
                kv = step_out.past_key_values

                # Stop at EOS
                if next_id == tokenizer.eos_token_id:
                    break

                all_new_ids.append(next_id)
                cur_id = torch.tensor([[next_id]], dtype=torch.long, device=device)
                cur_mask = torch.cat([
                    cur_mask,
                    torch.ones((1, 1), dtype=cur_mask.dtype, device=device),
                ], dim=1)

        del kv
    del past_key_values

    response_text = tokenizer.decode(all_new_ids, skip_special_tokens=True)
    response_token_pieces = [tokenizer.decode([int(tid)], skip_special_tokens=True) for tid in all_new_ids]
    resp_diag = response_position_diagnostics(response_text, response_token_pieces)

    return {
        "input_tokens": int(prompt_len),
        "first_token_id": first_token_id,
        "first_token_decoded": first_token_decoded,
        "chosen_letter": chosen_letter,
        "layer_candidate_logits": logits_by_layer,
        "layer_candidate_probs": probs_by_layer,
        "layer_margin": margins,
        "layer_entropy": entropies,
        "layer_argmax": argmax_letters,
        "layer_hidden_vectors": vecs,
        "layer_top20_tokens": top20_by_layer if top20 else None,
        "response_text": response_text,
        "response_token_ids": [int(x) for x in all_new_ids],
        "response_token_pieces": response_token_pieces,
        "response_diagnostics": resp_diag,
    }


def run_forward_batch(
    model,
    tokenizer,
    final_norm,
    option_buckets: Dict[str, List[int]],
    union_token_ids: List[int],
    union_W: torch.Tensor,
    union_b: Optional[torch.Tensor],
    option_union_indices: Dict[str, torch.Tensor],
    prompts: List[str],
    device: str,
    top20: bool,
    response_max_new_tokens: int,
) -> List[Dict]:
    """Fully batched inference: Pass 1 (hidden states) + Pass 2 (generation).

    Expects tokenizer.padding_side == "left" so the last token position is always
    the meaningful end-of-prompt for each sample.

    Returns a list of dicts (one per prompt) with identical keys to run_forward().
    """
    B = len(prompts)
    if B == 0:
        return []

    # --- Tokenize with left-padding ---
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    input_ids = inputs["input_ids"].to(device)      # [B, max_seq_len]
    attention_mask = inputs["attention_mask"].to(device)  # [B, max_seq_len]
    position_ids = _compute_position_ids(attention_mask)
    prompt_lens = attention_mask.sum(dim=1)  # [B] — number of real tokens per sample

    # --- Pass 1: Batched forward with hidden states + KV-cache ---
    with torch.no_grad():
        outputs = _model_forward_with_optional_position_ids(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states[1:]  # exclude embedding layer
    n_layers = len(hidden_states)

    # For each sample, the last real token position is at (max_seq_len - 1)
    # because of left-padding — the real tokens are right-aligned.
    # The position is: attention_mask[i].sum() - 1 gives the index of the
    # last real token... but since left-padding pushes tokens to the right,
    # the last real token is always at position (max_seq_len - 1).
    max_seq_len = input_ids.shape[1]
    # With left-padding, the last token for ALL samples is at position max_seq_len - 1.
    # But the last REAL token for sample i is at position (max_seq_len - 1) only if
    # the sample's real content ends there. With left-padding, real tokens are right-aligned,
    # so the last real token is always at position (max_seq_len - 1).
    last_positions = torch.full((B,), max_seq_len - 1, dtype=torch.long, device=device)

    # Extract first_token_id per sample from logits at last position.
    # With left-padding, last position = max_seq_len - 1 for all samples.
    batch_final_logits = outputs.logits[torch.arange(B, device=device), last_positions, :]  # [B, vocab]
    first_token_ids = torch.argmax(batch_final_logits, dim=-1)  # [B]

    # Save KV-cache, free large outputs.
    past_key_values = outputs.past_key_values
    del outputs, batch_final_logits

    # --- Batched per-layer hidden-state extraction ---
    # Extract hidden vectors at the last real token position for each sample, each layer.
    # all_h shape: [B, n_layers, d]
    all_h = torch.stack(
        [hidden_states[l][torch.arange(B, device=device), last_positions, :] for l in range(n_layers)],
        dim=1,
    )
    del hidden_states

    # Reshape for batched norm + projection: [B * n_layers, d]
    d = all_h.shape[-1]
    all_h_flat = all_h.reshape(B * n_layers, d)

    if final_norm is not None:
        all_h_norm_flat = final_norm(all_h_flat)
    else:
        all_h_norm_flat = all_h_flat

    # Batched projection: [B*n_layers, d] @ [d, U] = [B*n_layers, U]
    all_union_logits_flat = torch.matmul(all_h_norm_flat, union_W.t())
    if union_b is not None:
        all_union_logits_flat = all_union_logits_flat + union_b

    # Reshape to [B, n_layers, U]
    U = union_W.shape[0]
    all_union_logits = all_union_logits_flat.reshape(B, n_layers, U)
    all_h_norm = all_h_norm_flat.reshape(B, n_layers, d)
    del all_h_flat, all_h_norm_flat, all_union_logits_flat

    # --- Per-sample per-layer option scores (vectorized) ---
    results_hidden: List[Dict] = []
    for i in range(B):
        logits_by_layer: List[List[float]] = []
        probs_by_layer: List[List[float]] = []
        margins: List[float] = []
        entropies: List[float] = []
        argmax_letters: List[str] = []
        top20_by_layer: List[List[str]] = []

        for l in range(n_layers):
            union_logits_l = all_union_logits[i, l]  # [U]

            bucket_logits = []
            for opt in ["A", "B", "C", "D"]:
                idxs = option_union_indices[opt]
                bucket_logits.append(torch.logsumexp(union_logits_l.index_select(0, idxs), dim=0))
            bucket_logits_t = torch.stack(bucket_logits, dim=0).to(torch.float32)
            bucket_probs_t = torch.softmax(bucket_logits_t, dim=0)

            cand_logits = bucket_logits_t.detach().cpu().numpy().astype(np.float64)
            cand_probs = bucket_probs_t.detach().cpu().numpy().astype(np.float64)

            order = np.argsort(cand_probs)[::-1]
            top = cand_probs[order[0]]
            second = cand_probs[order[1]]
            margin = float(top - second)
            ent = entropy(cand_probs)
            top_letter = ["A", "B", "C", "D"][int(order[0])]

            logits_by_layer.append(cand_logits.tolist())
            probs_by_layer.append(cand_probs.tolist())
            margins.append(margin)
            entropies.append(ent)
            argmax_letters.append(top_letter)

            if top20:
                h_norm_l = all_h_norm[i, l:l+1, :]  # [1, d]
                vocab_logits_full = model.get_output_embeddings()(h_norm_l).squeeze(0)
                vals, ids = torch.topk(vocab_logits_full, k=20)
                toks = tokenizer.batch_decode(ids.unsqueeze(-1), skip_special_tokens=True)
                top20_by_layer.append([t.strip() for t in toks])

        first_tid = int(first_token_ids[i].item())
        first_decoded = tokenizer.decode([first_tid], skip_special_tokens=True)
        chosen = extract_choice(first_decoded)

        # PCA source vectors for this sample: [n_layers, d]
        vecs = all_h[i].detach().float().cpu().numpy()

        results_hidden.append({
            "input_tokens": int(prompt_lens[i].item()),
            "first_token_id": first_tid,
            "first_token_decoded": first_decoded,
            "chosen_letter": chosen,
            "layer_candidate_logits": logits_by_layer,
            "layer_candidate_probs": probs_by_layer,
            "layer_margin": margins,
            "layer_entropy": entropies,
            "layer_argmax": argmax_letters,
            "layer_hidden_vectors": vecs,
            "layer_top20_tokens": top20_by_layer if top20 else None,
        })

    del all_h, all_h_norm, all_union_logits

    # --- Pass 2: Batched generation (reusing KV-cache from Pass 1) ---
    remaining_tokens = response_max_new_tokens - 1  # Already have first token
    all_generated: List[List[int]] = [[int(first_token_ids[i].item())] for i in range(B)]

    if remaining_tokens > 0:
        eos_id = tokenizer.eos_token_id
        cur_ids = first_token_ids.unsqueeze(1)  # [B, 1]
        cur_mask = torch.cat([
            attention_mask,
            torch.ones((B, 1), dtype=attention_mask.dtype, device=device),
        ], dim=1)  # [B, max_seq_len + 1]
        kv = past_key_values
        done = torch.zeros(B, dtype=torch.bool, device=device)

        with torch.no_grad():
            for _ in range(remaining_tokens):
                step_position_ids = _compute_position_ids(cur_mask)[:, -1:]
                step_out = _model_forward_with_optional_position_ids(
                    model,
                    input_ids=cur_ids,
                    attention_mask=cur_mask,
                    past_key_values=kv,
                    position_ids=step_position_ids,
                    use_cache=True,
                    return_dict=True,
                )
                next_ids = torch.argmax(step_out.logits[:, -1, :], dim=-1)  # [B]
                kv = step_out.past_key_values

                # Check EOS per sample
                eos_mask = (next_ids == eos_id)
                done = done | eos_mask

                for j in range(B):
                    if not done[j]:
                        all_generated[j].append(int(next_ids[j].item()))
                    elif eos_mask[j] and len(all_generated[j]) > 0:
                        # EOS just hit this step — don't append it
                        pass

                if done.all():
                    break

                # For done samples, feed EOS to keep sequence aligned
                cur_ids = torch.where(
                    done.unsqueeze(1),
                    torch.full_like(next_ids.unsqueeze(1), eos_id),
                    next_ids.unsqueeze(1),
                )  # [B, 1]
                # Extend mask: active samples get 1, done samples get 0
                cur_mask = torch.cat([
                    cur_mask,
                    (~done).long().unsqueeze(1),
                ], dim=1)

        del kv
    del past_key_values

    # --- Build final results per sample ---
    batch_results: List[Dict] = []
    for i in range(B):
        gen_ids = all_generated[i]
        response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        response_token_pieces = [tokenizer.decode([int(tid)], skip_special_tokens=True) for tid in gen_ids]
        resp_diag = response_position_diagnostics(response_text, response_token_pieces)

        out = results_hidden[i]
        out["response_text"] = response_text
        out["response_token_ids"] = [int(x) for x in gen_ids]
        out["response_token_pieces"] = response_token_pieces
        out["response_diagnostics"] = resp_diag
        batch_results.append(out)

    return batch_results


def fit_pca(
    model,
    tokenizer,
    final_norm,
    option_buckets: Dict[str, List[int]],
    union_token_ids: List[int],
    union_W: torch.Tensor,
    union_b: Optional[torch.Tensor],
    option_union_indices: Dict[str, torch.Tensor],
    prompts: List[Dict],
    device: str,
    n_components: int,
    n_prompt_samples: int,
    inference_batch_size: int = 1,
    pca_fit_batch_size: int = 1024,
) -> Tuple[IncrementalPCA, int]:
    sample = prompts[: min(len(prompts), n_prompt_samples)]
    n_layers = int(getattr(model.config, "num_hidden_layers", 32))
    max_components_by_samples = max(2, len(sample) * n_layers)
    effective_components = int(min(n_components, max_components_by_samples))
    ipca = IncrementalPCA(n_components=effective_components)

    buffer = []
    # Process PCA warmup in batches for speed.
    for batch_start in tqdm(range(0, len(sample), inference_batch_size), desc="PCA warmup"):
        batch = sample[batch_start : batch_start + inference_batch_size]
        prompt_texts = [row["prompt_text"] for row in batch]

        if len(prompt_texts) == 1:
            # Single-prompt path (avoids padding overhead)
            out = run_forward(
                model=model,
                tokenizer=tokenizer,
                final_norm=final_norm,
                option_buckets=option_buckets,
                union_token_ids=union_token_ids,
                union_W=union_W,
                union_b=union_b,
                option_union_indices=option_union_indices,
                prompt=prompt_texts[0],
                device=device,
                top20=False,
                response_max_new_tokens=1,
            )
            buffer.append(out["layer_hidden_vectors"])
        else:
            batch_outs = run_forward_batch(
                model=model,
                tokenizer=tokenizer,
                final_norm=final_norm,
                option_buckets=option_buckets,
                union_token_ids=union_token_ids,
                union_W=union_W,
                union_b=union_b,
                option_union_indices=option_union_indices,
                prompts=prompt_texts,
                device=device,
                top20=False,
                response_max_new_tokens=1,
            )
            for out in batch_outs:
                buffer.append(out["layer_hidden_vectors"])

        total_rows = sum(x.shape[0] for x in buffer)
        if total_rows >= pca_fit_batch_size:
            mat = np.concatenate(buffer, axis=0)
            ipca.partial_fit(mat)
            buffer = []

    if buffer:
        mat = np.concatenate(buffer, axis=0)
        ipca.partial_fit(mat)

    return ipca, effective_components


def load_completed_prompt_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for row in iter_jsonl(path):
        pid = row.get("prompt_id")
        if pid:
            done.add(pid)
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="Single model inference + logging for Shape of Wisdom")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--meta-out", type=Path, required=True)
    parser.add_argument("--pca-out", type=Path, required=True)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--pca-sample-prompts", type=int, default=1000)
    parser.add_argument("--commit-thresholds", nargs="*", type=float, default=[0.1, 0.2, 0.3])
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference (>1 requires GPU)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top20-sanity", action="store_true")
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=0,
        help="(legacy) truncate manifest to first N prompts; prefer --max-new-prompts for chunking",
    )
    parser.add_argument(
        "--max-new-prompts",
        type=int,
        default=0,
        help="Process at most N pending prompts this invocation (resume-safe chunking)",
    )
    parser.add_argument(
        "--time-budget-seconds",
        type=int,
        default=0,
        help="Stop after ~N seconds of processing (checked between prompts)",
    )
    parser.add_argument(
        "--done-sentinel",
        type=Path,
        default=None,
        help="If set, write this file ONLY when all prompts are complete",
    )
    parser.add_argument("--response-max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--empty-cache-every",
        type=int,
        default=10,
        help="Force gc + backend cache clear every N prompts (0 disables)",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing output JSONL")
    args = parser.parse_args()

    device = detect_device(args.device)
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-padding is required for batched inference so the last token position
    # is always the meaningful end-of-prompt for each sample.
    tokenizer.padding_side = "left"

    load_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            dtype=dtype,
            **load_kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            **load_kwargs,
        )
    model.to(device)
    model.eval()

    final_norm = get_final_norm_module(model)

    # Canonical: option-token invariant buckets + logsumexp aggregation.
    option_buckets = build_option_token_buckets(tokenizer)
    option_buckets_fingerprint = buckets_sha256(option_buckets)
    union_token_ids, union_W, union_b, option_union_indices = _prepare_bucket_projection(model, device, option_buckets)

    # Legacy/meta-only: record tokenizer.encode("A") single-token IDs if available.
    legacy_candidate_token_ids: Dict[str, Optional[int]] = {}
    for letter in ["A", "B", "C", "D"]:
        try:
            ids = tokenizer.encode(letter, add_special_tokens=False)
            legacy_candidate_token_ids[letter] = int(ids[0]) if len(ids) == 1 else None
        except Exception:
            legacy_candidate_token_ids[letter] = None

    prompts = read_jsonl(args.prompts)
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    args.out.parent.mkdir(parents=True, exist_ok=True)

    completed_ids: set[str] = set()
    if args.resume and args.out.exists():
        completed_ids = load_completed_prompt_ids(args.out)
    elif args.out.exists():
        args.out.unlink()

    pending_prompts = [p for p in prompts if p.get("prompt_id") not in completed_ids]
    if args.max_new_prompts and args.max_new_prompts > 0:
        pending_prompts = pending_prompts[: args.max_new_prompts]

    # PCA (fit over layerwise hidden states).
    if args.resume and args.pca_out.exists():
        ipca = joblib.load(args.pca_out)
        pca_components_effective = int(getattr(ipca, "n_components", args.pca_components))
    else:
        ipca, pca_components_effective = fit_pca(
            model=model,
            tokenizer=tokenizer,
            final_norm=final_norm,
            option_buckets=option_buckets,
            union_token_ids=union_token_ids,
            union_W=union_W,
            union_b=union_b,
            option_union_indices=option_union_indices,
            prompts=prompts,
            device=device,
            n_components=args.pca_components,
            n_prompt_samples=args.pca_sample_prompts,
            inference_batch_size=args.batch_size,
        )
        args.pca_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(ipca, args.pca_out)

    if pca_components_effective != args.pca_components:
        print(
            f"[warn] pca_components reduced from {args.pca_components} to {pca_components_effective} due to sample size",
            flush=True,
        )

    started_ts = time.time()
    processed_this_run = 0
    batch_sz = max(1, args.batch_size)
    time_exceeded = False

    pbar = tqdm(total=len(pending_prompts), desc=f"Inference {args.model_name}")

    for batch_start in range(0, len(pending_prompts), batch_sz):
        if args.time_budget_seconds and args.time_budget_seconds > 0:
            if (time.time() - started_ts) >= float(args.time_budget_seconds):
                time_exceeded = True
                break

        batch_rows = pending_prompts[batch_start : batch_start + batch_sz]
        batch_prompts = [r["prompt_text"] for r in batch_rows]

        # Use single-prompt path for batch_size=1 (no padding overhead).
        if len(batch_prompts) == 1:
            batch_outs = [run_forward(
                model=model,
                tokenizer=tokenizer,
                final_norm=final_norm,
                option_buckets=option_buckets,
                union_token_ids=union_token_ids,
                union_W=union_W,
                union_b=union_b,
                option_union_indices=option_union_indices,
                prompt=batch_prompts[0],
                device=device,
                top20=args.top20_sanity,
                response_max_new_tokens=args.response_max_new_tokens,
            )]
        else:
            batch_outs = run_forward_batch(
                model=model,
                tokenizer=tokenizer,
                final_norm=final_norm,
                option_buckets=option_buckets,
                union_token_ids=union_token_ids,
                union_W=union_W,
                union_b=union_b,
                option_union_indices=option_union_indices,
                prompts=batch_prompts,
                device=device,
                top20=args.top20_sanity,
                response_max_new_tokens=args.response_max_new_tokens,
            )

        for row, out in zip(batch_rows, batch_outs):
            processed_this_run += 1

            layer_argmax = out["layer_argmax"]
            margins_out = out["layer_margin"]
            commit_layers = compute_commitment_layers(layer_argmax, margins_out, args.commit_thresholds)
            rev_count = reversal_count(layer_argmax)
            chosen_letter = out["chosen_letter"]
            correct = bool(chosen_letter == row["correct_key"])

            early_wrong_lock_in, late_correction, _ = classify_error_regimes(
                argmax_letters=layer_argmax,
                margins=margins_out,
                correct_key=row["correct_key"],
            )
            max_wrong = max_wrong_prob(out["layer_candidate_probs"], row["correct_key"])

            projected = ipca.transform(out["layer_hidden_vectors"])

            result = {
                "model_name": args.model_name,
                "model_id": args.model_id,
                "prompt_id": row["prompt_id"],
                "module": row["module"],
                "dataset": row["dataset"],
                "split": row["split"],
                "example_id": row["example_id"],
                "subject": row.get("subject"),
                "coarse_domain": row.get("coarse_domain"),
                "question": row.get("question"),
                "options": row.get("options"),
                "wrapper_id": row.get("wrapper_id"),
                "correct_key": row["correct_key"],
                "first_generated_token_id": out["first_token_id"],
                "first_generated_token_decoded": out["first_token_decoded"],
                "chosen_letter": chosen_letter,
                "is_correct": correct,
                "input_tokens": out["input_tokens"],
                "response_text": out["response_text"],
                "response_token_ids": out["response_token_ids"],
                "response_token_pieces": out["response_token_pieces"],
                "response_diagnostics": out["response_diagnostics"],
                "layer_candidate_logits": out["layer_candidate_logits"],
                "layer_candidate_probs": out["layer_candidate_probs"],
                "layer_margin": out["layer_margin"],
                "layer_entropy": out["layer_entropy"],
                "layer_argmax": out["layer_argmax"],
                "commitment_layer": commit_layers,
                "reversal_count": rev_count,
                "max_wrong_candidate_prob": max_wrong,
                "early_wrong_lock_in": early_wrong_lock_in,
                "late_correction": late_correction,
                "projected_hidden": projected.tolist(),
                "projected_hidden_128": projected.tolist(),  # backward-compatible key name
                "projected_dim": int(projected.shape[1]),
                "layer_top20_tokens": out["layer_top20_tokens"],
            }
            append_jsonl(args.out, result)

        pbar.update(len(batch_rows))

        # Periodic memory hygiene.
        batch_idx = batch_start // batch_sz + 1
        if args.empty_cache_every > 0 and (batch_idx % args.empty_cache_every == 0):
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()

    pbar.close()

    # Meta accuracy is over rows processed in this invocation + existing rows if resume.
    total_rows = list(iter_jsonl(args.out))
    overall_correct = sum(1 for r in total_rows if r.get("is_correct") is True)
    overall_n = len(total_rows)

    completed_all = bool(overall_n == len(prompts))

    meta = {
        "model_name": args.model_name,
        "model_id": args.model_id,
        "device": device,
        "dtype": str(dtype),
        "prompts_total_manifest": len(prompts),
        "prompts_completed": overall_n,
        "prompts_processed_this_run": int(processed_this_run),
        "accuracy_over_completed": overall_correct / max(1, overall_n),
        "commit_thresholds": args.commit_thresholds,
        "candidate_token_ids": legacy_candidate_token_ids,
        "candidate_scoring": {
            "kind": "option_bucket_logsumexp",
            "buckets": option_buckets,
            "buckets_sha256": option_buckets_fingerprint,
            "bucket_union_size": len(union_token_ids),
            "normalization": "NFKC + strip + conservative boundary punctuation strip + uppercase",
            "aggregation": "logsumexp over token logits per option bucket",
            "projection": "sliced lm_head over union(bucket token ids)",
        },
        "pca_components_requested": args.pca_components,
        "pca_components_effective": pca_components_effective,
        "pca_sample_prompts": args.pca_sample_prompts,
        "response_max_new_tokens": args.response_max_new_tokens,
        "empty_cache_every": args.empty_cache_every,
        "resume": bool(args.resume),
        "max_prompts": int(args.max_prompts),
        "max_new_prompts": int(args.max_new_prompts),
        "time_budget_seconds": int(args.time_budget_seconds),
        "completed_all": completed_all,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    args.meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if completed_all and args.done_sentinel is not None:
        args.done_sentinel.parent.mkdir(parents=True, exist_ok=True)
        args.done_sentinel.write_text(
            json.dumps(
                {"done": True, "generated_at_utc": meta["generated_at_utc"], "prompts_completed": overall_n},
                indent=2,
            ),
            encoding="utf-8",
        )

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
