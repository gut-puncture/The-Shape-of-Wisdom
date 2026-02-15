from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class BucketIndex:
    union_token_ids: List[int]
    indices_by_letter: Dict[str, List[int]]


def build_bucket_index(*, buckets: Dict[str, List[int]]) -> BucketIndex:
    """
    Precompute a stable union of bucket token IDs and per-letter indices into that union.
    """
    for k in ["A", "B", "C", "D"]:
        if k not in buckets or not isinstance(buckets[k], list) or not buckets[k]:
            raise ValueError(f"missing/non-empty bucket for {k}")
        if any((not isinstance(x, int)) for x in buckets[k]):
            raise ValueError(f"bucket {k} must contain ints")

    union = sorted(set(buckets["A"]) | set(buckets["B"]) | set(buckets["C"]) | set(buckets["D"]))
    pos = {tid: i for i, tid in enumerate(union)}
    idxs = {k: sorted({pos[int(t)] for t in buckets[k] if int(t) in pos}) for k in ["A", "B", "C", "D"]}
    for k in ["A", "B", "C", "D"]:
        if not idxs[k]:
            raise ValueError(f"empty bucket index for {k} (after union)")
    return BucketIndex(union_token_ids=union, indices_by_letter=idxs)


def _entropy_nats(probs: Any) -> Any:
    import torch  # noqa: PLC0415

    eps = 1e-12
    p = torch.clamp(probs, min=float(eps), max=1.0)
    return -(p * torch.log(p)).sum(dim=-1)


def compute_candidate_readout(
    *,
    hidden_last: Any,
    final_norm: Any,
    lm_head: Any,
    bucket_index: BucketIndex,
) -> Dict[str, Any]:
    """
    Compute option-token invariant candidate evidence from per-layer hidden states at decision position p.

    Inputs:
      hidden_last: torch.Tensor (batch, n_layers, hidden_dim)
    Returns:
      dict of torch tensors on the same device:
        - candidate_logits: (batch, n_layers, 4) float32
        - candidate_probs: (batch, n_layers, 4) float32
        - candidate_entropy: (batch, n_layers) float32
        - top_candidate_idx: (batch, n_layers) int64 in [0..3]
        - top2_margin_prob: (batch, n_layers) float32
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    if hidden_last.ndim != 3:
        raise ValueError("hidden_last must be 3D (batch, n_layers, hidden_dim)")
    bsz, n_layers, hidden_dim = (int(hidden_last.shape[0]), int(hidden_last.shape[1]), int(hidden_last.shape[2]))

    x = hidden_last.reshape(bsz * n_layers, hidden_dim).to(dtype=torch.float32)
    x = final_norm(x)

    # Gather only the union token weights (much cheaper than full vocab projection).
    union_ids = torch.tensor(bucket_index.union_token_ids, dtype=torch.long, device=x.device)
    w = lm_head.weight.index_select(0, union_ids).to(dtype=torch.float32)
    b = getattr(lm_head, "bias", None)
    b_sub = b.index_select(0, union_ids).to(dtype=torch.float32) if b is not None else None

    logits_sub = F.linear(x, w, b_sub)  # (bsz*n_layers, U)

    # Aggregate per option via logsumexp over bucket columns.
    out_logits = []
    for k in ["A", "B", "C", "D"]:
        idxs = torch.tensor(bucket_index.indices_by_letter[k], dtype=torch.long, device=x.device)
        v = logits_sub.index_select(1, idxs)
        out_logits.append(torch.logsumexp(v, dim=1))
    cand_logits = torch.stack(out_logits, dim=1)  # (bsz*n_layers, 4)
    cand_probs = torch.softmax(cand_logits, dim=1)
    ent = _entropy_nats(cand_probs)

    # Top candidate + top2 margin in probability space.
    top_idx = torch.argmax(cand_probs, dim=1)  # (bsz*n_layers,)
    top2 = torch.topk(cand_probs, k=2, dim=1).values  # (bsz*n_layers,2)
    margin = top2[:, 0] - top2[:, 1]

    return {
        "candidate_logits": cand_logits.reshape(bsz, n_layers, 4),
        "candidate_probs": cand_probs.reshape(bsz, n_layers, 4),
        "candidate_entropy": ent.reshape(bsz, n_layers),
        "top_candidate_idx": top_idx.reshape(bsz, n_layers),
        "top2_margin_prob": margin.reshape(bsz, n_layers),
    }


def project_hidden_pca(
    *,
    hidden_last: Any,
    mean: Any,
    components: Any,
) -> Any:
    """
    Project per-layer hidden states into the frozen PCA basis.

    Inputs:
      hidden_last: torch.Tensor (batch, n_layers, hidden_dim)
      mean: torch.Tensor (hidden_dim,)
      components: torch.Tensor (n_components, hidden_dim)
    Returns:
      torch.Tensor (batch, n_layers, n_components) float32
    """
    import torch  # noqa: PLC0415

    if hidden_last.ndim != 3:
        raise ValueError("hidden_last must be 3D")
    if mean.ndim != 1:
        raise ValueError("mean must be 1D")
    if components.ndim != 2:
        raise ValueError("components must be 2D")

    bsz, n_layers, hidden_dim = (int(hidden_last.shape[0]), int(hidden_last.shape[1]), int(hidden_last.shape[2]))
    if int(mean.shape[0]) != hidden_dim:
        raise ValueError("mean dim mismatch")
    if int(components.shape[1]) != hidden_dim:
        raise ValueError("components dim mismatch")

    x = hidden_last.reshape(bsz * n_layers, hidden_dim).to(dtype=torch.float32)
    x = x - mean.to(device=x.device, dtype=torch.float32)
    comps_t = components.to(device=x.device, dtype=torch.float32).T  # (hidden_dim, n_components)
    y = x @ comps_t
    return y.reshape(bsz, n_layers, int(components.shape[0]))


def idx_to_letter(idx: int) -> str:
    return ["A", "B", "C", "D"][int(idx)]

