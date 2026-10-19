from __future__ import annotations

from typing import Any


def _sorted_options(layer_options: dict[str, dict[str, float]], *, tie_eps: float) -> list[tuple[str, dict[str, float]]]:
    items = sorted(
        layer_options.items(),
        key=lambda item: (-float(item[1]["normalized_probability"]), item[0]),
    )
    return items


def _has_top_tie(layer_options: dict[str, dict[str, float]], *, tie_eps: float) -> bool:
    items = _sorted_options(layer_options, tie_eps=tie_eps)
    if len(items) < 2:
        return False
    p0 = float(items[0][1]["normalized_probability"])
    p1 = float(items[1][1]["normalized_probability"])
    return abs(p0 - p1) <= tie_eps


def candidate_set_for_layer(layer_options: dict[str, dict[str, float]], *, tau: float, tie_eps: float) -> list[str] | None:
    ordered = _sorted_options(layer_options, tie_eps=tie_eps)
    cumulative = 0.0
    selected: list[str] = []
    for idx, (option_id, stats) in enumerate(ordered):
        probability = float(stats["normalized_probability"])
        selected.append(option_id)
        cumulative += probability
        if cumulative + 1e-15 >= tau:
            if idx + 1 < len(ordered):
                current = probability
                next_probability = float(ordered[idx + 1][1]["normalized_probability"])
                if abs(current - next_probability) <= tie_eps:
                    return None
            return selected
    return selected


def stable_top1_depth(layers: list[dict[str, Any]], *, anchor_option: str, tie_eps: float) -> int | None:
    for idx in range(len(layers)):
        valid = True
        for later in layers[idx:]:
            options = later["options"]
            if _has_top_tie(options, tie_eps=tie_eps):
                valid = False
                break
            winner = _sorted_options(options, tie_eps=tie_eps)[0][0]
            if winner != anchor_option:
                valid = False
                break
        if valid:
            return idx
    return None


def candidate_entry_depth(layers: list[dict[str, Any]], *, anchor_option: str, tau: float, tie_eps: float) -> int | None:
    for idx, layer in enumerate(layers):
        candidate_set = candidate_set_for_layer(layer["options"], tau=tau, tie_eps=tie_eps)
        if candidate_set is None:
            continue
        if anchor_option in candidate_set:
            return idx
    return None


def margin_isolation_depth(layers: list[dict[str, Any]], *, anchor_option: str, delta: float, tie_eps: float) -> int | None:
    for idx in range(len(layers)):
        valid = True
        for later in layers[idx:]:
            ordered = _sorted_options(later["options"], tie_eps=tie_eps)
            if not ordered or ordered[0][0] != anchor_option:
                valid = False
                break
            if len(ordered) < 2:
                continue
            gap = float(ordered[0][1]["normalized_probability"]) - float(ordered[1][1]["normalized_probability"])
            if gap + tie_eps < delta:
                valid = False
                break
        if valid:
            return idx
    return None


def write_profile(layers: list[dict[str, Any]], *, winner: str) -> list[float]:
    contrasts = []
    for layer in layers:
        ordered = _sorted_options(layer["options"], tie_eps=0.0)
        runner_prob = 0.0
        for option_id, stats in ordered:
            if option_id != winner:
                runner_prob = float(stats["normalized_probability"])
                break
        winner_prob = float(layer["options"][winner]["normalized_probability"])
        contrasts.append(winner_prob - runner_prob)
    return contrasts


def _positive_increments(values: list[float]) -> list[float]:
    if not values:
        return []
    out = []
    last = values[0]
    for value in values[1:]:
        out.append(max(0.0, value - last))
        last = value
    return out


def _center_of_mass(increments: list[float]) -> float | None:
    if not increments:
        return None
    total = sum(increments)
    if total <= 0:
        return None
    weights = sum(idx * value for idx, value in enumerate(increments, start=1))
    return weights / total


def _last_fraction(increments: list[float], *, fraction: float) -> float | None:
    if not increments:
        return None
    total = sum(increments)
    if total <= 0:
        return None
    start = int(len(increments) * (1.0 - fraction))
    return sum(increments[start:]) / total


def compute_anchor_metrics(
    layers: list[dict[str, Any]],
    *,
    anchor_option: str,
    tau: float,
    delta: float,
    tie_eps: float,
) -> dict[str, Any]:
    d_id = candidate_entry_depth(layers, anchor_option=anchor_option, tau=tau, tie_eps=tie_eps)
    d_top1 = stable_top1_depth(layers, anchor_option=anchor_option, tie_eps=tie_eps)
    d_margin = margin_isolation_depth(layers, anchor_option=anchor_option, delta=delta, tie_eps=tie_eps)
    profile = write_profile(layers, winner=anchor_option)
    increments = _positive_increments(profile)
    return {
        "anchor_option": anchor_option,
        "d_id": d_id,
        "d_top1": d_top1,
        "d_margin": d_margin,
        "delta_top1": (None if d_id is None or d_top1 is None else d_top1 - d_id),
        "delta_margin": (None if d_id is None or d_margin is None else d_margin - d_id),
        "winner_runner_contrast_by_layer": profile,
        "positive_write_increments": increments,
        "write_center_of_mass": _center_of_mass(increments),
        "late_write_mass_last_20pct": _last_fraction(increments, fraction=0.2),
        "final_contrast": (profile[-1] if profile else None),
    }


def compute_primary_metrics(
    score_bundle: dict[str, Any],
    *,
    gold_answer: str | None,
    tau: float,
    delta: float,
    tie_eps: float,
) -> dict[str, Any]:
    layers = list(score_bundle["layers"])
    final_winner = str(score_bundle["final_winner"])
    metrics = {
        "final_winner_anchor": compute_anchor_metrics(layers, anchor_option=final_winner, tau=tau, delta=delta, tie_eps=tie_eps),
    }
    if gold_answer is not None:
        metrics["gold_answer_anchor"] = compute_anchor_metrics(
            layers,
            anchor_option=gold_answer,
            tau=tau,
            delta=delta,
            tie_eps=tie_eps,
        )
    return metrics

