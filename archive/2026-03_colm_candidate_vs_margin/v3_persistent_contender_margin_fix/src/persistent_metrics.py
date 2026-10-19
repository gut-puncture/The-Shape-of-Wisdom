from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LayerSnapshot:
    position: int
    layer_index: int
    scores: dict[str, float]
    ranking: list[str]
    top_score_tie: bool
    any_score_tie: bool


@dataclass(frozen=True)
class ExampleMetrics:
    model_name: str
    model_family: str
    model_size: str
    dataset_name: str
    example_id: str
    cache_key: str
    cache_relpath: str
    option_count: int
    n_layers: int
    final_winner: str
    bundle_final_winner: str | None
    final_winner_mismatch: bool
    final_layer_top_tie: bool
    intermediate_top_tie_count: int
    intermediate_any_tie_count: int
    nonmonotonic_layer_index: bool
    missing_option_scores: bool
    missing_options_any_layer: bool
    g_final: float | None
    normalized_gap_valid: bool
    d2_pos: int | None
    d1_pos: int | None
    d25_pos: int | None
    d50_pos: int | None
    d75_pos: int | None
    d2_layer: int | None
    d1_layer: int | None
    d25_layer: int | None
    d50_layer: int | None
    d75_layer: int | None
    delta_21_pos: int | None
    delta_150_pos: int | None
    delta_250_pos: int | None
    delta_225_pos: int | None
    delta_175_pos: int | None
    delta_275_pos: int | None
    delta_21_layer: int | None
    delta_150_layer: int | None
    delta_250_layer: int | None
    delta_225_layer: int | None
    delta_175_layer: int | None
    delta_275_layer: int | None
    norm_d2: float | None
    norm_d1: float | None
    norm_d25: float | None
    norm_d50: float | None
    norm_d75: float | None
    norm_delta_21: float | None
    norm_delta_150: float | None
    norm_delta_250: float | None
    norm_delta_225: float | None
    norm_delta_175: float | None
    norm_delta_275: float | None
    total_positive_write: float | None
    total_positive_write_nonpositive: bool
    write_share_after_d2: float | None
    write_share_after_d1: float | None
    write_share_last20: float | None
    write_center_of_mass: float | None
    only_final_layer_d2: bool
    only_final_layer_d1: bool
    only_final_layer_d25: bool
    only_final_layer_d50: bool
    only_final_layer_d75: bool
    gap_negative_majority: bool


class BundleValidationError(ValueError):
    pass


def _canonical_rank_order(
    option_ids: list[str],
    layer_scores: dict[str, float],
) -> list[str]:
    order = {option_id: ordinal for ordinal, option_id in enumerate(option_ids)}
    return sorted(option_ids, key=lambda option_id: (-layer_scores[option_id], order[option_id]))


def _has_any_tie(option_ids: list[str], layer_scores: dict[str, float]) -> bool:
    seen: set[float] = set()
    for option_id in option_ids:
        score = layer_scores[option_id]
        if score in seen:
            return True
        seen.add(score)
    return False


def _has_top_tie(option_ids: list[str], layer_scores: dict[str, float]) -> bool:
    scores = [layer_scores[option_id] for option_id in option_ids]
    if not scores:
        return False
    top = max(scores)
    return sum(1 for score in scores if score == top) > 1


def _layer_gap(layer_scores: dict[str, float], winner: str, ranking: list[str]) -> float | None:
    if winner not in layer_scores:
        return None
    runner_up = next((option_id for option_id in ranking if option_id != winner), None)
    if runner_up is None:
        return None
    return layer_scores[winner] - layer_scores[runner_up]


def _first_persistent_position(
    values: list[Any],
    predicate,
) -> int | None:
    for idx in range(len(values)):
        if all(predicate(value) for value in values[idx:]):
            return idx
    return None


def _normalize_depth(position: int | None, n_layers: int) -> float | None:
    if position is None:
        return None
    if n_layers <= 1:
        return 0.0
    return position / float(n_layers - 1)


def _layer_value(position: int | None, layers: list[LayerSnapshot]) -> int | None:
    if position is None:
        return None
    return layers[position].layer_index


def _delta(lhs: int | None, rhs: int | None) -> int | None:
    if lhs is None or rhs is None:
        return None
    return lhs - rhs


def _share_after_position(
    positive_writes: list[float],
    target_norm_depths: list[float],
    total_positive_write: float,
    threshold_position: int | None,
    n_layers: int,
) -> float | None:
    if total_positive_write <= 0:
        return None
    if threshold_position is None:
        return None
    if n_layers <= 1:
        return 0.0
    threshold_norm = threshold_position / float(n_layers - 1)
    subtotal = sum(
        value
        for value, norm_depth in zip(positive_writes, target_norm_depths, strict=True)
        if norm_depth >= threshold_norm
    )
    return subtotal / total_positive_write


def _share_last20(
    positive_writes: list[float],
    target_norm_depths: list[float],
    total_positive_write: float,
) -> float | None:
    if total_positive_write <= 0:
        return None
    subtotal = sum(
        value
        for value, norm_depth in zip(positive_writes, target_norm_depths, strict=True)
        if norm_depth >= 0.8
    )
    return subtotal / total_positive_write


def _write_center_of_mass(
    positive_writes: list[float],
    target_norm_depths: list[float],
    total_positive_write: float,
) -> float | None:
    if total_positive_write <= 0:
        return None
    weighted = sum(
        value * norm_depth
        for value, norm_depth in zip(positive_writes, target_norm_depths, strict=True)
    )
    return weighted / total_positive_write


def _normalize_gap_series(gaps: list[float | None], g_final: float | None) -> list[float | None]:
    if g_final is None or g_final <= 0:
        return [None for _ in gaps]
    return [None if gap is None else gap / g_final for gap in gaps]


def build_layer_snapshots(
    bundle: dict[str, Any],
    *,
    option_ids: list[str],
) -> tuple[list[LayerSnapshot], dict[str, bool]]:
    layers_raw = list(bundle.get("layers", []))
    if not layers_raw:
        raise BundleValidationError("bundle contains no layers")

    layer_snapshots: list[LayerSnapshot] = []
    nonmonotonic_layer_index = False
    missing_option_scores = False
    missing_options_any_layer = False
    previous_layer_index: int | None = None

    for position, layer in enumerate(layers_raw):
        layer_index = int(layer["layer_index"])
        if previous_layer_index is not None and layer_index <= previous_layer_index:
            nonmonotonic_layer_index = True
        previous_layer_index = layer_index

        option_block = dict(layer.get("options", {}))
        if set(option_block.keys()) != set(option_ids):
            missing_options_any_layer = True
        layer_scores: dict[str, float] = {}
        for option_id in option_ids:
            if option_id not in option_block:
                missing_option_scores = True
                continue
            score_record = option_block[option_id]
            if "normalized_probability" not in score_record:
                missing_option_scores = True
                continue
            layer_scores[option_id] = float(score_record["normalized_probability"])

        if len(layer_scores) != len(option_ids):
            continue

        ranking = _canonical_rank_order(option_ids, layer_scores)
        layer_snapshots.append(
            LayerSnapshot(
                position=position,
                layer_index=layer_index,
                scores=layer_scores,
                ranking=ranking,
                top_score_tie=_has_top_tie(option_ids, layer_scores),
                any_score_tie=_has_any_tie(option_ids, layer_scores),
            )
        )

    flags = {
        "nonmonotonic_layer_index": nonmonotonic_layer_index,
        "missing_option_scores": missing_option_scores,
        "missing_options_any_layer": missing_options_any_layer,
    }
    if len(layer_snapshots) != len(layers_raw):
        flags["missing_option_scores"] = True
    return layer_snapshots, flags


def compute_example_metrics(
    *,
    bundle: dict[str, Any],
    cache_key: str,
    cache_relpath: str,
    canonical_option_ids: list[str],
) -> ExampleMetrics:
    if not canonical_option_ids:
        raise BundleValidationError("canonical option order is empty")

    layers, flags = build_layer_snapshots(bundle, option_ids=canonical_option_ids)
    if not layers:
        raise BundleValidationError("no complete layer snapshots after validation")

    final_layer = layers[-1]
    final_winner = final_layer.ranking[0]
    bundle_final_winner = bundle.get("final_winner")
    final_winner_mismatch = bundle_final_winner is not None and str(bundle_final_winner) != final_winner

    ranks: list[int] = []
    gaps: list[float | None] = []
    for layer in layers:
        rank = layer.ranking.index(final_winner) + 1
        ranks.append(rank)
        gaps.append(_layer_gap(layer.scores, final_winner, layer.ranking))

    g_final = gaps[-1]
    normalized_gaps = _normalize_gap_series(gaps, g_final)
    normalized_gap_valid = g_final is not None and g_final > 0

    d2_pos = _first_persistent_position(ranks, lambda rank: rank <= 2)
    d1_pos = _first_persistent_position(ranks, lambda rank: rank == 1)
    d25_pos = _first_persistent_position(normalized_gaps, lambda gap: gap is not None and gap >= 0.25)
    d50_pos = _first_persistent_position(normalized_gaps, lambda gap: gap is not None and gap >= 0.50)
    d75_pos = _first_persistent_position(normalized_gaps, lambda gap: gap is not None and gap >= 0.75)

    positive_writes: list[float] = []
    target_norm_depths: list[float] = []
    if len(gaps) > 1:
        for idx in range(1, len(gaps)):
            previous = gaps[idx - 1]
            current = gaps[idx]
            if previous is None or current is None:
                positive_writes.append(0.0)
            else:
                positive_writes.append(max(current - previous, 0.0))
            target_norm_depths.append(_normalize_depth(idx, len(layers)) or 0.0)

    total_positive_write = sum(positive_writes) if positive_writes else 0.0
    total_positive_write_nonpositive = total_positive_write <= 0

    gap_negative_majority = sum(1 for gap in gaps if gap is not None and gap < 0) > (len(gaps) / 2.0)

    return ExampleMetrics(
        model_name=str(bundle["model_name"]),
        model_family=str(bundle["model_family"]),
        model_size=str(bundle["model_size"]),
        dataset_name=str(bundle["dataset_name"]),
        example_id=str(bundle["example_id"]),
        cache_key=cache_key,
        cache_relpath=cache_relpath,
        option_count=len(canonical_option_ids),
        n_layers=len(layers),
        final_winner=final_winner,
        bundle_final_winner=(None if bundle_final_winner is None else str(bundle_final_winner)),
        final_winner_mismatch=final_winner_mismatch,
        final_layer_top_tie=final_layer.top_score_tie,
        intermediate_top_tie_count=sum(1 for layer in layers[:-1] if layer.top_score_tie),
        intermediate_any_tie_count=sum(1 for layer in layers[:-1] if layer.any_score_tie),
        nonmonotonic_layer_index=flags["nonmonotonic_layer_index"],
        missing_option_scores=flags["missing_option_scores"],
        missing_options_any_layer=flags["missing_options_any_layer"],
        g_final=g_final,
        normalized_gap_valid=normalized_gap_valid,
        d2_pos=d2_pos,
        d1_pos=d1_pos,
        d25_pos=d25_pos,
        d50_pos=d50_pos,
        d75_pos=d75_pos,
        d2_layer=_layer_value(d2_pos, layers),
        d1_layer=_layer_value(d1_pos, layers),
        d25_layer=_layer_value(d25_pos, layers),
        d50_layer=_layer_value(d50_pos, layers),
        d75_layer=_layer_value(d75_pos, layers),
        delta_21_pos=_delta(d1_pos, d2_pos),
        delta_150_pos=_delta(d50_pos, d1_pos),
        delta_250_pos=_delta(d50_pos, d2_pos),
        delta_225_pos=_delta(d25_pos, d2_pos),
        delta_175_pos=_delta(d75_pos, d1_pos),
        delta_275_pos=_delta(d75_pos, d2_pos),
        delta_21_layer=_delta(_layer_value(d1_pos, layers), _layer_value(d2_pos, layers)),
        delta_150_layer=_delta(_layer_value(d50_pos, layers), _layer_value(d1_pos, layers)),
        delta_250_layer=_delta(_layer_value(d50_pos, layers), _layer_value(d2_pos, layers)),
        delta_225_layer=_delta(_layer_value(d25_pos, layers), _layer_value(d2_pos, layers)),
        delta_175_layer=_delta(_layer_value(d75_pos, layers), _layer_value(d1_pos, layers)),
        delta_275_layer=_delta(_layer_value(d75_pos, layers), _layer_value(d2_pos, layers)),
        norm_d2=_normalize_depth(d2_pos, len(layers)),
        norm_d1=_normalize_depth(d1_pos, len(layers)),
        norm_d25=_normalize_depth(d25_pos, len(layers)),
        norm_d50=_normalize_depth(d50_pos, len(layers)),
        norm_d75=_normalize_depth(d75_pos, len(layers)),
        norm_delta_21=(None if d2_pos is None or d1_pos is None else _normalize_depth(d1_pos, len(layers)) - _normalize_depth(d2_pos, len(layers))),
        norm_delta_150=(None if d1_pos is None or d50_pos is None else _normalize_depth(d50_pos, len(layers)) - _normalize_depth(d1_pos, len(layers))),
        norm_delta_250=(None if d2_pos is None or d50_pos is None else _normalize_depth(d50_pos, len(layers)) - _normalize_depth(d2_pos, len(layers))),
        norm_delta_225=(None if d2_pos is None or d25_pos is None else _normalize_depth(d25_pos, len(layers)) - _normalize_depth(d2_pos, len(layers))),
        norm_delta_175=(None if d1_pos is None or d75_pos is None else _normalize_depth(d75_pos, len(layers)) - _normalize_depth(d1_pos, len(layers))),
        norm_delta_275=(None if d2_pos is None or d75_pos is None else _normalize_depth(d75_pos, len(layers)) - _normalize_depth(d2_pos, len(layers))),
        total_positive_write=(None if total_positive_write_nonpositive else total_positive_write),
        total_positive_write_nonpositive=total_positive_write_nonpositive,
        write_share_after_d2=_share_after_position(
            positive_writes,
            target_norm_depths,
            total_positive_write,
            d2_pos,
            len(layers),
        ),
        write_share_after_d1=_share_after_position(
            positive_writes,
            target_norm_depths,
            total_positive_write,
            d1_pos,
            len(layers),
        ),
        write_share_last20=_share_last20(positive_writes, target_norm_depths, total_positive_write),
        write_center_of_mass=_write_center_of_mass(positive_writes, target_norm_depths, total_positive_write),
        only_final_layer_d2=(d2_pos == len(layers) - 1),
        only_final_layer_d1=(d1_pos == len(layers) - 1),
        only_final_layer_d25=(d25_pos == len(layers) - 1),
        only_final_layer_d50=(d50_pos == len(layers) - 1),
        only_final_layer_d75=(d75_pos == len(layers) - 1),
        gap_negative_majority=gap_negative_majority,
    )


def metrics_to_row(metrics: ExampleMetrics) -> dict[str, Any]:
    return metrics.__dict__.copy()
