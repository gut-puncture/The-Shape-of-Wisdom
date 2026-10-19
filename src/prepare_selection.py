from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable

from dataset_loader import CanonicalExample


@dataclass(frozen=True)
class SelectionRecord:
    dataset: str
    example_id: str
    source_row_index: int
    option_count: int
    question_length: int
    selection_reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _question_length(example: CanonicalExample) -> int:
    return len(example.question_raw)


def _sorted_by_length(examples: Iterable[CanonicalExample]) -> list[CanonicalExample]:
    return sorted(examples, key=lambda example: (_question_length(example), example.source_row_index))


def _record_for(example: CanonicalExample, *, selection_reason: str) -> SelectionRecord:
    return SelectionRecord(
        dataset=example.dataset_name,
        example_id=example.example_id,
        source_row_index=example.source_row_index,
        option_count=len(example.options),
        question_length=_question_length(example),
        selection_reason=selection_reason,
    )


def selection_records_sha256(records: Iterable[SelectionRecord]) -> str:
    payload = json.dumps(
        [record.to_dict() for record in records],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_selection_manifest(
    *,
    run_id: str,
    panel_name: str,
    selection_strategy: str,
    records: list[SelectionRecord],
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "panel_name": panel_name,
        "selection_strategy": selection_strategy,
        "selection_manifest_sha256": selection_records_sha256(records),
        "selections": [record.to_dict() for record in records],
    }


def select_examples_for_dataset(
    examples: list[CanonicalExample],
    *,
    max_examples: int,
    strategy: str,
) -> tuple[list[CanonicalExample], list[SelectionRecord]]:
    if max_examples < 0:
        raise ValueError(f"max_examples must be nonnegative, got {max_examples}")
    if strategy not in {"first_n", "representative_pilot_v1", "representative_panel_v1"}:
        raise ValueError(f"unsupported selection strategy: {strategy}")
    if max_examples == 0 or len(examples) <= max_examples:
        selected = list(examples if max_examples == 0 else examples[:max_examples])
        reason = "all_examples" if max_examples == 0 else "dataset_exhausted"
        records = [_record_for(example, selection_reason=reason) for example in selected]
        return selected, records
    if strategy == "first_n":
        selected = list(examples[:max_examples])
        records = [
            _record_for(example, selection_reason=f"first_n_position:{index + 1}")
            for index, example in enumerate(selected)
        ]
        return selected, records
    return _select_representative_panel_v1(examples, max_examples=max_examples, strategy=strategy)


def _select_representative_panel_v1(
    examples: list[CanonicalExample],
    *,
    max_examples: int,
    strategy: str,
) -> tuple[list[CanonicalExample], list[SelectionRecord]]:
    groups: dict[int, list[CanonicalExample]] = {}
    for example in examples:
        groups.setdefault(len(example.options), []).append(example)
    if len(groups) > max_examples:
        raise ValueError(
            f"{strategy} cannot satisfy one-example-per-option-count coverage "
            f"when option_count_groups={sorted(groups)} exceeds max_examples={max_examples}"
        )

    ordered_groups = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    majority_option_count = ordered_groups[0][0]
    non_majority_option_counts = [option_count for option_count, _ in ordered_groups[1:]]

    selected_by_id: dict[str, CanonicalExample] = {}
    record_by_id: dict[str, SelectionRecord] = {}
    for option_count in non_majority_option_counts:
        group = _sorted_by_length(groups[option_count])
        chosen = group[len(group) // 2]
        selected_by_id[chosen.example_id] = chosen
        record_by_id[chosen.example_id] = _record_for(
            chosen,
            selection_reason=f"non_majority_option_count_median:{option_count}",
        )

    remaining_slots = max_examples - len(selected_by_id)
    if remaining_slots < 0:
        raise ValueError(f"{strategy} over-selected before majority fill")

    majority_group = _sorted_by_length(groups[majority_option_count])
    for quantile_rank, index in enumerate(_evenly_spaced_midpoint_indices(len(majority_group), remaining_slots), start=1):
        chosen = majority_group[index]
        selected_by_id[chosen.example_id] = chosen
        record_by_id[chosen.example_id] = _record_for(
            chosen,
            selection_reason=(
                "majority_option_count_quantile:"
                f"{majority_option_count}:{quantile_rank}_of_{remaining_slots}"
            ),
        )

    selected = sorted(selected_by_id.values(), key=lambda example: example.source_row_index)
    records = [record_by_id[example.example_id] for example in selected]
    if len(selected) != max_examples:
        raise ValueError(
            f"{strategy} selected an unexpected number of examples: "
            f"expected={max_examples} actual={len(selected)}"
        )
    return selected, records


def _evenly_spaced_midpoint_indices(total_count: int, picks: int) -> list[int]:
    if picks < 0:
        raise ValueError(f"picks must be nonnegative, got {picks}")
    if picks == 0:
        return []
    if total_count <= 0:
        raise ValueError("total_count must be positive when picks > 0")
    if picks >= total_count:
        return list(range(total_count))
    return [min(total_count - 1, ((2 * rank + 1) * total_count) // (2 * picks)) for rank in range(picks)]
