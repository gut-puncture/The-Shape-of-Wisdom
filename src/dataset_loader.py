from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from dataset_registry import DatasetSpec


OPTION_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class CanonicalOption:
    option_id: str
    ordinal: int
    raw_text: str
    canonical_text: str


@dataclass(frozen=True)
class CanonicalExample:
    example_id: str
    dataset_name: str
    dataset_id: str
    dataset_revision: str
    split: str
    source_row_index: int
    question_raw: str
    question_canonical: str
    options: list[CanonicalOption]
    answer_key_raw: str
    answer_option_id: str | None
    metadata: dict[str, Any]
    anomaly_flags: list[str]
    exclusion_reason: str | None
    raw_record: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["options"] = [asdict(option) for option in self.options]
        return payload


def _text_or_empty(value: Any) -> str:
    return "" if value is None else str(value)


def _canonical_text(raw: str) -> str:
    return str(raw)


def _label_for_index(index: int) -> str:
    if index < 0 or index >= len(OPTION_LABELS):
        raise ValueError(f"unsupported option index: {index}")
    return OPTION_LABELS[index]


def _options_from_list(values: list[Any]) -> tuple[list[CanonicalOption], list[str]]:
    flags: list[str] = []
    options: list[CanonicalOption] = []
    seen = set()
    for idx, value in enumerate(values):
        raw = _text_or_empty(value)
        canonical = _canonical_text(raw)
        if not raw:
            flags.append("empty_option_text")
        if raw.strip() == "":
            flags.append("blank_option_text")
        if raw in seen:
            flags.append("duplicate_option_text")
        seen.add(raw)
        options.append(
            CanonicalOption(
                option_id=_label_for_index(idx),
                ordinal=idx,
                raw_text=raw,
                canonical_text=canonical,
            )
        )
    if len(options) != 4:
        flags.append(f"variable_option_count:{len(options)}")
    return options, sorted(set(flags))


def _normalize_answer_key(raw_key: Any, options: list[CanonicalOption], labels: list[str] | None = None) -> str | None:
    if raw_key is None:
        return None
    if isinstance(raw_key, int):
        if 0 <= raw_key < len(options):
            return options[raw_key].option_id
        return None
    raw = str(raw_key).strip()
    if not raw:
        return None
    upper = raw.upper()
    label_pool = labels or [opt.option_id for opt in options]
    if upper in label_pool:
        return upper
    if raw.isdigit():
        idx = int(raw)
        if 0 <= idx < len(options):
            return options[idx].option_id
    return None


def canonicalize_mmlu_row(row: dict[str, Any], *, spec: DatasetSpec, source_row_index: int) -> CanonicalExample:
    options, flags = _options_from_list(list(row.get("choices") or []))
    answer_option_id = _normalize_answer_key(row.get("answer"), options)
    exclusion_reason = None if answer_option_id else "unresolved_answer_key"
    return CanonicalExample(
        example_id=f"{spec.name}:{spec.split}:{source_row_index}",
        dataset_name=spec.name,
        dataset_id=spec.dataset_id,
        dataset_revision=spec.revision,
        split=spec.split,
        source_row_index=source_row_index,
        question_raw=_text_or_empty(row.get("question")),
        question_canonical=_canonical_text(_text_or_empty(row.get("question"))),
        options=options,
        answer_key_raw=_text_or_empty(row.get("answer")),
        answer_option_id=answer_option_id,
        metadata={"subject": row.get("subject")},
        anomaly_flags=flags,
        exclusion_reason=exclusion_reason,
        raw_record=dict(row),
    )


def canonicalize_arc_row(row: dict[str, Any], *, spec: DatasetSpec, source_row_index: int) -> CanonicalExample:
    choice_block = row.get("choices") or {}
    option_values = list(choice_block.get("text") or [])
    option_labels = [str(x).upper() for x in list(choice_block.get("label") or [])]
    options, flags = _options_from_list(option_values)
    answer_option_id = _normalize_answer_key(row.get("answerKey"), options, labels=option_labels or None)
    if option_labels and option_labels != [opt.option_id for opt in options]:
        flags = sorted(set(flags + ["noncanonical_option_labels"]))
    exclusion_reason = None if answer_option_id else "unresolved_answer_key"
    return CanonicalExample(
        example_id=f"{spec.name}:{spec.split}:{_text_or_empty(row.get('id')) or source_row_index}",
        dataset_name=spec.name,
        dataset_id=spec.dataset_id,
        dataset_revision=spec.revision,
        split=spec.split,
        source_row_index=source_row_index,
        question_raw=_text_or_empty(row.get("question")),
        question_canonical=_canonical_text(_text_or_empty(row.get("question"))),
        options=options,
        answer_key_raw=_text_or_empty(row.get("answerKey")),
        answer_option_id=answer_option_id,
        metadata={"source_id": row.get("id"), "label_sequence": option_labels},
        anomaly_flags=flags,
        exclusion_reason=exclusion_reason,
        raw_record=dict(row),
    )


def canonicalize_commonsenseqa_row(row: dict[str, Any], *, spec: DatasetSpec, source_row_index: int) -> CanonicalExample:
    choice_block = row.get("choices") or {}
    option_values = list(choice_block.get("text") or [])
    option_labels = [str(x).upper() for x in list(choice_block.get("label") or [])]
    options, flags = _options_from_list(option_values)
    answer_option_id = _normalize_answer_key(row.get("answerKey"), options, labels=option_labels or None)
    if option_labels and option_labels != [opt.option_id for opt in options]:
        flags = sorted(set(flags + ["noncanonical_option_labels"]))
    return CanonicalExample(
        example_id=f"{spec.name}:{spec.split}:{_text_or_empty(row.get('id')) or source_row_index}",
        dataset_name=spec.name,
        dataset_id=spec.dataset_id,
        dataset_revision=spec.revision,
        split=spec.split,
        source_row_index=source_row_index,
        question_raw=_text_or_empty(row.get("question")),
        question_canonical=_canonical_text(_text_or_empty(row.get("question"))),
        options=options,
        answer_key_raw=_text_or_empty(row.get("answerKey")),
        answer_option_id=answer_option_id,
        metadata={"source_id": row.get("id"), "question_concept": row.get("question_concept"), "label_sequence": option_labels},
        anomaly_flags=flags,
        exclusion_reason=(None if answer_option_id else "unresolved_answer_key"),
        raw_record=dict(row),
    )


def load_canonical_examples(spec: DatasetSpec, *, max_examples: int = 0) -> list[CanonicalExample]:
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(
        spec.dataset_id,
        spec.config_name,
        split=spec.split,
        revision=spec.revision,
    )
    rows = list(ds)
    if max_examples > 0:
        rows = rows[: int(max_examples)]
    examples = []
    for idx, row in enumerate(rows):
        if spec.dataset_id == "cais/mmlu":
            examples.append(canonicalize_mmlu_row(row, spec=spec, source_row_index=idx))
        elif spec.dataset_id == "allenai/ai2_arc":
            examples.append(canonicalize_arc_row(row, spec=spec, source_row_index=idx))
        elif spec.dataset_id == "tau/commonsense_qa":
            examples.append(canonicalize_commonsenseqa_row(row, spec=spec, source_row_index=idx))
        else:
            raise KeyError(f"unsupported dataset_id={spec.dataset_id}")
    return examples
