# Design Decisions

## Locked Decisions

- Primary readout: exact option-text continuation scoring.
- Primary normalization: mean token logprob.
- Secondary stored normalization: raw sum logprob.
- Primary anchor: final semantic winner.
- Secondary anchor: gold answer when available.
- Primary candidate-set threshold: `tau = 0.8`.
- Primary margin threshold: `delta = 0.15`.
- Persistence rule: a stability depth is valid only if the condition holds for all later layers.
- Tie policy: ties are explicit and may invalidate a metric rather than force a hidden tie-break.
- Prompt policy: classic benchmarks remain the main panel; contamination is a documented caveat.
- Paper-family policy: the paper-facing family panel is `Qwen2.5-Instruct`, `Gemma 2 IT`, and
  `Llama Instruct`, with exact revisions pinned in `configs/models.yaml`.
- Smoke/dev policy: local-friendly smoke validation stays on `qwen2p5_0p5b_instruct` and
  `smollm2_360m_instruct` until larger paper-panel models are validated on credentialed and
  higher-memory hardware.
- Hidden-state projection mode for the currently validated smoke families:
  `apply_final_norm_except_last`. Intermediate layer states are projected through the family final
  norm and LM head; the last returned hidden state is treated as already final-normalized.
- Failure-log policy: `run_controls.py` archives any pre-existing `outputs/logs/failures.jsonl`
  before starting a new run so current-run QC and resume logic cannot inherit stale failures.
- Preregistered positivity rule: evaluate `semantic_exact` with `identity` permutation at the
  `model × dataset` level. A row is primary-positive if the grouped-bootstrap 95% CI for the
  median `delta_margin` is entirely above `0`, with `min_valid_examples = 10`.
- Secondary robustness summary: on the same `model × dataset` subset, report whether more than
  `60%` of valid prompts have `delta_margin > 0` and the grouped-bootstrap 95% CI lower bound for
  that proportion is above `50%`. This is report-only and not the primary gate.
- Gold-answer sensitivity: compute the same median-`delta_margin` summary on the gold-answer
  anchor, but treat it as sensitivity analysis rather than the primary decision gate.
- Cross-dataset claim discipline: no pooled across-dataset binary “model positive” rule is
  preregistered. Cross-dataset summaries are descriptive counts only.

## Deferred Decisions

- Whether the snapshot exception will be retired before the full run.
- When each paper-panel model will be marked `family_ready: true` after credentialed/model-load
  validation.
