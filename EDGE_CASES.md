# Edge Cases

This file records edge-case policy as: why it matters, what it can artifactually create, what is
logged, and when it escalates from warning to hard failure.

## Repo / Snapshot

- External symlinks in `.venv*` matter because a dereferenced copy would cross repo boundaries.
  Logged as snapshot metadata. Hard failure if the mirror becomes partial or unresolved.
- Live-mutating root files matter because they can make a mirror look complete while being stale.
  Logged in `../BLOCKER_REPORT.md`. Hard failure unless the exception is explicitly accepted.

## Environment / Dependency

- Version drift matters because tokenizer and hidden-state behavior can change across
  `transformers`, `datasets`, and `tokenizers` releases.
  Logged in `ENVIRONMENT.md` and runtime environment manifests.
  Hard failure for unsupported or unpinned major drift in a full run.

## Dataset Ingestion

- `MMLU` answer values can appear as integer indices even when the feature declares labels.
  Logged as raw answer provenance. Hard failure only if the answer cannot be resolved.
- `ARC-Challenge` is not uniformly 4-choice. This is a warning, not an error, unless downstream
  code assumes 4 options.
- `CommonsenseQA` uses 5 options. This is expected and logged as variable option count.
- Missing question text, missing options, or unresolved answer mapping are hard failures for that
  example and must be recorded with counts.

## Canonicalization

- Raw and canonical text must both be preserved so no silent normalization is hidden.
- Duplicate or near-duplicate option texts are logged as warning flags and may be escalated if
  tokenization collisions make the readout degenerate.

## Prompt Construction

- Chat-template drift matters because the answer boundary can move silently.
  Logged via template ID plus runtime-rendered prompt string.
- Permutation remapping matters because an incorrect answer remap would invalidate every metric.
  This is hard-failed in tests and at runtime.
- Prompt overflow matters because one readout or permutation may truncate while another does not.
  Logged per unit and escalated to paired-comparison attrition in QC.

## Layerwise Extraction

- Hidden-state indexing matters because many HF models expose embeddings plus layer outputs.
  Logged by explicit layer indexing and final-layer equivalence checks.
- Final norm handling matters because projecting raw hidden states and post-norm states are not
  the same scientific object. The convention is explicit per adapter.
- NaN/Inf scores are hard failures.

## Option-Text Scoring

- Exact option text is the primary surface. No synonym expansion or lowercasing is allowed.
- Multi-token teacher forcing is required because later token probabilities depend on earlier
  forced tokens. Each token is logged in the layerwise audit path.
- Tokenization collisions are warnings by default and hard failures if they prevent distinct
  options from being distinguished under the configured readout.

## Controls

- Templated semantic and letter-label readouts are separate controls and must be kept paired with
  the semantic readout on the same valid example subset.
- Coverage differences are logged in QC and reported separately from paired results.

## Metrics / Stats

- Tie handling is explicit. Ambiguous depths may become `null` rather than forced.
- Family-level claims remain descriptive unless model-level positivity criteria are authored.
- Threshold fishing is disallowed. Sensitivity grids are labeled sensitivity analyses.

