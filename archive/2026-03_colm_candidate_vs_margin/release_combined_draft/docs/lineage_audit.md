# Lineage Audit

This document is release provenance. It is not paper narrative.

## Active Evidence Sources

- `outputs/runs/paper_merged_9x1000_20260330T122258Z`: merged nine-model run. It combines `paper_final_6x1000_20260330T071007Z` and `paper_remaining_3x1000_20260330T103703Z`, with `54,000` score rows, `54,000` metric rows, zero failure rows, and zero unpaired rows.
- `v3_persistent_contender_margin_fix/`: zero-new-inference persistence reanalysis from restored exact-option cache bundles. It derives `d2`, `d1`, `d25`, `d50`, `d75`, lag metrics, and late-write summaries.
- `/Users/shaileshrana/shape-of-wisdom_legacy_20260327T121445/results`: legacy three-model instrumented cache. It supplies trajectory types, attention/MLP tracing scalars, span deletion, negative controls, and substitution sensitivity summaries used in the mechanistic section.
- `submission/colm2026_openreview_anon/`: compact submission derivative of the persistence reanalysis. It excludes raw caches and absolute local paths, and provides the exemplar trajectory data used in this release.
- `paper/colm2026/`: older COLM-format paper package centered on an asymmetric first-hit candidate-set metric. It is retained only as historical context and is not used as scientific evidence for the main release claim.

## Historical Sources Reviewed And Rejected For Main Claims

- Tracked `paper/arxiv_upload/` and `paper/part1/` artifacts described older four-choice logit-readout dynamics, soft margins, PCA views, flow fields, and regime labels. Soft-margin and regime ideas are retained conceptually; PCA and flow-field claims are excluded from the streamlined paper.
- Historical v2 scripts and cached reports included causal/tracing/decomposition analyses. The release includes the audit-supported subset as mechanistic accounting, while excluding stronger full-circuit claims.
- Historical `claim_evidence_matrix.md` contains conditional mechanistic claims and blocked sensitivity links. It is used only to preserve caveats around substitution sensitivity.

## Snapshot Blocker Context

The legacy blocker report at `/Users/shaileshrana/shape-of-wisdom_legacy_20260327T121445/v2/BLOCKER_REPORT.md` records an unresolved root snapshot issue caused by a live-mutating log file. This release therefore does not ask readers to trust the live working tree. It ships a clean, hash-manifested artifact built from included derived data.

## Release Evidence Boundary

The main paper uses regenerated outputs from `data/derived/trajectory_rows.jsonl` and `data/derived/mechanistic_rows.jsonl`. The optional raw-cache verification path is documented separately, but the release rebuild does not require raw caches, model weights, GPU access, Hugging Face access, or network access after TeX cache warmup.
