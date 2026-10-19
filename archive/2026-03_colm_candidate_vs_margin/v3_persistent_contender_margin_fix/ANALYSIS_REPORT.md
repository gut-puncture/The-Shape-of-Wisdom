# Analysis Report

## Inputs
- Run: `paper_merged_9x1000_20260330T122258Z`
- Manifest rows examined: `9000`
- Derived rows produced: `9000`
- Models: `9`
- Prompt IDs: `1000`

## Method
The reanalysis uses only exact-option per-layer scores and defines fully persistence-based depths:
`d2`, `d1`, `d25`, `d50`, `d75`.

The primary paper quantities are:
- `d2`: persistent contender depth
- `d1`: persistent winner depth
- `d50`: persistent 50% final-gap depth
- `Delta_21 = d1 - d2`
- `Delta_250 = d50 - d2`
- late-write share after `d2`
- late-write share after `d1`
- late-write share in the last 20% of depth

The old first-hit-vs-persistent comparison was abandoned because it compared a transient event to a persistent one.

## Main findings
- Positive models under the declared v3 rule: `3 / 9`
- Pooled median `Delta_21`: `0.086` normalized depth
- Pooled median `Delta_250`: `0.146` normalized depth
- Pooled 95% CI for `Delta_250`: `[0.133, 0.148]`
- Pooled median write share after `d2`: `0.497`
- Pooled median write share after `d1`: `0.291`
- Pooled median write share in the last 20% of depth: `0.287`

## Remaining limitations
- Exact-option readout only
- Nine tested models only
- Some lag signs are partly favored by nesting, so magnitude and late-write concentration are the real objects
- Alternate answer-format controls are heterogeneous in evaluability and sometimes in final winner identity, so they are not primary evidence
- The MMLU slice is tiny and stays out of the main narrative

## Integrity notes
The cache integrity table is written to `outputs/tables/cache_integrity_summary.csv`.
Degenerate and excluded cases are written to `outputs/tables/degenerate_case_counts.csv`.
