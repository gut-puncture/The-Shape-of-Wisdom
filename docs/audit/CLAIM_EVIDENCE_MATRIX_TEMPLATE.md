# Claim-to-Evidence Matrix Template

Use this template per run to map every requirement to objective evidence, controls, and validator status.

| requirement_id | claim_type | stage/report | artifact path | control path | validator/gate | status | notes |
|---|---|---|---|---|---|---|---|
| RQ-001 | descriptive | stage00a | `runs/<run_id>/v2/00a_generate_baseline_outputs.report.json` | same | parser surface contract | pending | |
| RQ-002 | descriptive | stage02 | `runs/<run_id>/v2/decision_metrics.parquet` | `runs/<run_id>/v2/02_compute_decision_metrics.report.json` | top1-logit authority | pending | |
| RQ-003 | descriptive | stage03 | `runs/<run_id>/v2/prompt_types.parquet` | `runs/<run_id>/v2/03_classify_trajectories.report.json` | trajectory gates | pending | |
| RQ-004 | descriptive | stage05 | `runs/<run_id>/v2/span_paraphrase_stability.parquet` | `runs/<run_id>/v2/05_span_counterfactuals.report.json` | paraphrase stability | pending | |
| RQ-005 | descriptive | stage06 | `runs/<run_id>/v2/06_select_tracing_subset.report.json` | same | domain/trajectory balance | pending | |
| RQ-006 | mechanistic | stage08 | `runs/<run_id>/v2/attention_contrib_by_span.parquet` | `runs/<run_id>/v2/08_attention_and_mlp_decomposition.report.json` | split and fit gates | pending | |
| RQ-007 | causal | stage09 | `runs/<run_id>/v2/ablation_results.parquet` | `runs/<run_id>/v2/09_causal_tests.report.json` | causal gate set | pending | |
| RQ-008 | causal | stage10 | `runs/<run_id>/v2/span_deletion_causal.parquet` | `runs/<run_id>/v2/10_causal_validation_tools.report.json` | statistical/control gates | pending | |
| RQ-009 | mechanistic | stage05/07 | `runs/<run_id>/v2/05_span_counterfactuals.report.json` | `runs/<run_id>/v2/07_run_tracing.report.json` | version floor | pending | |
| RQ-010 | descriptive | stage00a/05/07 | `runs/<run_id>/v2/00a_generate_baseline_outputs.report.json` | stage reports | checkpoint/thermal contract | pending | |
| RQ-011 | descriptive | stage00 | `runs/<run_id>/v2/00_run_experiment.report.json` | same | provenance snapshot contract | pending | |
| RQ-012 | descriptive | stage00a | `runs/<run_id>/v2/00a_generate_baseline_outputs.report.json` | same | fresh baseline regeneration | pending | |
| RQ-013 | descriptive | stage00 | `runs/<run_id>/v2/00_run_experiment.report.json` | same | fail-closed readiness propagation | pending | |
| RQ-014 | descriptive | stage14 | `runs/<run_id>/v2/meta/readiness_audit.json` | same | GO/NO-GO audit verdict | pending | |

## Status rules

- `pass`: required artifact exists and control report indicates `pass=true`.
- `fail`: artifact missing, control missing, or control `pass!=true`.
- `partial`: artifact present but control/validator is incomplete.
- `gap`: requirement has no direct artifact or control path.

## Pass-Condition Contract

- Use `pass_conditions.all_of` in `V3_REQUIREMENT_LEDGER.yaml`.
- Allowed condition types:
  - `json_equals`
  - `json_gte`
  - `json_len_eq`
  - `json_len_gte`
  - `self_contract` (for `RQ-014` audit self-check only)
- Every non-`self_contract` condition must include `source`, `path`, and `value` where applicable.
