# V3 Edge-Case Coverage Map

Normalized key format: `subsystem + trigger + expected_behavior`.

| edge_case_id | normalized_key | implementation_refs | test_refs | status |
|---|---|---|---|---|
| EC-001 | `parser + first-token letter variants + resolve canonical A/B/C/D deterministically` | `src/sow/judging/deterministic_parser.py` | `tests/v2/test_parser_option_variant_matrix.py`, `tests/test_parser_regression.py` | covered |
| EC-002 | `metrics + parser/text mismatch + correctness follows final top1 logits` | `src/sow/v2/metrics.py`, `scripts/v2/02_compute_decision_metrics.py` | `tests/v2/test_metrics_correctness_source.py` | covered |
| EC-003 | `stage00a + empty/partial manifest + fail closed` | `scripts/v2/00a_generate_baseline_outputs.py` | `tests/v2/test_stage00a_baseline_generation_contract.py` | covered |
| EC-004 | `stage00a + thermal pressure serious + checkpoint exit rc95 with resume-safe checkpoint` | `src/sow/v2/baseline_inference.py`, `scripts/v2/00a_generate_baseline_outputs.py`, `scripts/v2/run_full_local_v2.sh` | `tests/v2/test_stage00a_thermal_checkpoint_exit.py`, `tests/test_stage13_inference_checkpoint.py` | covered |
| EC-005 | `baseline layerwise + embedding-state inclusion + enforce transformer-only contiguous layer indices` | `src/sow/v2/baseline_inference.py`, `scripts/v2/00a_generate_baseline_outputs.py` | `tests/v2/test_stage00a_layer_index_contract.py` | covered |
| EC-006 | `orchestrator runtime + stale baseline count + recompute before heavy-stage backend decision` | `scripts/v2/00_run_experiment.py` | `tests/v2/test_runtime_threshold_boundaries.py` | covered |
| EC-007 | `readiness audit + unknown condition schema + fail closed with schema_errors` | `scripts/v2/14_readiness_audit.py`, `docs/audit/V3_REQUIREMENT_LEDGER.yaml` | `tests/v2/test_stage14_readiness_audit_contract.py` | covered |
| EC-008 | `readiness audit + self-reference rq014 + evaluate against in-memory audit payload` | `scripts/v2/14_readiness_audit.py`, `docs/audit/V3_REQUIREMENT_LEDGER.yaml` | `tests/v2/test_stage14_readiness_audit_contract.py` | covered |
| EC-009 | `assets stage11 + duplicate final report assembly + single-source report build` | `scripts/v2/11_generate_paper_assets.py` | `tests/v2/test_generate_assets_gate.py` | covered |
| EC-010 | `repro contract + transformer version floor + fail before baseline model load` | `src/sow/v2/model_nuances.py`, `src/sow/v2/baseline_inference.py` | `tests/v2/test_model_nuances_version_floor.py` | covered |
