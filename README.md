# Semantic Candidate-Set vs Winner Isolation

This repository root contains the rigor-first codebase for testing whether semantic candidate-set
formation stabilizes earlier in depth than winner isolation.

## Snapshot Status

- Root-level `v2/` was created as a mirror of the pre-existing repo.
- Snapshot verification is **not fully clean** because `logs/launchd_autopilot.out.log` changed
  during two verification passes.
- Work in this directory is proceeding only because that exception is documented in
  [archived blocker report](/Users/shaileshrana/shape-of-wisdom_legacy_20260327T121445/v2/BLOCKER_REPORT.md).
- On March 27, 2026, the latest experiment code was promoted to repository root.
- Legacy repo contents, including the historical `v2/` tree, were archived to
  `/Users/shaileshrana/shape-of-wisdom_legacy_20260327T121445/`.

## Current Phase

- Phase 1: scaffold, docs, configs, and contract tests
- Phase 2: core dataset/prompt/scoring/metrics/QC modules and runners
- Phase 3: smoke path executed end to end on March 26, 2026
- Phase 4a/4b: preregistered positivity criteria, panel split, family-readiness reporting, and
  full-sweep preflight gates implemented on March 27, 2026
- Phase 5: latest-code promotion to repository root and legacy archive completed on March 27, 2026
- Phase 6: run-scoped reliability refactor completed on March 27, 2026
- Phase 7: fail-closed optimization stages A-D completed on March 29, 2026
  - contextual continuation tokenization is now the single scorer/audit source of truth
  - grouped exact scoring shares prompt work across readouts and shared prefixes
  - `run_gpu_preflight.py` now performs prompt compatibility checks, exactness canaries, and
    batch-size calibration before the long path
  - a deterministic `paper_pilot` panel is now available for end-to-end paper-pipeline calibration
    on 5 prompts per dataset before any paid long run
- Executed in this phase window:
  - `python3 -m pytest -q` (`57 passed`)
  - `SCWI_ALLOW_INFERENCE=1 python3 run_smoke_test.py --run-id smoke_opt_20260329T150000Z --device cpu --scorer-engine optimized_exact`
- Not yet executed in this phase window:
  - `run_full_sweep.py` (blocked by paper-panel readiness gates)

Smoke outputs now present:

- `outputs/runs/smoke_opt_20260329T150000Z/qc/gpu_preflight.json`: prompt compatibility, model-load equivalence, and calibration report for the optimized scorer path
- `outputs/runs/smoke_opt_20260329T150000Z/smoke_test/smoke_manifest.json`: `24` computed units with grouped exact scoring and zero failed units
- `outputs/runs/smoke_opt_20260329T150000Z/metrics/primary_metrics.jsonl`: `24` run-scoped metric rows from the optimized scorer path
- `outputs/runs/smoke_opt_20260329T150000Z/qc/token_audits.jsonl`: `96` contextual token audit rows
- `outputs/runs/smoke_opt_20260329T150000Z/qc/qc_summary.json`: `0` failures and fully paired readout coverage after grouped scoring
- `outputs/runs/smoke_refactor_20260327T130500Z/smoke_test/smoke_manifest.json`: `24` computed units
- `outputs/runs/smoke_refactor_20260327T131200Z/smoke_test/smoke_manifest.json`: `24` cache-hit units
- `outputs/runs/smoke_refactor_20260327T130500Z/metrics/primary_metrics.jsonl`: `24` run-scoped metric rows
- `outputs/runs/smoke_refactor_20260327T130500Z/qc/token_audits.jsonl`: `96` run-scoped token audit rows
- `outputs/runs/smoke_refactor_20260327T130500Z/qc/qc_summary.json`: `0` failures and fully paired readout coverage
- `outputs/runs/smoke_refactor_20260327T130500Z/stats/model_dataset_positivity.json`: preregistered model×dataset positivity summaries
- `manifests/latest_run.json` and `manifests/latest_run_smoke.json`: latest run pointers for downstream stages

## Commands

```bash
cd /Users/shaileshrana/shape-of-wisdom
python3 -m pytest -q
SCWI_ALLOW_INFERENCE=1 python3 run_smoke_test.py --run-id smoke_YYYYMMDDTHHMMSSZ --device cpu --dtype float32 --scorer-engine optimized_exact
```

Stages can also be run independently against an explicit run:

```bash
cd /Users/shaileshrana/shape-of-wisdom
python3 run_prepare.py --smoke --run-id smoke_YYYYMMDDTHHMMSSZ
python3 run_gpu_preflight.py --smoke --run-id smoke_YYYYMMDDTHHMMSSZ --device cpu --dtype float32 --scorer-engine optimized_exact
SCWI_ALLOW_INFERENCE=1 python3 run_controls.py --run-id smoke_YYYYMMDDTHHMMSSZ --device cpu --dtype float32 --scorer-engine optimized_exact
python3 run_metrics.py --run-id smoke_YYYYMMDDTHHMMSSZ
python3 run_qc.py --run-id smoke_YYYYMMDDTHHMMSSZ
python3 run_stats.py --run-id smoke_YYYYMMDDTHHMMSSZ
python3 run_resume_failed.py --run-id smoke_YYYYMMDDTHHMMSSZ
```

Paper-panel readiness reporting is a separate explicit step:

```bash
cd /Users/shaileshrana/shape-of-wisdom
python3 run_family_readiness.py --panel paper --model-name qwen2p5_1p5b_instruct --device cpu
python3 run_family_readiness.py --panel paper --model-name gemma2_2b_it --device cpu
python3 run_family_readiness.py --panel paper --model-name llama3p2_1b_instruct --device cpu --timeout-seconds 60
```

Representative readiness artifacts already present:

- `outputs/qc/family_readiness_qwen2p5_1p5b_instruct.json`
- `outputs/qc/family_readiness_gemma2_2b_it.json`
- `outputs/qc/family_readiness_llama3p2_1b_instruct.json`

Full sweep remains opt-in and blocked by default:

```bash
cd /Users/shaileshrana/shape-of-wisdom
SCWI_ALLOW_FULL_SWEEP=1 python3 run_full_sweep.py --confirm-full-sweep --run-id paper_YYYYMMDDTHHMMSSZ --device cuda:0 --dtype float32 --scorer-engine optimized_exact
```

Budget-safe paper pilot path:

```bash
cd /Users/shaileshrana/shape-of-wisdom
python3 run_prepare.py --panel paper_pilot --run-id paper_pilot_YYYYMMDDTHHMMSSZ
python3 run_gpu_preflight.py --panel paper_pilot --run-id paper_pilot_YYYYMMDDTHHMMSSZ --device cuda:0 --dtype bfloat16 --scorer-engine optimized_exact --hourly-rate-usd <rate>
SCWI_ALLOW_INFERENCE=1 python3 run_controls.py --panel paper_pilot --run-id paper_pilot_YYYYMMDDTHHMMSSZ --device cuda:0 --dtype bfloat16 --scorer-engine optimized_exact --max-batch-size 0
python3 run_resume_failed.py --run-id paper_pilot_YYYYMMDDTHHMMSSZ
python3 run_metrics.py --run-id paper_pilot_YYYYMMDDTHHMMSSZ
python3 run_qc.py --run-id paper_pilot_YYYYMMDDTHHMMSSZ
python3 run_stats.py --run-id paper_pilot_YYYYMMDDTHHMMSSZ
```

## Output Paths

- `outputs/runs/<run_id>/prepared_data/`
- `outputs/caches/`
- `outputs/runs/<run_id>/metrics/`
- `outputs/runs/<run_id>/stats/`
- `outputs/runs/<run_id>/qc/`
- `outputs/runs/<run_id>/logs/`
- `manifests/latest_run.json`
- `manifests/latest_run_<panel>.json`

## Config Files

- `configs/models.yaml`
- `configs/datasets.yaml`
- `configs/prompts.yaml`
- `configs/readouts.yaml`
- `configs/metrics.yaml`
- `configs/controls.yaml`
- `configs/runtime.yaml`

## Panel Split

- Smoke/dev panel:
  - Models: `qwen2p5_0p5b_instruct`, `smollm2_360m_instruct`
  - Datasets: `mmlu_abstract_algebra`, `ai2_arc_challenge`
  - Readouts: semantic exact, templated semantic, letter label
  - Permutations: identity, reverse
- Paper-facing panel:
  - Families: `Qwen2.5-Instruct`, `Gemma 2 IT`, `Llama Instruct`
  - Models are revision-pinned in `configs/models.yaml`
  - Datasets: `mmlu_abstract_algebra`, `ai2_arc_challenge`, `commonsenseqa_validation`
  - Readouts: semantic exact, templated semantic, letter label
  - Permutations: identity, reverse

## Preregistered Positivity Rule

- Primary rule:
  - Scope: `model × dataset`
  - Filter: `semantic_exact` + `identity` permutation
  - Statistic: median `delta_margin`
  - Decision: grouped-bootstrap 95% CI lower bound must be `> 0`
  - Minimum valid examples: `10`
- Secondary robustness summary:
  - Scope: same subset as the primary rule
  - Statistic: proportion of prompts with `delta_margin > 0`
  - Decision: observed proportion `> 0.6` and grouped-bootstrap 95% CI lower bound `> 0.5`
  - Report-only, not the main gate

## Known Blockers

- Snapshot exception for the live-mutating root log file
- `Gemma` and `Llama` require manual Hugging Face access approval before readiness can pass
- Paper-panel Qwen sizes remain `family_ready: false` until readiness validation is run on the
  target hardware
- Full sweep is blocked until every paper-panel model resolves as enabled and `family_ready: true`
- Run-scoped analysis intentionally ignores legacy top-level `outputs/*` artifacts; only `outputs/runs/<run_id>/...`
  and the corresponding run manifest are authoritative
- Exact KV-cache reuse is still intentionally disabled pending target-GPU calibration; the current
  optimized path stops at stages A-D because those already preserve exactness and passed smoke
- Default CLI dtype is now `float32`; any reduced-precision GPU run must be requested explicitly
  and should only be used after target-hardware calibration and review
