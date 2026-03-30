# Blockers

## Active

1. Snapshot exception: root log file `logs/launchd_autopilot.out.log` changed during snapshot
   verification. Work is proceeding under a documented exception recorded in
   `/Users/shaileshrana/shape-of-wisdom_legacy_20260327T121445/v2/BLOCKER_REPORT.md`.
2. The paid `paper_pilot` run still needs the exact spot hourly rate in USD so
   `run_gpu_preflight.py --hourly-rate-usd ...` can emit a real cost forecast instead of a zeroed
   placeholder.
3. `run_full_sweep.py` remains intentionally blocked pending a successful `paper_pilot` run and an
   explicit post-pilot decision that the full paper run fits the budget/time envelope; the runtime
   guard must still be enabled manually even after readiness passes.
4. The staged optimization path currently stops at stages A-D. Exact KV-cache reuse is still
   intentionally disabled until target-GPU calibration proves it is needed and exact-equivalent for
   a given family.
5. The `letter_label` fast path is now measured on rented H100 hardware and behaves correctly, but
   it is only a modest runtime lever. Most of the remaining runtime pressure comes from bounded-
   drift batching limits on the Qwen family, `gemma-2-9b-it`, `gemma-2-27b-it`, and
   `Meta-Llama-3.1-8B-Instruct`.

## Resolved

1. Smoke inference completed on March 26, 2026 for `qwen2p5_0p5b_instruct` and
   `smollm2_360m_instruct` across `mmlu_abstract_algebra` and `ai2_arc_challenge`, including
   semantic, templated-semantic, and letter-label readouts plus the reverse permutation control.
2. Final-layer equivalence failures for the validated smoke families were traced to an incorrect
   projection rule. The explicit implementation decision is now: apply final norm to intermediate
   hidden states and skip final norm on the last returned hidden state for the currently validated
   families.
3. Stale failure-log contamination was fixed by archiving pre-existing `outputs/logs/failures.jsonl`
   before a new `run_controls` invocation. The archived pre-fix log is stored under
   `outputs/logs/archive/`.
4. The preregistered positivity rule is now authored in `configs/metrics.yaml` and enforced by the
   stats output path and full-sweep preflight.
5. `OLMo-2-Instruct` was removed from the default paper-facing family plan and replaced by the
   user-approved `Qwen + Gemma + Llama` paper panel.
6. The reliability refactor removed scientific dependence on global cache scans and top-level
   `outputs/*` directories. The authoritative analysis path is now `outputs/runs/<run_id>/...`
   plus `manifests/latest_run*.json`, with cache reuse preserved as a storage-only concern.
7. The fail-closed optimization refactor is now active:
   - contextual continuation tokenization is the scorer/audit source of truth
   - grouped exact scoring shares prompt work across readouts without changing scored surfaces
   - run manifests record scorer engine, equivalence checks, batch plan, and OOM backoff history
   - `run_gpu_preflight.py` now gates prompt compatibility, model-load equivalence, and
     calibration before long runs
8. Optimized smoke inference completed on March 29, 2026 for
   `qwen2p5_0p5b_instruct` and `smollm2_360m_instruct` with the grouped exact scorer and GPU
   preflight path. Representative artifacts:
   `outputs/runs/smoke_opt_20260329T150000Z/qc/gpu_preflight.json`,
   `outputs/runs/smoke_opt_20260329T150000Z/smoke_test/smoke_manifest.json`, and
   `outputs/runs/smoke_opt_20260329T150000Z/qc/qc_summary.json`.
9. A deterministic `paper_pilot` panel now exists to run the full paper pipeline end to end on 5
   prompts per dataset with a representative selection manifest and run-scoped hash. This is the
   intended paid-GPU calibration path before any full experiment.
10. The full paper panel passed remote family-readiness validation on March 30, 2026 on the rented
    H100 80GB with credentialed Hugging Face access. Qwen and Llama passed in
    `outputs/qc/family_readiness_paper.json`; Gemma passed after adding explicit
    `user_prefix` prompt handling and modeling Gemma's final-logit softcap in
    `outputs/qc/family_readiness_gemma2_2b_it.json`,
    `outputs/qc/family_readiness_gemma2_9b_it.json`, and
    `outputs/qc/family_readiness_gemma2_27b_it.json`.
11. The first `paper_pilot` controls attempt exposed a real syntax bug in `run_controls.py`
    (`IndentationError` at line 265). The indentation bug was fixed locally, the file now compiles,
    and the full local suite passed again before the spot-instance interruption.
12. The interrupted spot attempt was superseded by a successful H100 pilot run
    `paper_pilot_20260329T201258Z`, which completed end to end with `0` failures and complete
    paired coverage. Representative artifacts:
    `outputs/runs/paper_pilot_20260329T201258Z/run_manifest.json`,
    `outputs/runs/paper_pilot_20260329T201258Z/qc/qc_summary.json`, and
    `outputs/runs/paper_pilot_20260329T201258Z/logs/stage_timing.jsonl`.
13. The `letter_label` control now has an explicit fail-closed root-only fast path:
    - it is eligible only when every contextual label surface is exactly one token
    - it records explicit per-readout execution modes and timing fields
    - canary equivalence now records drift statistics and can automatically disable the fast path
      for a model if needed
    - scorer feature hashes now change when the fast path is disabled so old and new cache bundles
      cannot mix
14. The reduced final panel run `paper_final_6x1000_20260330T071007Z` completed end to end on
    March 30, 2026 with `0` total QC failures and complete paired coverage (`paired_rows=36000`,
    `unpaired_rows=0`). Stage timings are recorded in
    `/data/shape-of-wisdom/logs/full_runs/paper_final_6x1000_20260330T071007Z/stage_timing.jsonl`
    and mirrored to
    `/data/shape-of-wisdom/outputs/runs/paper_final_6x1000_20260330T071007Z/logs/stage_timing_remote.jsonl`.
15. The remaining-model run `paper_remaining_3x1000_20260330T103703Z` completed end to end on
    March 30, 2026 with `0` failures and complete paired coverage (`paired_rows=18000`,
    `unpaired_rows=0`). Both chunk runs were downloaded locally (excluding HF/model/pip/torch
    caches and excluding wholesale `outputs/caches/`), and a merged 9-model analysis set is now
    materialized at `outputs/runs/paper_merged_9x1000_20260330T122258Z`
    (`score_rows=54000`, `metric_rows=54000`, `failure_rows=0`).
16. A fresh anonymous COLM 2026 paper package now exists at `paper/colm2026/` and is generated
    directly from the merged 9-model run, not from legacy manuscript files. The current draft
    compiles locally via `tectonic` to `paper/colm2026/main.pdf`, renders page-image QA artifacts
    under `paper/colm2026/page_pngs/`, keeps the main text within the nine-page COLM limit before
    references, and includes the requested AI-use disclosure in the supplementary material.
