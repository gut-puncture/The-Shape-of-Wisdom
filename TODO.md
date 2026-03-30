# TODO

- [x] Phase 0: create `v2/` mirror and document snapshot blocker
- [x] Phase 1: create workspace, docs, configs, and contract tests
- [x] Phase 2: implement dataset/prompt/scoring/metrics/QC core modules and runner skeletons
- [x] Phase 2 verification: run local contract tests and non-inference prepare/QC/stats path
- [x] Phase 3: execute the smoke path with documented outputs
- [x] Phase 4a: author preregistered positivity criteria and split smoke/dev vs paper panels
- [x] Phase 4b: implement family-readiness reporting and full-sweep preflight gating
- [x] Phase 5: promote the latest experiment code to repository root and archive legacy contents
- [x] Phase 5b: refactor runners to use explicit run manifests, config-driven controls, honest
  device propagation, real resume behavior, and run-scoped analysis outputs
- [x] Phase 6: implement fail-closed optimization stages A-D
  contextual continuation tokenization, grouped exact scorer, readout sharing, dynamic batch
  backoff, GPU preflight, and target-hardware equivalence gates
- [x] Phase 6 verification: pass full local test suite and execute optimized smoke run with
  `run_gpu_preflight.py` + `run_controls.py` using the grouped scorer path
- [x] Phase 6f: add a deterministic `paper_pilot` panel and representative 5-per-dataset
  prepare path so we can run the full paper pipeline end to end on a budget-safe subset before
  renting a long GPU job
- [x] Phase 4c: validate the paper panel on credentialed/higher-memory hardware and mark approved
  models `family_ready: true`
- [x] Phase 4c pilot: run the end-to-end `paper_pilot` path on rented credentialed GPU hardware and
  use its forecast plus observed timing to decide whether the final paper run fits the budget/time
  envelope
  - March 30, 2026 H100 spot attempt reached `host_snapshot` (`0.065s`), `prepare` (`7.09s`),
    and `gpu_preflight` (`170.179s`) successfully under run
    `paper_pilot_20260329T194536Z`
  - the first `controls` attempt failed immediately with an `IndentationError` in `run_controls.py`;
    the bug was fixed and local plus remote compilation/tests were revalidated before resuming
  - the spot host was then reclaimed or replaced during `controls_retry1`, so the pilot still needs
    completion on a reachable GPU before we can trust a full-run forecast
  - March 30, 2026 H100 follow-up run `paper_pilot_20260329T201258Z` completed end to end with
    `0` failures and complete paired coverage
  - observed staged timings were `host_snapshot=0.069s`, `prepare=6.611s`,
    `gpu_preflight=1231.235s`, `controls=739.802s`, `resume_failed=0.197s`, `metrics=0.874s`,
    `qc=0.253s`, and `stats=0.227s`
  - the current grounded warm-cache H100 full-run estimate from that pilot is roughly `~21h`, so
    further optimization and/or scope reduction is still needed for the tighter user budget target
- [x] Phase 7a: add an explicit fail-closed `letter_label` root-only fast path with per-readout
  execution-mode metadata, drift reporting, and feature-hash invalidation
- [x] Phase 7a verification: pass the full local suite after adding the `letter_label` root-only
  fast path and the new execution-mode / drift-report tests
- [ ] Phase 4d: rerun a clean root snapshot after the live root log is quiesced
- [ ] Phase 6e: add KV-cache reuse only if target-GPU calibration still shows the current exact
  grouped scorer is too slow after stages A-D
- [ ] Phase 7b: run bounded-drift batching experiments on rented H100 hardware to see whether the
  current batch-size-1 families can safely move upward without winner flips
  - March 30, 2026 one-hour H100 tuning session found and fixed a real deployment issue: the rented
    host had still been importing the older `src/scoring_engine.py`, so the first “relaxed”
    calibration results were invalid
  - after syncing the patched scorer correctly, a fresh preflight run
    `paper_pilot_tuning_relaxed_fixed_20260330T072000Z` moved `google/gemma-2-2b-it` from safe
    batch `1` to safe batch `32`; the remaining Qwen / larger Gemma / Llama 8B families still hit
    bounded-drift limits on specific calibration prompts
  - the corrected end-to-end `paper_pilot` controls stage for that run completed in `602.50s`
    versus the earlier `739.802s`, an observed speedup of about `18.6%`
  - the next decision is product-level, not implementation-level: either keep optimizing the
    remaining bounded-drift bottlenecks, or reduce the final paid panel to `6` models and roughly
    `1000` prompts
- [x] Phase 7c: prepare a reduced final-run panel that is launch-ready under the tighter budget
  target
  - added runtime panel `paper_final_6x1000`
  - fixed final panel shape to `6` models: `qwen(1.5b,7b)`, `gemma(2b,27b)`, and
    `llama(1b,8b)`
  - fixed exact dataset allocation to `1000` prompts total:
    `11 mmlu + 195 arc + 794 csqa`
  - local dry run `paper_final_6x1000_local_check` completed successfully
  - full local test suite passed after the panel and selection-limit changes
- [x] Phase 8: execute the paid final run end to end on remote H100 and archive all artifacts on
  attached disk
  - March 30, 2026 launch started on host `65.108.33.101` under run
    `paper_final_6x1000_20260330T071007Z`
  - completed end to end successfully at `2026-03-30T09:55:10Z` with staged timings:
    `family_readiness=2202s`, `prepare=24s`, `gpu_preflight=288s`, `controls=7130s`,
    `resume_failed=1s`, `metrics=236s`, `qc=6s`, `stats=8s`
  - total wall time was `9895s` (`2h44m55s`)
  - QC reported `total_failures=0` with full paired coverage (`paired_rows=36000`,
    `unpaired_rows=0`)
  - per-stage timing and stage logs are written under
    `/data/shape-of-wisdom/logs/full_runs/paper_final_6x1000_20260330T071007Z/`
- [x] Phase 8b: execute the remaining 3 paper models on the same 1000-example split and prepare a
  merge-ready combined analysis set
  - added runtime panel `paper_remaining_3x1000` for:
    `qwen2p5_3b_instruct`, `gemma2_9b_it`, `llama3p2_3b_instruct`
  - panel keeps the same `11/195/794` dataset allocation, readouts, and permutations as
    `paper_final_6x1000` so outputs stay schema-compatible for merge
  - remote run `paper_remaining_3x1000_20260330T103703Z` completed end to end with staged outputs:
    `prepare`, `run_gpu_preflight`, `run_controls`, `run_metrics`, `run_qc`, `run_stats`
  - QC and coverage checks are clean for this run: `failure_count=0`, `total_failures=0`,
    `groups_missing_any_readout=0`, `unpaired_rows=0`
  - paper-required artifacts for both chunk runs were downloaded locally (excluding HF/model/pip/
    torch caches and excluding wholesale `outputs/caches/`) under:
    `outputs/runs/paper_final_6x1000_20260330T071007Z`,
    `outputs/runs/paper_remaining_3x1000_20260330T103703Z`, and
    `logs/full_runs/{paper_final_6x1000_20260330T071007Z,paper_remaining_3x1000_20260330T103703Z}`
  - merged 9-model analysis set materialized locally as
    `outputs/runs/paper_merged_9x1000_20260330T122258Z` with:
    `prepared_examples=1000`, `score_rows=54000`, `metric_rows=54000`,
    `token_audits=258876`, `failure_rows=0`, `positivity_rows=81`
  - merged QC is clean: `total_failures=0`, `unpaired_rows=0`
- [x] Phase 9: build the anonymous COLM 2026 paper package from the merged 9-model run with
  generated assets, local PDF compilation, and page-image QA
  - fresh manuscript workspace created at `paper/colm2026/` using the official COLM 2026 style
    files only as formatting assets, not as a content template
  - all paper figures, tables, and numeric macros are now generated from
    `outputs/runs/paper_merged_9x1000_20260330T122258Z` via
    `scripts/paper/build_colm2026_assets.py`
  - anonymous submission draft compiled successfully to `paper/colm2026/main.pdf` via
    `scripts/paper/build_colm2026_pdf.sh`
  - rendered page QA artifacts are stored under `paper/colm2026/page_pngs/`
  - current draft keeps the main text within the COLM 2026 nine-page limit before references and
    includes the requested AI-use disclosure in the supplementary material
