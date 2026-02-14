# Shape of Wisdom — Run Learnings / Ops Notes

## Qwen filtering learnings (2026-02-06)

1. **Shard size + checkpointing beats long runs**
   - Stable pattern: shard-based filtering with `offset` + `limit` and persisted `state.json`.
   - Large monolithic runs were kill-prone; resumable shards made progress durable.

2. **Good default shard size on this Mac mini**
   - `limit=120` worked well after hardening.
   - Last shard should auto-shrink to remaining rows.

3. **Memory behavior on 32GB unified memory**
   - Qwen filter process RSS peaks around ~21.5 GB (observed).
   - Practical headroom remains, but avoid concurrent heavy model jobs.

4. **Reliability patterns that mattered**
   - Stale lock detection + PID-based lock recovery.
   - Per-shard logs (`qwen_offset_<offset>.log`).
   - Always write structured status JSON after each step.

5. **Merge step must be explicit**
   - `merge_qwen_filter_shards.py` must receive explicit `--shard-dir` + `--out` in orchestrators.

---

## OpenAI batch sanity learnings (gpt-5-nano)

1. **When batch is `completed` but no `output_file_id`**
   - Always check `error_file_id` and download it.
   - Inspect `request_counts` (`completed` / `failed`) before assuming anomaly.

2. **Critical token-budget rule (persist this)**
   - For GPT-5 sanity batches, do **NOT** use tiny completion caps.
   - Use `max_completion_tokens ~ 2000` by default for stability.
   - Previous `max_completion_tokens=1` caused 10/10 failures with output-limit error.

3. **Model-specific param caveat (gpt-5-nano)**
   - Do not force `temperature=0` for this batch path; it returned 400 unsupported-value errors.
   - Use model defaults unless docs explicitly confirm override support.

4. **Batch discipline**
   - Keep one prompt per request for deterministic tracing.
   - Persist both `batch_output.jsonl` and `batch_error.jsonl`.

---

## macOS crash-hardening learnings (OpenMP / libomp)

1. **Do not use `KMP_DUPLICATE_LIB_OK=TRUE` as a fix**
   - It can suppress the duplicate-runtime guard but still leave unstable state (including SIGSEGV/SIGABRT).
   - New policy: explicitly run with `KMP_DUPLICATE_LIB_OK=FALSE`.

2. **Sanitize dynamic-loader env before torch/sklearn runs**
   - Inherited `DYLD_LIBRARY_PATH` / `DYLD_INSERT_LIBRARIES` can force an extra `libomp` into process startup.
   - New policy: unset both before heavy Python jobs.

2b. **Critical root cause found: mixed package provenance (numpy vs torch)**
   - `torch` loaded `.../site-packages/torch/lib/libomp.dylib` while Homebrew `numpy` pulled `libopenblas -> /opt/homebrew/opt/libomp/lib/libomp.dylib`.
   - That created true dual-OpenMP-runtime load in one process.
   - Fix applied: install/use user-site `numpy` wheel (Accelerate-backed, no Homebrew libomp dependency), keeping core stack on a consistent provenance.

3. **Standard launcher for Python jobs on this Mac mini**
   - Use `scripts/macos_safe_python_env.sh` for heavy/model-adjacent runs.
   - It enforces:
     - clean DYLD env,
     - no duplicate-runtime bypass,
     - conservative thread caps (`OMP/MKL/NUMEXPR/VECLIB=1`),
     - tokenizer parallelism off,
     - MPS fallback on.

4. **Operational pattern change**
   - All long-running inference/filter pipeline entrypoints now invoke Python through the safe launcher.
   - Avoid ad-hoc shell env exports for OpenMP behavior.

5. **Command pattern (copy this shape)**
   - `scripts/macos_safe_python_env.sh python3 scripts/run_inference.py ...`
   - `scripts/macos_safe_python_env.sh python3 scripts/filter_one_token_viability.py ...`

## Execution policy

- Heavy model runs: one at a time (Qwen/Llama/Mistral sequential) on 32GB unified memory.
- Add a thermal cooldown: wait **20 minutes** between heavy inference steps (after one heavy run ends, before starting the next heavy run).
- For very long heavy stages (hours), also use **intra-stage cooldowns**: run ~**90 minutes**, then cool down **15–20 minutes**, then resume.
- While heavy inference is running, use idle CPU for code/validation/report tasks in parallel.
- Every step must end with explicit validation artifact before the next step begins.
- For model-adjacent Python runs on macOS, use `scripts/macos_safe_python_env.sh`.
- Autonomy rule: at each task completion, inspect outputs + gates, then either (a) trigger the next planned step, or (b) branch to remediation if outputs are unexpected; do not wait for manual prompting unless blocked.
- Prefer end-of-task OpenClaw callbacks (`scripts/run_with_openclaw_callback.sh` and `scripts/wait_for_file_then_callback.sh`) over frequent polling.
- For chunked long runs with intra-stage cooldowns, use:
  - `--time-budget-seconds` + `--done-sentinel` in `scripts/run_inference.py`
  - the loop wrapper `scripts/run_with_cooldowns_loop.sh`
- Keep polling cron checks low-frequency (10-minute cadence) as fallback only.
- API key location for this project: `.secrets/llm_keys.env` (permission 600); never log key values.

---

## DeepSeek adjudication learnings (unresolved-only)

- **Rule:** Always run the deterministic parser first. Only send rows to DeepSeek when the parser is **unresolved** (no choice) or explicitly **conflicting**.
- **Why:** This minimizes judge variance + cost, and keeps most labels fully auditable/deterministic.
- **Tool:** `scripts/deepseek_adjudicate_unresolved.py`

Example (adjudicate only unresolved rows in a judged JSONL; safe to re-run with resume):

- `scripts/macos_safe_python_env.sh python3 scripts/deepseek_adjudicate_unresolved.py \
  --input artifacts/<...>/deterministic_judged.jsonl \
  --output artifacts/<...>/deepseek_adjudicated.jsonl \
  --report artifacts/<...>/deepseek_adjudicated_report.json \
  --resume`

Dry-run (no API calls; validates filtering + resume logic):

- `python3 scripts/deepseek_adjudicate_unresolved.py --input ... --output ... --report ... --dry-run`

Operational notes:
- Expects `DEEPSEEK_API_KEY` in env.
- Enforces strict JSON output schema and retries once if the model output is invalid.
