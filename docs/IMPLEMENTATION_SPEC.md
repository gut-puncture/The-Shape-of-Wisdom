# Shape of Wisdom — Implementation Plan (Fresh Rewrite, Full Engineering Contract)

This plan is written to be given to an autonomous coding agent.  
The agent must treat this as a strict contract, not a suggestion.  
If something is unclear, the agent must stop and ask a targeted question instead of guessing.  
If something is expensive, the agent must build validators and smoke tests first.

## 1) Outcome and non-negotiable scientific claim

The entire pipeline exists to support one claim:

For a single prompt, a transformer’s internal decision state evolves across layers in a structured and repeatable way that can be read out and quantified.  
This evolution has two repeatable signatures:
1) The decision state becomes more decisive across depth (commitment).  
2) Across many prompts, decision states organize into stable regions that correlate with domains and with systematic wrong commitments (wrong basins).

A robustness control exists to ensure the signatures are not prompt-format artifacts.  
Everything else is scaffolding to make this claim auditable, reproducible, and difficult to accidentally break.

## 2) Hard constraints that shape the whole architecture

No human code review should be assumed.  
Correctness must be enforced by deterministic validators and regression tests.

Every expensive stage must be stage-gated:
- Build artifact.
- Validate artifact.
- Only then proceed to the next expensive stage.

Every run must be reproducible from:
- Prompt manifests (with secure hash algorithm 256 (SHA-256) checksums).
- Model identifiers plus revision or commit hash when possible.
- Recorded command lines.
- Recorded random seeds.
- Stored configuration snapshot.

All outputs must be append-only and resumable:
- Never “rewrite in place” for results.
- Always resume safely without duplicating rows or silently skipping rows.

Prompt manifests are expensive inputs:
- Never overwrite expensive manifests in place.
- Any “fix” must output a new file with a new name and a recorded canonicalization report.

## 3) Repository layout (minimum viable, rewrite-friendly)

Required to exist and to stay stable across rewrites:

NOTE (2026-02-14): Prior methodology references (PDF / extracted text) are deprecated and should not be treated as canonical.
This repo uses `docs/IMPLEMENTATION_SPEC.md` (this contract) and `docs/METHODOLOGY_NARRATIVE_V2.md` (paper narrative) as the
canonical methodology references going forward.

- `data/experiment_inputs/`
  - `main_prompts.jsonl`  
  - `robustness_prompts_v2.jsonl`  
  - `build_summary.json`  

- `configs/`
  - `coarse_domain_mapping.json` (native -> coarse domain mapping for dataset build and reporting)

- `docs/`
  - `IMPLEMENTATION_SPEC.md` (this document)
  - `METHODOLOGY_NARRATIVE_V2.md` (optional but recommended scientific narrative)

- `STATE.md` (runtime + execution log; updated after every stage)
- `PROJECT_PLAN.md` (stable checklist of stages; keep it free of run-specific data)

- `artifacts/parser_edge_case_regression/`
  - `regression_cases.json` (canonical deterministic parser regression suite)

Everything else can be regenerated.

Recommended new code layout:

- `src/sow/`
  - `cli.py` (command line entry point)
  - `config.py` (load and validate run configuration)
  - `hashing.py` (SHA-256 helpers)
  - `io_jsonl.py` (append-only writers, resumable readers)
  - `manifest/`
    - `schema.py` (manifest row schema + invariants)
    - `build_main.py`
    - `build_robustness.py`
    - `validate.py`
    - `canonicalize.py`
  - `tokenization/`
    - `option_buckets.py` (A/B/C/D token buckets per model)
    - `normalize.py`
  - `judging/`
    - `deterministic_parser.py`
    - `parser_tests.py` (loads regression file and asserts exact match)
    - `llm_fallback.py` (only for unresolved rows; strict schema; full audit logs)
  - `inference/`
    - `load_model.py`
    - `runner.py`
    - `logit_lens.py`
    - `batching.py`
    - `resume_keys.py`
    - `gates.py` (batch-consistency, resume simulation, etc.)
  - `pca/`
    - `sample.py`
    - `fit.py`
    - `project.py`
    - `repro_tests.py`
  - `analysis/`
    - `metrics.py`
    - `aggregate.py`
    - `plots.py`
    - `reports.py`

- `runs/`
  - Each run has its own directory: `runs/<run_id>/`
  - Never mix outputs from different run identifiers.

## 4) Configuration: one file controls everything

Create a single configuration file per run:
- `runs/<run_id>/run_config.yaml`

It must contain:
- `run_id` (string, immutable)
- `random_seed` (integer, immutable)
- `models` (list)
  - Each model has:
    - `model_id` (for example, a Hugging Face identifier)
    - `revision` (commit hash or version tag if available)
    - `dtype` (for example, float16 or bfloat16)
    - `device` (for example, cuda, metal, or cpu)
- `dataset`
  - dataset identifier and subset settings
- `prompting`
  - baseline template identifier
  - robustness wrapper set identifier and expected wrapper identifiers
- `generation`
  - decoding settings (must be fixed; no tuning after looking at results)
- `pca`
  - sample size
  - component count (default 128)
  - sampling policy
- `filters`
  - prompt length constraints
  - one-token compliance thresholds
  - Primary Core Corpus size target
  - Common Compatible Core thresholds

The code must snapshot the exact configuration into:
- `runs/<run_id>/meta/config_snapshot.yaml`
and include its SHA-256 checksum in every report.

## 5) Data model contracts (schemas you must not break)

### 5.1 Example schema (dataset-level canonical format)

One row per question:

- `example_id` (stable identifier)
- `domain_native` (original subject label)
- `domain_coarse` (mapped coarse domain for readability)
- `question_text`
- `options` object:
  - `A`, `B`, `C`, `D` (strings)
- `correct_key` (one of A/B/C/D)

Store as:
- `data/derived/examples.jsonl` (append-only, but generated deterministically)

### 5.2 Prompt manifest schema (baseline and robustness)

One row per (example, wrapper):

Required fields per row:
- `example_id`
- `wrapper_id` (baseline uses a fixed id like `plain_exam`)
- `prompt_text`
- `options` (A/B/C/D strings again, duplicated for self-contained joins)
- `correct_key`
- `prompt_uid` (collision-free unique id)
- `manifest_sha256` (hash of canonicalized row representation, optional but recommended)

Uniqueness invariants:
- Baseline: exactly 1 row per `example_id`
- Robustness: exactly 1 row per (`example_id`, `wrapper_id`)
- Wrapper set must match expected list exactly (same identifiers and count)

The manifest file itself gets:
- `runs/<run_id>/manifests/<name>.jsonl`
- `runs/<run_id>/manifests/<name>.meta.json`
  - includes file SHA-256
  - build settings
  - wrapper list
  - dataset settings
  - build timestamp
  - code version hash if available

### 5.3 Inference output schema (append-only JSON lines)

One row per (model, prompt_uid).  
Primary join key is (`model_id`, `prompt_uid`).

Required fields:
- Identification
  - `run_id`
  - `model_id`
  - `model_revision`
  - `prompt_uid`
  - `example_id`
  - `wrapper_id`
- Prompt details (or a hash pointer)
  - `prompt_text_sha256`
  - `prompt_length_tokens`
- Generation outputs
  - `first_generated_token_id`
  - `first_generated_token_text`
  - `generated_text` (store full text or a capped length)
  - `generation_settings` (embedded or referenced)
- Deterministic grading
  - `parsed_choice` (A/B/C/D or null)
  - `is_correct` (true/false/null)
  - `parser_status` (resolved / unresolved / conflict)
  - `parser_signals` (structured explanation)
- Layerwise candidate competition readouts
  - For each layer index `l`:
    - `candidate_logits` for A/B/C/D (after bucket aggregation)
    - `candidate_probs` for A/B/C/D (softmax restricted to candidates only)
    - `candidate_entropy` (entropy over those four probabilities)
    - `top_candidate` (A/B/C/D)
    - `top2_margin_prob` (top prob minus second prob)
- Trajectory summary metrics
  - `flip_count`
  - `commitment_layer_by_margin_threshold` (dictionary keyed by threshold)
- Representation vectors
  - `projected_hidden_128` per layer (only after PCA basis is frozen)

All arrays must be consistent length across layers.  
All numeric outputs must have explicit float type in serialization (avoid mixed integer/float confusion).

## 6) Stage plan overview (the only allowed execution order)

Stage order is not flexible because principal component analysis depends on hidden vectors and must be frozen before full runs.

Stage 0: Environment lock and smoke test.  
Stage 1: Build canonical examples dataset.  
Stage 2: Build baseline prompt manifest.  
Stage 3: Build robustness prompt manifest.  
Stage 4: Validate manifests and canonicalize if needed.  
Stage 5: Build per-model option token buckets (A/B/C/D).  
Stage 6: Deterministic answer parser implementation + regression suite pass.  
Stage 7: Pilot inference (small) to measure one-token compliance and prompt viability.  
Stage 8: Build Primary Core Corpus (filtered headline set).  
Stage 9: Build Common Compatible Core across all models (intersection).  
Stage 10: Choose and freeze PCA sample membership (deterministic).  
Stage 11: Run PCA sample extraction inference (only those prompts).  
Stage 12: Fit PCA basis once per model, pooled across layers.  
Stage 13: Full inference runs with on-the-fly PCA projection and full logging.  
Stage 14: Analysis and report generation.

Every stage writes:
- outputs into `runs/<run_id>/`
- a validation report JSON
- an updated `STATE.md` entry

No stage is allowed to proceed if its validator fails.

## 7) Stage 0 — Environment lock and smoke test

Goal: prevent “it worked on my machine” failures.

Actions:
- Pin Python version and dependencies.
- Pin PyTorch version.
- Pin Transformers library version.
- If using GPUs, pin CUDA toolkit and driver expectations in a plain text note.

Required artifacts:
- `runs/<run_id>/meta/environment.json`
  - python version
  - package versions
  - operating system
  - device name
- `runs/<run_id>/meta/smoke_test.json`
  - confirms:
    - model loads successfully
    - tokenization works
    - one prompt inference runs end-to-end
    - hidden states can be extracted
    - candidate scoring code runs

Validator:
- Smoke test must complete in under a small fixed time and exit with success.
- If it fails, stop immediately.

## 8) Stage 1 — Build canonical examples dataset

Input:
- Multiple-choice dataset source with labels and known correct key.

Output:
- `data/derived/examples.jsonl`
- `data/derived/examples.meta.json`

Implementation notes:
- Normalize Unicode in question and options using compatibility decomposition followed by composition (Unicode NFKC).
- Preserve original text as well if you want auditability, but canonicalized text is what prompts should use.
- Map native subject labels to coarse domains using the deterministic mapping file at `configs/coarse_domain_mapping.json`.

Validator:
- Every row has exactly four options.
- `correct_key` is one of A/B/C/D.
- No empty option strings.
- `example_id` uniqueness holds.

## 9) Stage 2 — Build baseline prompt manifest

Goal: create the canonical prompt format where the answer begins immediately after the prompt.

Prompt rules:
- Options are visible.
- Prompt ends with literal `Answer:` (exact casing).
- The instruction demands output of only one letter in {A,B,C,D}.
- Do not force constrained decoding for the canonical runs. Keep decoding unconstrained unless explicitly scoped as a control.

Important implementation detail:
The prompt should include a trailing space after `Answer:` so the model naturally emits “ A” in many tokenizers.  
This is why option token buckets must include both “A” and “ A” forms.

Output:
- `runs/<run_id>/manifests/baseline_manifest.jsonl`
- `runs/<run_id>/manifests/baseline_manifest.meta.json`

Validator (must be strict):
- Exactly one row per `example_id`.
- `prompt_uid` is unique across the file.
- Wrapper id is always `plain_exam`.
- Prompt ends with `Answer:` plus trailing space.
- File SHA-256 recorded.

## 10) Stage 3 — Build robustness prompt manifest (20 wrappers)

Goal: create 20 wrapper “skins” that preserve question and options but change surface structure.

Hard rules:
- Wrapper identifiers must be fixed and stable and must match the expected wrapper list exactly.
- Each wrapper prompt used for inference must end with literal `Answer:` plus a trailing space and must demand one-letter output.

Default expected robustness wrapper_id list (v2):
- `academic_abstract`
- `ascii_box`
- `changelog_entry`
- `csv_inline`
- `graphql_query`
- `haiku_riddle`
- `html_form`
- `ini_file`
- `irc_log`
- `key_equals`
- `legal_clause`
- `meeting_minutes`
- `protobuf_msg`
- `quest_briefing`
- `recipe_instruction`
- `regex_match`
- `s_expression`
- `shell_heredoc`
- `toml_config`
- `tweet_thread`

Wrapper construction policy:
You can implement wrappers as deterministic templates.  
This is the default recommendation because it is easier to validate.  
If you generate wrappers with another language model, you must store:
- generator model identifier
- generator revision
- full raw generation output
- a semantic-preservation validation report
and you must treat the generated manifest as expensive and immutable.

Output:
- `runs/<run_id>/manifests/robustness_manifest_v2.jsonl`
- `runs/<run_id>/manifests/robustness_manifest_v2.meta.json`

Validator:
- For every `example_id`, the manifest contains exactly 20 wrapper rows.
- The wrapper set matches the expected wrapper identifier list exactly.
- `prompt_uid` uniqueness holds.
- `prompt_text` always ends with `Answer:` plus trailing space.
- No wrapper produces empty content or drops options.

## 11) Stage 4 — Manifest canonicalization and integrity guarantees

Problem:
Manifests are the backbone. Any silent duplication or missing pair corrupts all paired comparisons.

Canonicalization policy (deterministic):
- If duplicates exist for a key:
  - For this project, the paid robustness v2 input contains a known mixture of older+newer rows for some `(example_id, wrapper_id)`.
  - We **keep the later file occurrence** (larger source line index) and drop the earlier occurrence.
  - Log all dropped duplicates in a canonicalization report (line numbers + hashes + prompt ids).
- If missing wrappers exist:
  - Fail fast by default.
  - A narrow deterministic repair is allowed only for the known missing pair `(mmlu::test::12183, ascii_box)` using a local
    reproducible template (logged in the canonicalization report as a repair event).
  - Any other missing wrapper pairs are treated as a hard failure and require an explicit regeneration mode that:
    - records generator model id + revision
    - stores the full raw generator output
    - writes a regeneration report

Outputs:
- `runs/<run_id>/manifests/<name>.canonical.jsonl` (only if needed)
- `runs/<run_id>/manifests/<name>.canonicalization_report.json`

Non-negotiable:
Never overwrite the original manifest file.  
Always write a new canonical file.

## 12) Stage 5 — Option token buckets (A/B/C/D) per model

Why:
Tokenizers often represent “A” differently based on leading space and punctuation.  
Your candidate scoring must be robust to this.

Goal:
For each model, build a bucket of token identifiers that correspond to each letter option.

Procedure:
- Enumerate a small set of textual variants per letter:
  - `A`, ` A`, `\nA`, `(A)`, `A.`, `A:` (and similarly for B/C/D)
- Tokenize each variant.
- Collect token identifiers that decode to a piece that normalizes to the target letter.
- Normalize decoding using:
  - Unicode NFKC
  - strip whitespace
  - uppercase
  - remove surrounding punctuation for bucket membership checks

Aggregation rule:
When computing evidence for a letter, aggregate token logits within a bucket using log-sum-exp.  
This avoids unfairly penalizing letters with multiple valid tokens.

Fail-fast:
If any bucket is empty for any model, stop.  
This methodology cannot proceed for that model without changing the approach.

Outputs:
- `runs/<run_id>/token_buckets/<model_id_fs>.json` where `<model_id_fs>` is a filesystem-safe form of the HF model id
  (for example `meta-llama__Llama-3.1-8B-Instruct`).

Validator:
- Each of A/B/C/D has at least one token identifier.
- Buckets are disjoint or, if overlaps exist, overlaps are logged and handled deterministically.

## 13) Stage 6 — Deterministic answer parser + regression suite

Why:
The model will not always output exactly one letter.  
You must grade deterministically first, then only send unresolved cases to a language-model judge.

Rules the parser must implement:
- Unicode NFKC normalization.
- Canonicalize Unicode minus variants to `-`.
- Lowercase for matching.
- Normalize whitespace.

Signal extraction priority:
1) Letter candidates A/B/C/D from:
   - first generated token if it begins with a letter after normalization
   - cue patterns in response text: `answer: X`, `final answer is X`, `option X`, `choice X`, `pick X`, `select X`
   - start-of-response fallbacks: `(A)`, `A.`, `A:`
2) Numeric candidates:
   - first token numeric canonicalization
   - cue patterns similar to above
   - leading numeric in response
   - special-case recognition of pi as “pi” or “π”
3) Numeric-to-option mapping:
   - strict exact match to a unique numeric option value
   - strict match to a unique leading numeric at the start of an option string
   - only then an ordinal fallback 1..4 maps to A..D (low-confidence but deterministic)
4) Option-text substring hits as last resort:
   - normalized option text substring in normalized response
   - ignore ultra-short options under 3 characters

Resolution rules:
- Only resolve if a single unique choice is supported and there is no conflict.
- If multiple distinct letters appear, mark unresolved.
- If numeric and letter conflict, mark unresolved.
- If numeric matches multiple options, mark unresolved.
- If substring hits match multiple options, mark unresolved.

Regression:
- Load `artifacts/parser_edge_case_regression/regression_cases.json`
- The parser must match expected decisions exactly.

Outputs:
- `runs/<run_id>/validation/parser_regression_report.json`

Fail-fast:
If regression fails, stop the entire pipeline.

## 14) Stage 7 — Pilot inference for one-token compliance and viability

Goal:
Confirm that the canonical prompting actually yields a first token that is usually one of A/B/C/D.  
Confirm that the first token is one-token under tokenization and that the methodology is viable.

Pilot size:
- About 200 to 500 baseline prompts.  
- Stratified across coarse domains.

Run policy:
- Use the exact generation settings planned for full runs.
- Record first generated token and text.
- Do not run any expensive layerwise extraction in the pilot beyond what is needed.

Outputs:
- `runs/<run_id>/pilot/<model_id>_pilot_outputs.jsonl`
- `runs/<run_id>/pilot/<model_id>_pilot_report.json`

Metrics in pilot report:
- One-token compliance rate: fraction where first token normalizes to A/B/C/D.
- Fraction where deterministic parser resolves to a letter.
- Accuracy based on deterministic parser resolution (plus unresolved count).
- Domain breakdown of compliance.

Gates:
- If compliance is low, stop and adjust prompting before doing anything else.
- Do not “push through” and hope it gets better at scale.

## 15) Stage 8 — Build Primary Core Corpus (filtered headline prompt set)

Goal:
Create the main prompt set used for headline results.  
This set must satisfy methodological preconditions.

Filters:
- Prompt-length safety. Avoid truncation.
- Candidate letter tokenization sanity for the model.
- Pilot one-token compliance expectation.

Policy:
- Filter raw examples down to those likely to behave correctly.
- Then sample to a fixed size target (for example 3,000) stratified across coarse domains.

Outputs:
- `runs/<run_id>/manifests/pcc_baseline.jsonl`
- `runs/<run_id>/manifests/pcc_robustness.jsonl`
- `runs/<run_id>/manifests/pcc_report.json`

Validator:
- Stratification meets minimum per-domain counts.
- Manifest invariants still hold.

## 16) Stage 9 — Build Common Compatible Core (intersection across models)

Goal:
Ensure cross-model comparisons are not confounded by different filters producing different prompt sets.

Definition:
Common Compatible Core is the intersection of kept prompts across all chosen models.

Gate thresholds (must be fixed before full runs):
- Overall retention ratio at least 0.80
- Per-domain retention at least 0.60

If thresholds fail:
- Stop and do not proceed silently.
- Either reduce ambition (fewer models) or revisit prompting viability.

Outputs:
- `runs/<run_id>/manifests/ccc_baseline.jsonl`
- `runs/<run_id>/manifests/ccc_robustness.jsonl`
- `runs/<run_id>/manifests/ccc_report.json`

## 17) Stage 10 — Freeze PCA sample membership deterministically

Why:
You must fit principal component analysis once per model on a representative sample of hidden vectors pooled across layers.  
You must use the same basis for every layer.  
You must freeze the sampling plan before expensive runs.

Sampling policy:
- Use seeded random sampling or stratified random sampling.
- Never take the first N rows of a grouped file.
- Record membership as prompt identifiers, not just counts.

Define:
- `pca_sample_prompts`: list of prompt_uids selected
- `pca_layers`: either all layers or a fixed subset of layers used for pooling
- `random_seed`: from config

Outputs:
- `runs/<run_id>/pca/<model_id>_sample_membership.json`

Validator:
- Same seed must produce identical membership list.
- The membership file must contain SHA-256 of the source manifest.

## 18) Stage 11 — PCA sample extraction inference

Goal:
Extract full hidden vectors for the sample prompts across layers, without running full-scale inference.

Key detail:
The decision readout position `p` is the last token of the input prompt, right before the answer begins.  
You must extract hidden states at that position for each layer.

Output volume:
This is intentionally small.  
Store full hidden vectors only here.

Outputs:
- `runs/<run_id>/pca/<model_id>_sample_hidden.npz` (or an equivalent binary format)
- `runs/<run_id>/pca/<model_id>_sample_hidden.meta.json`
  - includes prompt_uids, layer indices, hidden dimension, seed, model revision

Validator:
- Shape checks match expected layers and hidden size.
- Re-running produces identical results within float tolerance when seeds and settings match.

## 19) Stage 12 — Fit PCA basis once per model, pooled across layers

Goal:
Fit a single principal component analysis basis per model using pooled sample vectors from multiple layers.

Non-negotiable:
- One PCA basis per model.
- Reuse the same basis for projecting every layer.
- Default component count is 128.

Outputs:
- `runs/<run_id>/pca/<model_id>_pca_basis.npz`
- `runs/<run_id>/pca/<model_id>_pca_basis.meta.json`
  - includes explained variance ratios
  - sample membership hash
  - fitting library version
  - seed

Validator:
- PCA reproducibility test:
  - same sample + same seed => identical basis hash
- Basis hash recorded for the run.

## 20) Stage 13 — Full inference runner with layerwise readouts and PCA projection

This is the main expensive stage.  
It must be robust, resumable, and validated.

### 20.1 Inference mechanics

For each (model, prompt_uid):
- Tokenize prompt.
- Run generation to obtain:
  - first generated token id and decoded text
  - full generated text (or capped)
- Independently run a forward pass that yields hidden states per layer for the input sequence.
  - Extract hidden state at position `p` for each layer.
- Compute candidate evidence for A/B/C/D for each layer using a training-free “logit lens” style readout:
  - Apply the model’s final normalization module to the layer hidden state output.
  - Apply the model’s output projection module (language modeling head).
  - Aggregate token logits into A/B/C/D using the precomputed token buckets and log-sum-exp.
  - Convert to candidate-only probabilities and compute entropy and margins.

Then:
- Run deterministic parser on the generated text and store grading outputs.
- Compute trajectory metrics:
  - flip count
  - commitment layer for multiple fixed margin thresholds

Then:
- Project each layer hidden state into 128 dimensions using the frozen PCA basis.
- Store `projected_hidden_128` per layer.

### 20.2 Key engineering decisions to implement correctly

Position `p`:
- `p` is the last token index of the input prompt tokens.
- The prompt must end in `Answer:` plus a trailing space, so `p` is unambiguous.

Layer indexing:
- Store a consistent indexing scheme:
  - layer 0 corresponds to the first transformer block output
  - final layer corresponds to the last transformer block output
- Also store the model’s reported number of layers to prevent mismatches across models.

Batching:
- Implement streaming batches.
- Ensure batch results are identical to batch size 1 results (gate described below).

Resumability:
- Output is append-only JSON lines.
- Every row must have a deterministic resume key:
  - `resume_key = sha256(model_id + prompt_uid)`
- On resume, build an in-memory set of already completed resume keys from the output file header scan.
- Never partially write a row. Write one line atomically.

Sharding:
- Support sharding by:
  - prompt range
  - fixed shard size
  - time budget
- Write a shard completion sentinel file only when the shard is fully complete.

### 20.3 Outputs

Per model:
- `runs/<run_id>/outputs/<model_id>/baseline_outputs.jsonl`
- `runs/<run_id>/outputs/<model_id>/robustness_outputs.jsonl`
- `runs/<run_id>/outputs/<model_id>/run_meta.json`
  - includes model id, revision, dtype, device, basis hash, manifest hash

### 20.4 Canonical generation settings policy

Canonical runs:
- Use unconstrained decoding.
- Use greedy decoding (`do_sample=false`) for determinism unless explicitly declared otherwise in config.
- Use a fixed maximum new tokens value of 24 for baseline and robustness (fixed in config).
- Fix temperature and any sampling-related settings (even if unused under greedy).
- Never tune after looking at results.

If you want a constrained-decoding audit mode:
- Put it in a separate run identifier and label it clearly as a control.
- Do not mix with canonical outputs.

## 21) Stage 13.1 — Correctness gates for inference (must exist and must pass)

Before any full run:
1) Manifest validator pass.
2) Batch-consistency test:
   - Run the same small set of prompts with batch size 1 and batch size 4.
   - Compare:
     - first generated token id
     - candidate logits per layer (within float tolerance)
     - trajectory metrics
3) Resume test:
   - Simulate interruption mid-run.
   - Resume.
   - Final number of unique resume keys must equal manifest rows.
4) PCA reproducibility test:
   - Same seed => identical sample membership list
   - Same sample => identical PCA basis hash

Fail-fast:
If any gate fails, stop and fix.  
Do not proceed to expensive runs.

## 22) Stage 14 — Analysis and reports

Inputs:
- Output JSON lines from all models.
- Manifests.
- PCA basis metadata.
- Parser regression pass report.

Outputs:
- `runs/<run_id>/analysis/`
  - `per_prompt_metrics.parquet` (or comma-separated values)
  - `layerwise_aggregates.parquet`
  - `robustness_comparisons.parquet`
  - `figures/` (portable images)
  - `final_report.json` (structured summary for the paper)

Analysis computations (must be deterministic):
- Coverage:
  - one-token compliance rate
  - deterministic parser resolution rate
  - unresolved rate
- Commitment:
  - distribution of commitment layers for each margin threshold
  - flip count distribution
  - entropy curves across layers
  - margin curves across layers
  - split by correct vs incorrect
- Robustness:
  - wrapper-wise deltas relative to baseline
  - stability of decision trajectories across wrappers for the same example
- Structure:
  - domain clustering metrics over projected vectors
  - separation between correct and incorrect in representation space
  - wrong-choice clustering signals if present
- All plots must be versioned by run_id and contain embedded metadata text in the figure metadata or an adjacent JSON.

Validation:
- Row counts match manifests.
- No duplicate keys.
- No missing layers.
- Basis hash matches.

## 23) Local machine discipline (Mac Mini thermal stability) and GPU cost discipline

Local machine rules:
- One heavy inference process at a time.
- Use conservative thread caps:
  - OpenMP threads 1
  - Math kernel threads 1
- Avoid environment hacks that mask broken linking issues.
- Use time-budgeted shards to support cooldown.
- After each shard, write a shard report and update `STATE.md`.

GPU virtual machine rules:
- Assume root disk is ephemeral.
- Use a persistent attached disk mounted at `/data` for:
  - model cache
  - outputs
  - analysis
- Run a smoke test before full runs.
- Download only final analysis artifacts and validation reports before termination.

## 24) Run bookkeeping: state file and “done” markers

Every stage writes:
- `runs/<run_id>/validation/<stage_name>_report.json`
- A line item appended to `STATE.md`:
  - what ran
  - command line
  - input hashes
  - output hashes
  - pass/fail result
  - what is next

For each stage that produces a file that will be consumed later:
- Write a `.done` sentinel:
  - `runs/<run_id>/sentinels/<stage_name>.done`
- The sentinel must include:
  - output file paths
  - SHA-256 hashes
  - timestamp
  - git commit hash if available

No consumer stage should run unless its dependency sentinel exists and matches hashes.

## 25) Reset policy (avoid mixing protocols)

When restarting from scratch:
- Create a new `run_id`.
- Never reuse output files.
- Keep only:
  - the expensive manifests
  - methodology documents
  - parser regression suite
- Delete or archive:
  - old outputs
  - old PCA bases
  - old analysis artifacts
so no accidental cross-run mixing is possible.

## 26) Decision points (defaults provided, must not be silently changed)

Decision point A: Which three models?
- Default: three open instruction-tuned decoder-only transformer models of similar scale.
- Must be declared in config with revisions.

Decision point B: Primary Core Corpus size.
- Default: 3,000 baseline prompts plus 3,000 * 20 robustness prompts.

Decision point C: PCA sample size.
- Default: 1,000 prompts across layers pooled.
- Must be large enough to stabilize components.

Decision point D: Generation maximum new tokens.
- Default: baseline 24, robustness 24.
- Must be fixed in config.

If any of these defaults are not desired, the agent must ask before coding the final pipeline behavior.

---

# What the agent must build first (recommended execution sequence for coding)

1) Manifest schema + validators + canonicalization report system.  
2) Deterministic parser + regression suite pass.  
3) Option token bucket builder + fail-fast checks.  
4) Minimal inference runner that logs first token and generated text.  
5) Hidden state extraction at position `p` across layers.  
6) Candidate scoring across layers using bucket aggregation.  
7) Batch-consistency gate and resume gate.  
8) PCA sample membership + extraction + basis fit + projection.  
9) Full inference with projection.  
10) Analysis and report generator.

If the agent tries to start by writing the full pipeline without gates and tests, that is a failure of instruction-following.
