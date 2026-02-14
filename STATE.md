# Shape of Wisdom - STATE (Template)

This file is the single source of truth for:
1. What is running now (if anything)
2. What is known to be correct/implemented (and validated)
3. What is known to be wrong/unsafe to run
4. The next concrete steps to return to a rigorous, cost-aware run

Update rules (non-negotiable):
- Append-only: add a new dated entry; do not rewrite prior entries except to fix obvious typos.
- After every stage, record: command, inputs (hashes), outputs (hashes), validator results, and the next step.
- If anything is unclear or contradictory, stop and resolve before spending GPU money.

Canonical references:
- `docs/IMPLEMENTATION_SPEC.md` (engineering contract + stage gates)
- `docs/METHODOLOGY_NARRATIVE_V2.md` (scientific narrative; if present)
- `PROJECT_PLAN.md` (stable checklist; must remain run-agnostic)

## Active Run Snapshot (fill only while something is running)
- run_id:
- machine: (local / GPU VM)
- start_time_local:
- current_stage:
- command:
- expected_stop_condition:
- outputs_dir:

## Frozen Decisions (must not change mid-run)
- models: (model_id + revision + dtype)
- dataset build: (source, version, coarse mapping file hash)
- manifests: (baseline + robustness manifest hashes)
- generation settings: (max_new_tokens, temperature, sampling policy)
- judging policy: (deterministic parser + LLM fallback rules, if any)
- PCA policy:
  - sample membership file path + hash:
  - basis file path(s) + hash(es):
- resume key definition:

## Current Blockers / Risks
- (none)

## Latest Gate Results (quick summary)
- Stage 0 (env lock + smoke test):
- Stage 1 (examples build):
- Stage 2 (baseline manifest):
- Stage 3 (robustness manifest):
- Stage 4 (manifest validation/canonicalization):
- Stage 5 (option token buckets):
- Stage 6 (parser regression suite):
- Stage 7 (pilot inference viability):
- Stage 8 (PCC build):
- Stage 9 (CCC build):
- Stage 10 (freeze PCA membership):
- Stage 11 (PCA sample extraction):
- Stage 12 (fit PCA basis):
- Stage 13 (full inference):
- Stage 14 (analysis + reports):

## Append-Only Log

### YYYY-MM-DD HH:MM (local) - Stage X - PASS/FAIL
- command:
- inputs (paths + SHA-256):
- outputs (paths + SHA-256):
- validators (paths + PASS/FAIL):
- notes:
- next:


### 2026-02-14 12:47 (local) - Stage 0 - init-run-config - PASS
- command: python3 sow.py init-run --run-id m1_20260214_124722 --seed 12345
- inputs (paths + SHA-256):
  - (none)
- outputs (paths + SHA-256):
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/run_config.yaml: 7134bcc6bc996d696ec152e0a9113775e430d34f45c7a4fec71caa15cc77e096
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/meta/config_snapshot.yaml: 7134bcc6bc996d696ec152e0a9113775e430d34f45c7a4fec71caa15cc77e096
- validators (paths + PASS/FAIL):
  - (none)
- notes: Pinned model revisions; greedy decoding; max_new_tokens=24; PCA sample size=1000.
- next: Stage 2/3/4 - build-manifests

### 2026-02-14 12:47 (local) - Stage 2/3/4 - build+canonicalize+validate manifests - PASS
- command: python3 sow.py build-manifests --run-id m1_20260214_124722
- inputs (paths + SHA-256):
  - /Users/shaileshrana/shape-of-wisdom/data/experiment_inputs/main_prompts.jsonl: 2f8608ce26fcc7091ad185ba74675d3fa6e61f974132e8055637ea847824bbbc
  - /Users/shaileshrana/shape-of-wisdom/data/experiment_inputs/robustness_prompts_v2.jsonl: d39835d1123dcc1e7b65fd83f9043a764c39ec9ab641b796d343f9f1c4457d44
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/run_config.yaml: 7134bcc6bc996d696ec152e0a9113775e430d34f45c7a4fec71caa15cc77e096
- outputs (paths + SHA-256):
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/manifests/baseline_manifest.jsonl: 90f9f9f6c1e0097ee74e71ff26dff517d444a0a7eca4063d2a9b3bf8876e8219
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/manifests/robustness_manifest_v2.jsonl: 57c676b3ead7627b5d720c0aacdba1284925cc84fee83bff063d724c87ce085d
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/manifests/baseline_manifest.meta.json: 94753e94b3c38892979b5addce5b726d9ed0412622beb03c8fb9b57730ff8917
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/manifests/robustness_manifest_v2.meta.json: d747ec43f412717b6798ef76c3aa2021a4b257339cb31f7de39ce7119561aacc
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/manifests/baseline_manifest.canonicalization_report.json: 45d75b2d1371939bfe0ee79d1cafa17b77178c20df8a5d8597742ac3cc95c697
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/manifests/robustness_manifest_v2.canonicalization_report.json: 932db1287b1cf85c77b33469897f97edee54c77d47ca7a725989e732816eeda2
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/validation/manifests_report.json: 9a99e0ed3463bdbe06673000030d657ac0f99d862701a0fdc4aa08b8563dd6c9
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/sentinels/manifests.done: 1d83cb21b440166d4f5c4b519050ae6513195811c66e55bf843ac5bd0f5f553e
- validators (paths + PASS/FAIL):
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/validation/manifests_report.json: 9a99e0ed3463bdbe06673000030d657ac0f99d862701a0fdc4aa08b8563dd6c9
- notes: Robustness v2: keep-last-line for duplicate (example_id, wrapper_id); drop out-of-wrapper-set; repair missing ascii_box for mmlu::test::12183; enforce suffix boundary.
- next: Stage 6 - parser-regression

### 2026-02-14 12:47 (local) - Stage 6 - parser regression suite - PASS
- command: python3 sow.py parser-regression --run-id m1_20260214_124722
- inputs (paths + SHA-256):
  - /Users/shaileshrana/shape-of-wisdom/artifacts/parser_edge_case_regression/regression_cases.json: 5ca9a14648b8e0fe35527fbaeecc08c2f5fe00828e4435f3e5e3f7b0bc4d9533
- outputs (paths + SHA-256):
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/validation/parser_regression_report.json: 221b877a089614c300a9f98a8d38b5a365ee3604ae7f4b3d1aba1d3cb1d6cd41
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/sentinels/parser_regression.done: 55f3c18a6da1b0c88b4befd9fa8941d13997917b746114df9fa3904a9fec7be6
- validators (paths + PASS/FAIL):
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/validation/parser_regression_report.json: 221b877a089614c300a9f98a8d38b5a365ee3604ae7f4b3d1aba1d3cb1d6cd41
- next: Stage 10 - pca-membership

### 2026-02-14 12:48 (local) - Stage 10 - freeze PCA sample membership - PASS
- command: python3 sow.py pca-membership --run-id m1_20260214_124722
- inputs (paths + SHA-256):
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/manifests/baseline_manifest.jsonl: 90f9f9f6c1e0097ee74e71ff26dff517d444a0a7eca4063d2a9b3bf8876e8219
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/manifests/robustness_manifest_v2.jsonl: 57c676b3ead7627b5d720c0aacdba1284925cc84fee83bff063d724c87ce085d
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/run_config.yaml: 7134bcc6bc996d696ec152e0a9113775e430d34f45c7a4fec71caa15cc77e096
- outputs (paths + SHA-256):
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/pca/qwen2.5-7b-instruct_sample_membership.json: 02e6a63b214b599ed388f69c432d6f568c0da7c811f918cace618aaff088675c
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/pca/llama-3.1-8b-instruct_sample_membership.json: 4d6cf19c08a1525d6757b882468d3d67694ba6cbd1f7c71085e49f5b95977d9e
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/pca/mistral-7b-instruct-v0.3_sample_membership.json: d48009c87f7c4521d09d4621143bf217c84a3762743f9a0b71fdffa0b64cc8ae
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/validation/pca_membership_report.json: ebd630a5f7674ac66cd1469da17471908864cef22d01b8d846c6f5d0150c50b2
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/sentinels/pca_membership.done: 655829185da0c68ea08f2dc6703cb09438625b11c84a986250c9e5dada6be472
- validators (paths + PASS/FAIL):
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/validation/pca_membership_report.json: ebd630a5f7674ac66cd1469da17471908864cef22d01b8d846c6f5d0150c50b2
- notes: Membership is stratified uniformly over (wrapper_id, coarse_domain) strata and is deterministic for the frozen seed.
- next: Milestone 1 complete (no PCA fit / no inference yet)

### 2026-02-14 12:49 (local) - Frozen Decisions Snapshot (run_id=m1_20260214_124722) - PASS
- command: derived from /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/run_config.yaml (pinned + hashed)
- inputs (paths + SHA-256):
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/run_config.yaml: 7134bcc6bc996d696ec152e0a9113775e430d34f45c7a4fec71caa15cc77e096
- outputs (paths + SHA-256):
  - (none)
- validators (paths + PASS/FAIL):
  - (none)
- notes: models pinned: Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28; meta-llama/Llama-3.1-8B-Instruct@0e9e39f249a16976918f6564b8830bc894c89659; mistralai/Mistral-7B-Instruct-v0.3@c170c708c41dac9275d15a8fff4eca08d52bab71
- notes: decoding: greedy (do_sample=false), max_new_tokens=24
- notes: robustness wrapper_ids_v2 (exact, 20): academic_abstract, ascii_box, changelog_entry, csv_inline, graphql_query, haiku_riddle, html_form, ini_file, irc_log, key_equals, legal_clause, meeting_minutes, protobuf_msg, quest_briefing, recipe_instruction, regex_match, s_expression, shell_heredoc, toml_config, tweet_thread
- notes: prompt boundary: every inference prompt ends with literal `Answer: ` (trailing space); robustness canonicalization appends `Return only the letter (A, B, C, or D).` + final `Answer: `
- notes: robustness canonicalization: filter wrapper_id to wrapper_ids_v2; for duplicate (example_id, wrapper_id) keep later file occurrence; repair only missing (mmlu::test::12183, ascii_box) via deterministic template
- notes: PCA policy: sample_size=1000, stratified(wrapper_id, coarse_domain) uniform-over-strata, n_components=128; seed=12345; membership files are in /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/pca/
- notes: resume key definition: sha256(model_id + ':' + prompt_uid)
- next: Stage 5 - option token buckets

### 2026-02-14 13:50 (local) - Stage 5 - option token buckets (A/B/C/D) per model - PASS
- command: python3 sow.py token-buckets --run-id m1_20260214_124722
- inputs (paths + SHA-256):
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/run_config.yaml: 7134bcc6bc996d696ec152e0a9113775e430d34f45c7a4fec71caa15cc77e096
- outputs (paths + SHA-256):
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/token_buckets/Qwen__Qwen2.5-7B-Instruct.json: 598bc72008d0e3fee9842a1c1b78a2c28acaa9fcd34314e882387bbd2d1270d2
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/token_buckets/meta-llama__Llama-3.1-8B-Instruct.json: 787cdc48c4f0e4ceb1094906278433f3542410cb15086ed9b02cb6cb9cd84d28
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/token_buckets/mistralai__Mistral-7B-Instruct-v0.3.json: 9bb8576e4e12004194dbef7578e4c136cd25ceb1f005f70f312e2e2956fbe921
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/validation/token_buckets_report.json: a8cac1a8675dcb464379fcc6382afdd394e2e251075f8e337c619b086e3e6ec0
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/sentinels/token_buckets.done: 98f478ec6062526f521b70c74b5a9bb0cb44ac7922e263ea46ac4de567065938
- validators (paths + PASS/FAIL):
  - /Users/shaileshrana/shape-of-wisdom/runs/m1_20260214_124722/validation/token_buckets_report.json: a8cac1a8675dcb464379fcc6382afdd394e2e251075f8e337c619b086e3e6ec0
- notes: Built tokenizer-derived A/B/C/D token buckets with fixed variant templates; fail-fast if any bucket empty or overlaps detected.
- next: Stage 7 - pilot inference (after confirming Stage 6 PASS already)
