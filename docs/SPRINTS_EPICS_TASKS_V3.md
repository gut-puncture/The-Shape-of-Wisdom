# Sprints, Epics, and Tasks V3 (Exhaustive)

## Sprint 0: Archive + Governance Reset
Epic: Deactivate legacy path and establish v3-only governance
- Task: Archive legacy paper directory to `v1/paper_v1`
- Task: Archive legacy methodology/spec docs to `v1/docs_v1`
- Task: Archive legacy scripts to `v1/scripts_v1`
- Task: Create project-local `AGENTS.md` with Anthropic/ICML-NeurIPS quality bar
- Task: Ensure active references point only to v3 docs
- Task: Add migration note for all contributors

## Sprint 1: Spec + Backlog Lock
Epic: Freeze objective, methods contract, and exhaustive backlog
- Task: Create `PAPER_OBJECTIVE_V3.md`
- Task: Create `IMPLEMENTATION_PLAN_V3.md`
- Task: Define acceptance gates and stop conditions
- Task: Define output artifact contract
- Task: Define compute fallback policy (>36h task)
- Task: Define reproducibility contract (seed/hash/version pinning)
- Task: Define reporting contract (methods/results/final report)

## Sprint 2: Data and Metric Foundations
Epic: Build deterministic metric substrate for all downstream analyses
- Task: Extract baseline layerwise records to parquet
- Task: Compute delta, boundary, drift, competitor identity
- Task: Compute flips, commitment, entropy, margins
- Task: Validate schema integrity and null safety
- Task: Add deterministic regression tests for metric calculations
- Task: Add row-count parity checks against source manifests

## Sprint 3: Trajectory Typing and Region Analysis
Epic: Generate phase diagram and basin mechanics
- Task: Implement four-way trajectory classifier
- Task: Produce per-model type counts and balance diagnostics
- Task: Compute basin gaps from projected representations
- Task: Compute entry/exit/re-entry events
- Task: Generate phase diagram and trajectory family plots
- Task: Validate classifier stability under threshold sensitivity

## Sprint 4: Span Taxonomy and Counterfactual Labeling
Epic: Build operational evidence/distractor spans without hand-labeling
- Task: Parse prompt spans (instruction/stem/options/post-options)
- Task: Implement span mutation generator (delete/replace)
- Task: Run counterfactual effects on sampled prompts
- Task: Label spans as evidence/distractor/neutral by effect sign/magnitude
- Task: Compute label stability under paraphrase controls
- Task: Add random-length-matched controls

## Sprint 5: Tracing Instrumentation
Epic: Open the box for attention routing and MLP injection
- Task: Implement decoder-layer hook manager
- Task: Capture attention outputs and MLP outputs per layer
- Task: Compute decision direction vectors
- Task: Compute decision-aligned scalar contributions
- Task: Aggregate attention mass by operational span labels
- Task: Persist compact scalar outputs instead of full vectors where possible
- Task: Validate drift reconstruction fidelity

## Sprint 6: Causal Validation
Epic: Establish causal credibility for mechanism claims
- Task: Implement component ablation simulation/execution path
- Task: Implement activation patching path
- Task: Implement span-deletion causal tests
- Task: Implement negative controls (random layers/spans/shuffles)
- Task: Estimate effect sizes and uncertainty intervals
- Task: Flag failed causal directionality checks

## Sprint 7: Runtime and Thermal Safety
Epic: Enforce Mac-first safety and deterministic resumability
- Task: Extend thermal governor with `pause_mode`
- Task: Add `checkpoint_exit` semantics and event logging
- Task: Patch Stage13 to invoke thermal governance in loop
- Task: Add thermal resume supervisor script
- Task: Add throughput-based runtime estimator
- Task: Add backend decision gate (>36h => GPU-only task)
- Task: Preserve deterministic resume keys across restarts

## Sprint 8: Orchestration and Asset Packaging
Epic: Run whole pipeline from one orchestrator and package final artifacts
- Task: Build `scripts/v2/00_run_experiment.py`
- Task: Add mode support (`smoke`, `single_model`, `full`)
- Task: Standardize run outputs under `runs/<run_id>/v2`
- Task: Generate required figures/tables
- Task: Generate methods and results markdown
- Task: Build final artifact folder and SHA256 manifest

## Sprint 9: Test-First and Quality Gates
Epic: Enforce behavior via tests before production runs
- Task: Add v2 unit tests for metrics, typing, spans, tracing, causal tools, runtime policy, thermal policy
- Task: Add v2 smoke integration test with tiny synthetic run
- Task: Extend stage13 tests for thermal-governor invocation
- Task: Ensure all tests pass deterministically
- Task: Add reproducibility test pass (seeded hash parity)

## Sprint 10: Review, Re-Review, Re-Review
Epic: three-pass bug elimination protocol
- Task: Review pass 1 (correctness + contract compliance)
- Task: Review pass 2 (adversarial edge cases + failure injection)
- Task: Review pass 3 (reproducibility + deterministic replay)
- Task: Log unresolved risks explicitly if any remain
- Task: Block merge unless all critical findings resolved

## Missed Tasks Pass 1
- Task: Add explicit schema versioning for every parquet/json artifact
- Task: Add artifact lineage metadata linking each file to source hashes
- Task: Add fail-fast check for stale/cross-run file mixing
- Task: Add CLI-level `--dry-run` plan validation mode
- Task: Add top-level changelog for v3 pipeline evolution

## Missed Tasks Pass 2
- Task: Add guardrails for tokenizer drift across model revision changes
- Task: Add float dtype consistency checks across MPS/CPU paths
- Task: Add plot generation determinism checks (fixed style + seeds)
- Task: Add data leakage checks in any learned decomposition models
- Task: Add explicit confidence interval method registry in methods output

## Missed Tasks Pass 3
- Task: Add automatic rerun recommendation engine for failed gates
- Task: Add artifact completeness verifier before final bundle publication
- Task: Add machine profile capture (thermal/memory/runtime) per stage
- Task: Add dead-letter queue handling for partially failed stage outputs
- Task: Add one-command “final audit” report that summarizes all gate statuses
