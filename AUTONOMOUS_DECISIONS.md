# Autonomous Decisions

This file records implementation decisions made while the user was unavailable.

## March 27, 2026

1. I encoded the approved preregistered positivity rule at the `model × dataset` level rather
   than inventing a pooled across-dataset binary rule.
   Why: the user explicitly approved the per-dataset version of rule 1 and rule 2, but did not
   approve any stronger pooled claim.
2. I kept the smoke/dev panel on `qwen2p5_0p5b_instruct` and `smollm2_360m_instruct`.
   Why: both are already locally validated and small enough to keep the default smoke path
   reproducible on CPU-friendly hardware.
3. I switched the paper-facing panel to `Qwen + Gemma + Llama` with exact revisions pinned in
   `configs/models.yaml`.
   Why: this matches the user-approved paper family direction and removes the weaker provisional
   `OLMo` family from the main plan.
4. I treated `Gemma` and `Llama` access gates as explicit blockers instead of substituting
   alternative families.
   Why: silent model substitution would weaken the scientific panel definition.
5. I left every paper-panel model `family_ready: false` unless it was already validated locally.
   Why: full-sweep execution should fail closed until readiness checks pass on the actual target
   environment.
6. I deferred the clean root-snapshot rerun.
   Why: the user asked for that rerun only when the live root log writer is quiesced, and that
   condition has not been re-established yet.
7. I made `run_full_sweep.py` refuse to run unless positivity is authored and every paper-panel
   model resolves as both enabled and `family_ready`.
   Why: the project requirement is to block expensive runs until the rigorous gates are explicit.
8. I added `--timeout-seconds` to `run_family_readiness.py` after a paper-family probe spent too
   long in gated resolution.
   Why: a readiness tool that can hang on access-gated models is not rigorous enough for unattended
   execution.
