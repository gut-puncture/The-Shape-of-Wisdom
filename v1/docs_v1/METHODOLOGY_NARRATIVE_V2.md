# Shape of Wisdom — Methodology & Paper Narrative (Unified, Rigorous, Readable)

Last updated: 2026-02-13  
Owner: Shailesh + Nira

---

## 0) One-paragraph summary

This study asks: **how and when LLMs form decisions under surface-level prompt perturbations**. We use one fixed prompt set across baseline and perturbations, run unconstrained generation, judge correctness with a deterministic parser plus DeepSeek adjudication only for unresolved cases, and separately analyze internal decision dynamics from layer-wise answer-option scores at the first decoding step **using option-token invariance** (surface forms like `A`/`a`, leading-space tokens, unicode variants, punctuation-wrapped forms are treated as the same option). The paper narrative is: **(1) behavior changes under perturbations, (2) internal convergence/commitment shifts explain those behavioral changes, (3) topology reveals where these shifts live in representation space**. Response-format variability is treated as a measured diagnostic, not as the core decision signal.

---

## 1) What we are trying to prove (and not prove)

### Core claim
Surface-level prompt changes can alter model decision formation: they shift commitment timing, confidence concentration, and representational geometry, which then appears as correctness drops.

### Not the claim
- Not claiming models always express decisions in token 1.
- Not claiming every wrong answer is due to format artifacts.
- Not claiming constrained decoding behavior equals natural behavior.

---

## 2) First-principles decomposition (the confound problem)

We separate three variables:

1. **Decision variable**: which option the model internally prefers.
2. **Expression variable**: how/when that preference is verbalized (A, 8, explanation, delayed answer, etc.).
3. **Evaluation variable**: whether final answer is correct.

If these are mixed, convergence claims break. This design keeps them distinct and then links them causally in analysis.

---

## 3) Design principles

1. **Single anchor prompt set** for all analyses (baseline + all perturbations).
2. **No constrained decoding in primary runs** (avoid artificial entropy compression).
3. **Same scoring protocol across conditions** (deterministic parser + adjudication for unresolved only).
4. **Mechanistic analyses use internal decision signals**, not raw output formatting.
5. **Format behavior is reported explicitly** as a diagnostic (answer-position lag, first-token compliance).

---

## 4) Data, models, and prompt variants

### Models
- Qwen2.5-7B-Instruct
- Llama-3.1-8B-Instruct
- Mistral-7B-Instruct-v0.3

### Prompt set
- One fixed anchor set of 3000 base MCQ items (keyed by `example_id`).
- Baseline condition: one prompt per `example_id` with `wrapper_id=plain_exam`.
  - Manifest: `data/experiment_inputs/main_prompts.jsonl`
- Robustness conditions: multiple surface-level wrapper rewrites per `example_id` (target is 20 wrappers).
  - Manifest: `data/experiment_inputs/robustness_prompts_v2.jsonl`
- Exactly the same 3000 `example_id`s must be used across baseline and perturbations.

### Why this matters
This preserves direct comparability for every baseline-vs-perturbed pair at prompt level.
It also allows us to define unambiguous joins by `(model_name, example_id, wrapper_id)` and avoid prompt-id collisions.

---

## 5) Generation and logging protocol

For each (model, prompt_id, condition):

1. Run **unconstrained** generation.
2. Log full response text and first generated token.
3. Log layer-wise internal features at first decoding step:
   - **per-option bucketed score traces** (A/B/C/D) invariant to option token surface forms
   - margins, entropy, argmax trajectory
   - representation vectors (for mapping/topology)

### Canonical decoding settings (fixed for the run)
- Greedy decoding (`do_sample=false`) for determinism.
- `max_new_tokens = 24` for baseline and robustness.

### Prompt boundary (required for one-token viability + mechanistic position `p`)
All prompts used for inference must end with literal `Answer: ` (note the trailing space). This:
1. Makes the first generated token very likely to be one of the option letters (often as a leading-space token like `" A"`).
2. Makes the decision readout position `p` unambiguous: the last input token right before the answer begins.

### Why
- Full response is needed for correctness adjudication.
- First-step layer traces are needed for convergence/commitment topology claims.

### Representation vectors and PCA (why `projected_hidden` exists)
Hidden states are high-dimensional and expensive to store for 60k prompts * 3 models.
We therefore store a compressed representation:
- `projected_hidden_128`: per-layer hidden vector projected into a 128D PCA subspace.

This is used only for mapping/topology style analyses (for example `scripts/analyze_mapping.py`).
It is not used for:
- option-token invariant scoring (A/B/C/D trajectories)
- correctness judgement / deterministic salvage
- commitment / reversal / entropy metrics

PCA is still part of the scientific claims because it affects geometry results.
Therefore PCA fitting must be done in a way that is representative and auditable.

### PCA policy (rigor requirements)
For each model, the PCA basis must be:
1. Fit on a representative subset of prompts
   - Seeded random sampling or stratified sampling (not "first N rows" from a grouped file)
   - Target PCA sample size: **1000 prompt instances**
2. Auditable
   - Record the seed, sampling strategy, and the exact sampled prompt identifiers in metadata
3. Consistent across conditions (recommended)
   - Fit one basis per model on a pooled sample drawn from both baseline and robustness prompts
   - Reuse that same PCA basis for projecting baseline and robustness for that model (required for comparable geometry)

---

## 6) Correctness judgement protocol (same for baseline and perturbations)

### Stage 1: Deterministic parser (primary)
Parse robustly for:
- letter forms (`A`, `(A)`, `Option A`, `A is correct`, etc.)
- numeric forms (exact numeric options, leading numeric matches, pi/π)
- conflicts marked unresolved, never force-resolved

### Stage 2: DeepSeek adjudication (fallback only)
Apply only on unresolved/conflicting cases with strict rubric + JSON output.

Model selection gate (cost/rigor):
- Run a small evaluation on ~20 unresolved cases using `deepseek-chat`.
- If accuracy is acceptable, use `deepseek-chat` for adjudication; otherwise use `deepseek-reasoner`.
- For `deepseek-reasoner`, use `max_new_tokens=4000` and explicitly instruct the model to keep reasoning short.
- Store raw judge responses for full auditability.

### Outputs
- `judged_choice` in {A,B,C,D,None}
- `judged_correct` in {True, False, None}
- `judgement_source` in {deterministic, deepseek}

### Why
Deterministic first preserves auditability; DeepSeek fallback recovers difficult tail cases at low cost.

---

## 7) Convergence and commitment definitions (mechanistic)

We analyze layer-wise first-step decision trajectories (not post-hoc text formatting), using an **option-token invariant** scoring probe:

**Option-token invariance:** each option (A/B/C/D) is represented as a *bucket* of token IDs whose decoded form normalizes to that option (case/whitespace/unicode/punctuation robust). Per-layer option scores are aggregated from the bucket logits with a principled reducer (logsumexp), then normalized across options.

1. **Convergence index** (per layer): concentration of option probability mass (e.g., 1 - normalized entropy).
2. **Commitment layer**: earliest layer where eventual winner stabilizes with margin threshold.
3. **Reversal count**: number of winner flips across layers.
4. **Early wrong lock-in / late correction** diagnostics.

### Why this is valid
These metrics measure **internal decision formation**. Expression lag is tracked separately, so we do not confuse “thought” with “formatting”.

---

## 8) Topology and mapping

From hidden vectors (same forward pass), build:
- domain-wise geometry summaries
- trajectory maps across layers
- cluster separation and collapse behavior

### Why
If perturbations change decision dynamics, topology should show where that instability emerges in representation space.

---

## 9) Response-format diagnostics (explicitly measured)

These are diagnostic/validity checks, not primary estimands:

1. First-token option compliance rate.
2. Earliest token position where answer becomes parseable (`<=1`, `<=5`, `<=10`, `>10`/missing).
3. Baseline vs perturbation shift in answer-position lag.

### Why
This quantifies expression artifacts and ensures readers see how often they can affect naive first-token-only evaluation.

---

## 10) Analysis sequence (paper narrative flow)

### Section A — Behavioral effect
Show baseline vs perturbation correctness deltas on the anchor set (judged responses).

### Section B — Decision dynamics mechanism
Show that perturbations alter convergence/commitment trajectories (same prompt IDs).

### Section C — Geometric mechanism
Show topology/mapping shifts align with where decision dynamics destabilize.

### Section D — Expression diagnostics
Show first-token/lag behavior; demonstrate it is measured and controlled, not ignored.

### Section E — Synthesis
Conclude: perturbations impact both behavior and internal decision pathways; format effects exist but do not explain away the core mechanism.

---

## 11) Possible outcomes and what they mean

### Outcome 1
- Correctness drops under perturbations
- Commitment becomes later / less stable
- Topology becomes less separated

**Interpretation:** strong evidence that perturbations disrupt decision formation itself.

### Outcome 2
- Correctness stable
- Commitment/topology stable

**Interpretation:** model is robust to tested perturbations.

### Outcome 3
- Correctness drops
- Commitment/topology mostly stable
- Big answer-position lag shift

**Interpretation:** expression/readout fragility dominates; internal decision may be less affected.

### Outcome 4
- Correctness stable
- Commitment/topology shift

**Interpretation:** compensatory behavior; internal process changes without endpoint failure.

---

## 12) Reporting rules for rigor + readability

1. Use one anchor set in all headline plots.
2. Always report sample counts at each stage:
   - total anchor rows
   - deterministic resolved
   - adjudicated fallback
   - unresolved residual (if any)
3. Keep mechanistic and behavioral metrics distinct, then connect them in synthesis.
4. Keep response-format diagnostics in one compact subsection/table.
5. Put parser/adjudication details and prompts in appendix for reproducibility.

---

## 13) Threats to validity and mitigations

1. **Parser bias** -> deterministic rules + unresolved fallback + audit files.
2. **Judge model bias** -> use only for unresolved tail, fixed rubric, low-variance JSON schema.
3. **Prompt-set bias** -> domain retention reports and sensitivity runs (CCC/PCC comparisons).
4. **Resource limits** -> sequential heavy runs + strict validation per stage.
5. **PCA sampling bias** -> PCA sample must be representative (seeded random/stratified) and recorded; reuse PCA basis across conditions.

---

## 14) Implementation handoff (what engineering must produce)

1. Unified manifests for anchor prompt IDs and condition pairing.
2. Judgement artifacts with per-row provenance (`deterministic` vs `deepseek`).
3. Mechanistic metrics per row/layer from first decoding step.
4. Diagnostic answer-position metrics.
5. Final merged analysis table keyed by `(model, prompt_id, condition)` for all plots/tables.

---

## 15) Short version for the paper intro/method paragraph

“We study robustness of LLM decision formation under surface-level prompt perturbations. For a fixed prompt set used across all conditions, we evaluate endpoint correctness using a deterministic parser with adjudication only for unresolved outputs, and independently analyze layer-wise first-step option trajectories to quantify convergence and commitment. We then link these dynamics to representational topology shifts. This separates internal decision formation from output-format variability while preserving prompt-level comparability across all analyses.”
