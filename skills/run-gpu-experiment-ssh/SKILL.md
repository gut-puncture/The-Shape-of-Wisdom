---
name: run-gpu-experiment-ssh
description: Run large SSH-based GPU experiments safely and efficiently with a local-first workflow, strict preflight checks, stable caching on attached disk, deterministic smoke gates, and fast recovery from partial failures.
---

# GPU SSH Experiment Blueprint

Use this skill when running experiments on remote GPUs over SSH (for example RunPod-style pods) where mistakes are expensive in time and bandwidth.

## Outcomes

After following this process, you should have:
- reproducible runs under attached-disk storage (not ephemeral root cache)
- smoke validation before full inference
- explicit failure triage with known root causes and fixes
- small downloadable artifact bundles for analysis and plotting

## Golden Rules

1. Do as much as possible locally first.
2. Always run GPU work through preflight-aware scripts.
3. Never run long inference without HF token + runs root + cache path set.
4. Keep the remote payload minimal and explicit.
5. Treat smoke as a hard gate before full runs.

## Local-First Workflow

### 1) Validate locally before touching GPU

Run local tests first:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

If tests fail locally, do not start GPU runs.

### 2) Prepare a minimal transfer set

Send code + required inputs only. At minimum include:
- `sow.py`
- `src/`
- `scripts/`
- `configs/`
- `tests/` (optional but recommended for remote sanity checks)
- `data/experiment_inputs/main_prompts.jsonl`
- `data/experiment_inputs/robustness_prompts_v2.jsonl`
- `data/experiment_inputs/build_summary.json`
- `artifacts/parser_edge_case_regression/regression_cases.json`

Missing `regression_cases.json` causes `parser-regression` failure.

## SSH + Disk Setup

### 1) Connect

```bash
ssh -i ~/Downloads/private_key.pem -p <PORT> root@<GPU_IP>
```

### 2) Use attached disk only

For RunPod-style mounts this is usually `/workspace`.

Recommended layout:
- repo: `/workspace/shape-of-wisdom`
- runs: `/workspace/shape-of-wisdom-runs`
- hf cache: `/workspace/hf`
- torch cache: `/workspace/torch`

## Dependency Bootstrap (Remote)

Run once per fresh pod image:

```bash
cd /workspace/shape-of-wisdom
python3 -m pip install --upgrade pip
python3 -m pip install --no-cache-dir \
  transformers huggingface_hub pyyaml pandas matplotlib scipy \
  scikit-learn sentencepiece safetensors
```

`scikit-learn` is required for `pca-fit`.

## HF Token Handling (Critical)

Gated Llama models require auth. Store token on GPU:

```bash
mkdir -p /root/.secrets
chmod 700 /root/.secrets
printf '%s' "$HF_TOKEN_VALUE" > /root/.secrets/hf_token.txt
chmod 600 /root/.secrets/hf_token.txt
```

Do not echo token values in logs.

## Always Use Preflight

Preferred entrypoints:
- `scripts/gpu/run_smoke_20.sh`
- `scripts/gpu/run_full.sh`

Or explicitly:

```bash
cd /workspace/shape-of-wisdom
export SOW_RUNS_ROOT=/workspace/shape-of-wisdom-runs
source scripts/gpu/preflight.sh
sow_preflight
```

Preflight enforces:
- `SOW_RUNS_ROOT` exists on attached disk
- GPU + CUDA availability
- gated Llama access
- required input files
- cache defaults under `/workspace` or `/data`

## Canonical Run Sequence

### 1) Smoke (baseline-only fast path)

```bash
cd /workspace/shape-of-wisdom
export SOW_RUNS_ROOT=/workspace/shape-of-wisdom-runs
bash scripts/gpu/run_smoke_20.sh <run_id> baseline_only
```

### 2) Full baseline-only inference + analysis

```bash
cd /workspace/shape-of-wisdom
export SOW_RUNS_ROOT=/workspace/shape-of-wisdom-runs
bash scripts/gpu/run_full.sh <run_id> baseline_only
```

### 3) Bundle and fetch only small artifacts

On GPU:
```bash
bash scripts/gpu/pack_analysis_bundle.sh <run_id>
```

Locally:
```bash
bash scripts/gpu/fetch_analysis_bundle.sh \
  root@<GPU_IP> <run_id> /workspace/shape-of-wisdom-runs .
```

## Known Failure Modes and Fixes

### 1) `missing run config` under `/workspace/shape-of-wisdom/runs/...`

Cause:
- command executed without `SOW_RUNS_ROOT`

Fix:
- export `SOW_RUNS_ROOT=/workspace/shape-of-wisdom-runs`
- prefer run scripts that source preflight

### 2) Gated repo 401 for Llama

Cause:
- missing/unexported HF token in current shell
- manual command path bypassed preflight

Fix:
- ensure `/root/.secrets/hf_token.txt` exists
- run `sow_preflight` before stage commands

### 3) `ModuleNotFoundError: sklearn` at `pca-fit`

Cause:
- missing `scikit-learn`

Fix:
- install `scikit-learn` in pod Python environment

### 4) `FileNotFoundError` for parser regression cases

Cause:
- remote payload omitted `artifacts/parser_edge_case_regression/regression_cases.json`

Fix:
- transfer this file explicitly

### 5) Stage 13 smoke fails batch consistency

Observed root causes:
- decoder-only padding semantics + batched generation mismatch
- manual env paths causing inconsistent cache/model behavior

Required mitigation:
- run latest `src/sow/inference/stage13.py`
- run through preflight env
- verify `validation/stage13_smoke_report*.json`

## Padding/Determinism Lesson (Important)

For decoder-only models in Stage 13:
- right padding can drift first generated token across batch sizes
- left padding without explicit `position_ids` can drift hidden readouts across batch sizes

Current robust approach:
- left padding in Stage 13 tokenizer
- explicit `position_ids` computed from `attention_mask`
- pass those `position_ids` to both forward pass and generation

This is a core source of GPU smoke instability; do not regress this behavior.

## Recovery from Partial Runs

If a run fails mid-pipeline:
1. inspect latest `validation/*.json` and `sentinels/*.done`
2. install missing deps or fix env
3. resume from the failed stage (do not restart from stage0 unless needed)
4. expect `.attemptN` files due append-only behavior

Useful checks:

```bash
ls -1 /workspace/shape-of-wisdom-runs/<run_id>/validation
ls -1 /workspace/shape-of-wisdom-runs/<run_id>/sentinels
```

## Efficient Transfer Pattern

When `rsync` is unavailable locally, use tar-over-SSH:

```bash
COPYFILE_DISABLE=1 tar -cf - \
  sow.py src scripts configs tests \
  data/experiment_inputs/main_prompts.jsonl \
  data/experiment_inputs/robustness_prompts_v2.jsonl \
  data/experiment_inputs/build_summary.json \
  artifacts/parser_edge_case_regression/regression_cases.json \
| ssh -i ~/Downloads/private_key.pem -p <PORT> root@<GPU_IP> \
  'mkdir -p /workspace/shape-of-wisdom && tar --no-same-owner --no-same-permissions -xf - -C /workspace/shape-of-wisdom'
```

## Practical Checklists

### Before Smoke
- local tests pass
- required files transferred
- HF token file present
- dependencies installed
- preflight passes

### Before Full Run
- smoke report exists
- smoke gate is acceptable for your policy
- attached disk has enough free space

### Before Decommissioning GPU
- bundle created under `/workspace/shape-of-wisdom-runs/<run_id>/bundles`
- analysis bundle downloaded locally
- run metadata/report artifacts verified locally

## Example End-to-End (Concrete)

```bash
# local
python3 -m unittest discover -s tests -p 'test_*.py' -v

# ssh
ssh -i ~/Downloads/private_key.pem -p 21228 root@195.26.233.74

# remote
cd /workspace/shape-of-wisdom
export SOW_RUNS_ROOT=/workspace/shape-of-wisdom-runs
bash scripts/gpu/run_smoke_20.sh rtx6000ada_baseline_20260216 baseline_only
bash scripts/gpu/run_full.sh rtx6000ada_baseline_20260216 baseline_only
```

If any command is run manually outside these scripts, replicate preflight exports first.
