# GPU Run Notes (RTX 6000 Ada, Ubuntu 22.04)

Goals:
- Keep all heavy artifacts (HF cache, runs outputs, analysis) on the attached persistent disk.
- Run a tiny Stage 13 smoke (20 prompts) before starting full 63k inference.
- Download only analysis + validation artifacts back to the Mac mini (not full per-prompt JSONL).

## Required Environment
- CUDA-capable PyTorch (`torch.cuda.is_available() == True`)
- HF auth: set `HUGGINGFACE_HUB_TOKEN` (or `HF_TOKEN`) in your shell
- Persistent disk mounted at `/data` (recommended)
- Python packages required by this pipeline:
  - `transformers`, `huggingface_hub`, `pyyaml`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`, `sentencepiece`, `safetensors`

Quick install (if base image is missing packages):
```bash
python3 -m pip install --upgrade pip
python3 -m pip install --no-cache-dir \
  transformers huggingface_hub pyyaml pandas matplotlib scipy \
  scikit-learn sentencepiece safetensors
```

## Recommended Layout On GPU VM
- Repo clone: `/data/shape-of-wisdom` (or `/home/ubuntu/shape-of-wisdom`)
- Runs root: `/data/shape-of-wisdom-runs`
- HF cache: `/data/hf`

## One-Time Setup (example)
```bash
export HUGGINGFACE_HUB_TOKEN="...your token..."
export HF_HOME=/data/hf
export TRANSFORMERS_CACHE=/data/hf/transformers
export HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TORCH_HOME=/data/torch
export SOW_RUNS_ROOT=/data/shape-of-wisdom-runs

# Keep CPU noise down (spec section 23).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

## Data Inputs
The repo does **not** include paid inputs (`data/` is gitignored). You must copy these onto the GPU VM:
- `data/experiment_inputs/main_prompts.jsonl`
- `data/experiment_inputs/robustness_prompts_v2.jsonl`
- `data/experiment_inputs/build_summary.json`
- `configs/coarse_domain_mapping.json` (in repo)

Also required for parser regression stage:
- `artifacts/parser_edge_case_regression/regression_cases.json`
  - If missing, `sow.py parser-regression` fails early.

## Run Scripts
- `scripts/gpu/run_smoke_20.sh`: run Stage 0..12 + Stage 13 smoke (20 prompts)
  - Optional mode: `baseline_only` skips robustness smoke.
- `scripts/gpu/run_full.sh`: run full baseline + robustness inference + analysis
  - Optional mode: `baseline_only` runs only baseline inference + baseline-only mechanistic analysis.
- `scripts/gpu/pack_analysis_bundle.sh`: bundle analysis artifacts for download
- `scripts/gpu/fetch_analysis_bundle.sh`: run locally to scp the small analysis bundle back from the GPU VM

All scripts assume:
- you run them from the repo root
- runs are written on attached disk (`SOW_RUNS_ROOT`)
  - if unset, preflight now defaults to:
    - `/data/shape-of-wisdom-runs` when `/data` exists
    - otherwise `/workspace/shape-of-wisdom-runs`

## OOM Backoff (No-Idle Full Runs)
`scripts/gpu/run_full.sh` now uses OOM-aware batch backoff for inference stages:
- default retry chain: `16,12,8,6,4,2,1`
- on CUDA OOM, it retries automatically with the next smaller batch
- non-OOM failures still stop immediately (to avoid hiding real bugs)

Override chain if needed:
```bash
export SOW_BATCH_RETRY_CHAIN=12,8,6,4,2,1
bash scripts/gpu/run_full.sh <run_id> baseline_only
```

## Do Not Skip Preflight
Always run through `scripts/gpu/run_smoke_20.sh` / `scripts/gpu/run_full.sh` (or `source scripts/gpu/preflight.sh && sow_preflight`) before manual stage commands.

Why:
- loads HF token from `/root/.secrets/hf_token.txt` when present
- validates gated Llama access before long runs
- pins caches to attached disk (`/workspace` or `/data`)
- validates required inputs exist

Skipping preflight commonly causes:
- gated-repo 401 errors for Llama
- duplicate model downloads into default cache paths
- missing `SOW_RUNS_ROOT` and wrong run directory resolution

## Stage 13 Determinism Notes
Decoder-only models are sensitive to padding semantics in batched generation:
- right padding can cause first-token drift across batch sizes
- left padding without explicit `position_ids` can also drift in forward-pass readouts

Current fix in `src/sow/inference/stage13.py`:
- use left padding for Stage 13
- derive and pass explicit `position_ids` from `attention_mask` for forward and generate

If Stage 13 smoke fails with batch-consistency mismatches:
1. confirm GPU has latest `src/sow/inference/stage13.py`
2. rerun smoke with preflight env (not ad-hoc `python3 sow.py ...` without exports)
3. inspect `validation/stage13_smoke_report*.json` and `batch_consistency_gate` fields
4. treat `max_abs_diff_candidate_logits` as telemetry unless `enforce_candidate_logits_atol=true`;
   baseline gate pass/fail should primarily follow:
   - structural parity (`mismatches` empty)
   - `max_abs_diff_candidate_probs <= atol_probs`
   - no high-margin top-candidate flips (`hard_margin_flip_count == 0`)

## Download Only Analysis Artifacts
After `run_full.sh` finishes, a bundle is written under:
- `$SOW_RUNS_ROOT/<run_id>/bundles/analysis_bundle_<run_id>.tar.gz`

Examples:
```bash
# Full pipeline (baseline + robustness)
bash scripts/gpu/run_smoke_20.sh <run_id> full
bash scripts/gpu/run_full.sh <run_id> full

# Faster mechanistic path (baseline only; convergence/commitment/topology)
bash scripts/gpu/run_smoke_20.sh <run_id> baseline_only
bash scripts/gpu/run_full.sh <run_id> baseline_only
```

Fetch it to the Mac mini (or any local machine) with:
```bash
bash scripts/gpu/fetch_analysis_bundle.sh ubuntu@<gpu-ip> <run_id> /data/shape-of-wisdom-runs .
tar -xzf "analysis_bundle_<run_id>.tar.gz"
```
