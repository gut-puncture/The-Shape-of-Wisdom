# GPU Run Notes (RTX 6000 Ada, Ubuntu 22.04)

Goals:
- Keep all heavy artifacts (HF cache, runs outputs, analysis) on the attached persistent disk.
- Run a tiny Stage 13 smoke (20 prompts) before starting full 63k inference.
- Download only analysis + validation artifacts back to the Mac mini (not full per-prompt JSONL).

## Required Environment
- CUDA-capable PyTorch (`torch.cuda.is_available() == True`)
- HF auth: set `HUGGINGFACE_HUB_TOKEN` (or `HF_TOKEN`) in your shell
- Persistent disk mounted at `/data` (recommended)

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

## Run Scripts
- `scripts/gpu/run_smoke_20.sh`: run Stage 0..12 + Stage 13 smoke (20 prompts)
- `scripts/gpu/run_full.sh`: run full baseline + robustness inference + analysis
- `scripts/gpu/pack_analysis_bundle.sh`: bundle analysis artifacts for download
- `scripts/gpu/fetch_analysis_bundle.sh`: run locally to scp the small analysis bundle back from the GPU VM

All scripts assume:
- you run them from the repo root
- `SOW_RUNS_ROOT` is set to a location on the attached disk

## Download Only Analysis Artifacts
After `run_full.sh` finishes, a bundle is written under:
- `$SOW_RUNS_ROOT/<run_id>/bundles/analysis_bundle_<run_id>.tar.gz`

Fetch it to the Mac mini (or any local machine) with:
```bash
bash scripts/gpu/fetch_analysis_bundle.sh ubuntu@<gpu-ip> <run_id> /data/shape-of-wisdom-runs .
tar -xzf "analysis_bundle_<run_id>.tar.gz"
```
