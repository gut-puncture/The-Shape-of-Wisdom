# Shape of Wisdom — Thermal Policy

Goal: keep the Mac mini productive without sustained thermal stress.

## Default rules
- One heavy model process at a time.
- **Between heavy stages:** 20-minute breather.
- **Within long heavy stages:** run ~90 minutes, then cool 15–20 minutes, then resume.

## Implementation
Use `scripts/run_inference.py` with:
- `--resume`
- `--time-budget-seconds 5400` (90 minutes)
- `--done-sentinel <path>` (written only when all prompts complete)

And wrap it with:
- `scripts/run_with_cooldowns_loop.sh --run-seconds 5400 --cooldown-seconds 1200 ...`

This keeps progress durable (JSONL append + resume-by-prompt-id) while reducing sustained heat.
