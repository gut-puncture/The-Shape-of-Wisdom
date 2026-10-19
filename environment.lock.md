# Environment Lock

This file records the intended behavior for the first locked execution pass.

- Python target: `3.11.x` or `3.12.x`
- Base execution policy is authored in `REQUIREMENTS.md`, `ENVIRONMENT.md`, and the pinned model
  and dataset revisions in `configs/`
- Exact GPU execution lock remains deferred until the target hardware and installed package set are
  frozen for the paper-facing run
- Current local environment is documented in `ENVIRONMENT.md`
- Full sweep remains blocked until the environment lock is finalized
