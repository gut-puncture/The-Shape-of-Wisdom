# Model Nuances V2 (HF-Documented)

This file records model-specific decisions applied in the V2 pipeline.

## Sources
- [Qwen2.5-7B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Llama-3.1-8B-Instruct model card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [Mistral-7B-Instruct-v0.3 model card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [Transformers Qwen2 docs](https://huggingface.co/docs/transformers/model_doc/qwen2)
- [Transformers Llama docs](https://huggingface.co/docs/transformers/model_doc/llama)
- [Transformers Mistral docs](https://huggingface.co/docs/transformers/model_doc/mistral)
- [Transformers generation with LLMs (padding-side guidance)](https://huggingface.co/docs/transformers/llm_tutorial)

## Implemented decisions
1. Left padding is enforced for decoder-only batched inference.
2. Explicit `position_ids` are derived from `attention_mask` for batch-stable token positioning.
3. If tokenizer has no `pad_token_id`, we set it to `eos_token_id`.
4. Tracing forces eager attention implementation when attention tensors are required.
5. On MPS we force `float16` for consistency and memory safety.
6. Pinned model revisions are kept in `configs/experiment_v2.yaml`; decoding is deterministic (`do_sample=false`, fixed seeds).

## Version floor reminders from model cards
- Qwen2.5: Transformers >= 4.37.0
- Llama 3.1: Transformers >= 4.43.0
- Mistral v0.3: Transformers >= 4.42.0
