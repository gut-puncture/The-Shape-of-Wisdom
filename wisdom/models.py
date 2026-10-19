"""The models studied, pinned to exact Hugging Face revisions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    revision: str
    family: str
    # Gemma-2 needs eager attention: the SDPA path silently drops its attention-logit softcap.
    attn_implementation: str = "sdpa"
    # A randomly initialised copy of `repo`'s architecture (the no-learned-structure control).
    untrained: bool = False


MODELS = {
    m.key: m
    for m in [
        ModelSpec("llama8b", "meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659", "llama"),
        ModelSpec("qwen7b", "Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28", "qwen"),
        ModelSpec("gemma9b", "google/gemma-2-9b-it", "11c9b309abf73637e4b6f9a3fa1e92e615547819", "gemma", "eager"),
        ModelSpec("llama1b", "meta-llama/Llama-3.2-1B-Instruct", "9213176726f574b556790deb65791e0c5aa438b6", "llama"),
        ModelSpec("llama8b_untrained", "meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659", "llama", untrained=True),
    ]
}
