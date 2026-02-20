from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


def iter_decoder_layers(model_obj: Any) -> List[Any]:
    core = getattr(model_obj, "model", None)
    layers = getattr(core, "layers", None)
    if isinstance(layers, (list, tuple)):
        return list(layers)
    if hasattr(layers, "__iter__"):
        return list(layers)
    raise ValueError("unable to resolve decoder layers via model.layers")


@dataclass
class ComponentTrace:
    attn_outputs: List[Any] = field(default_factory=list)
    mlp_outputs: List[Any] = field(default_factory=list)


class ComponentTracer:
    """Capture per-layer attention and MLP outputs via forward hooks."""

    def __init__(self, model_obj: Any) -> None:
        self._model_obj = model_obj
        self._hooks: List[Any] = []
        self.trace = ComponentTrace()

    @staticmethod
    def _first_tensor(output: Any) -> Any:
        if isinstance(output, tuple):
            return output[0]
        return output

    def _attn_hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        self.trace.attn_outputs.append(self._first_tensor(output))

    def _mlp_hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        self.trace.mlp_outputs.append(self._first_tensor(output))

    def __enter__(self) -> "ComponentTracer":
        for layer in iter_decoder_layers(self._model_obj):
            self._hooks.append(layer.self_attn.register_forward_hook(self._attn_hook))
            self._hooks.append(layer.mlp.register_forward_hook(self._mlp_hook))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def capture_component_outputs(
    *,
    model_obj: Any,
    input_ids: Any,
    attention_mask: Any,
    position_ids: Any | None = None,
    output_attentions: bool = True,
) -> dict[str, Any]:
    with ComponentTracer(model_obj) as tracer:
        out = model_obj(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
            output_attentions=bool(output_attentions),
            return_dict=True,
        )

    return {
        "model_output": out,
        "attn_outputs": tracer.trace.attn_outputs,
        "mlp_outputs": tracer.trace.mlp_outputs,
    }
