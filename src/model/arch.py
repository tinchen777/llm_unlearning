
from __future__ import annotations
from torch import nn
from functools import wraps
from typing import Any, Callable, Dict, Literal, Tuple, TypeVar


def get_decoder_layers(model: Any) -> nn.ModuleList:
    """The decoder-layer ModuleList, found structurally (works through PEFT/DDP wrappers)."""
    n_layers = model.config.num_hidden_layers
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) == n_layers and name.endswith("layers"):
            return module
    raise ValueError(f"Cannot find the decoder layers (ModuleList of length {n_layers}) in {type(model).__name__}.")


_MLP_OUT_PROJ_NAMES = ("down_proj", "fc2", "c_proj", "dense_4h_to_h")
_ATTN_OUT_PROJ_NAMES = ("o_proj", "out_proj", "c_proj", "dense")


def _first_attr(obj: nn.Module, names: Tuple[str, ...], desc: str) -> nn.Module:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise ValueError(f"No {desc} {names} in {type(obj).__name__}.")


def _post_attn_norm(layer: nn.Module) -> nn.Module:
    if not hasattr(layer, "post_attention_layernorm"):
        raise ValueError(
            f"`post_attn_norm` needs a sequential block with `post_attention_layernorm`; "
            f"{type(layer).__name__} is a parallel attention/MLP block."
        )
    return layer.post_attention_layernorm


_LAYER_MODULES: Dict[str, Callable[[nn.Module], nn.Module]] = {
    "block": lambda layer: layer,
    "attn": lambda layer: layer.self_attn,
    "attn_out_proj": lambda layer: _first_attr(layer.self_attn, _ATTN_OUT_PROJ_NAMES, "attention output projection"),
    "mlp": lambda layer: layer.mlp,
    "mlp_out_proj": lambda layer: _first_attr(layer.mlp, _MLP_OUT_PROJ_NAMES, "MLP output projection"),
    "post_attn_norm": _post_attn_norm,
}

# point_name = (module_name, "input" or "output")
POINTS: Dict[str, Tuple[str, Literal["input", "output"]]] = {
    "block_out": ("block", "output"),
    "mlp_out": ("mlp", "output"),
    "down_proj_in": ("mlp_out_proj", "input"),
    "resid_mid": ("post_attn_norm", "input"),
}


def resolve_point(layer: nn.Module, point: str) -> Tuple[nn.Module, str]:
    """Returns (module, reads_input) of `point` in a decoder layer.

    `side` is "input" when the point is the INPUT of the module, "output" when it is its OUTPUT.
    """
    if point not in POINTS:
        raise ValueError(f"Unknown point `{point}`, choose from {tuple(POINTS)}.")
    name, side = POINTS[point]
    return _LAYER_MODULES[name](layer), side


def get_layer_module(model: Any, layer_idx: int, name: str) -> nn.Module:
    """The module `name` (one of `_LAYER_MODULES`) of decoder layer `layer_idx`."""
    if name not in _LAYER_MODULES:
        raise ValueError(f"Unknown layer module `{name}`, choose from {tuple(_LAYER_MODULES)}.")
    return _LAYER_MODULES[name](get_decoder_layers(model)[layer_idx])


def get_out_proj(model: Any, layer_idx: int) -> nn.Linear:
    """The MLP output projection (`down_proj` / `fc2`) of decoder layer `layer_idx`."""
    return get_layer_module(model, layer_idx, "mlp_out_proj")  # type: ignore


F = TypeVar("F", bound=Callable[..., Any])


def eval_mode(func: F) -> F:
    """Temporarily set the model to evaluation mode."""

    @wraps(func)
    def wrapper(model: Any, *args, **kwargs):
        was_training = model.training
        model.eval()

        try:
            return func(model, *args, **kwargs)
        finally:
            model.train(was_training)

    return wrapper  # type: ignore[return-value]
