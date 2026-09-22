"""Activation access: layer resolution, forward hooks and position pooling.

Shared by the layer-selection stage and by the trainer, so that "the activation
of question q at layer l" means exactly the same thing in both.

Position strategies (following LUNAR's choice of acting on the prompt region):

* `last_prompt` - the last token before the answer starts. This is the token the
  answer is generated from, i.e. where the entity representation is read out.
* `prompt_mean` - mean over the prompt tokens.
* `answer`      - the answer tokens (`labels != IGNORE_INDEX`).
* `last`        - the last non-padding token.
* `all`         - every non-padding token.

A neighbour question carries no answer, so `answer` falls back to `last_prompt`
for the target side; this is handled by `pool_activations(..., is_target=True)`.
"""

from __future__ import annotations
from contextlib import contextmanager
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
POSITION_STRATEGIES = ("last_prompt", "prompt_mean", "answer", "last", "all")


# --------------------------------------------------------------------------
# module resolution
# --------------------------------------------------------------------------
def unwrap_model(model: Any) -> Any:
    """Strip DeepSpeed / DDP / accelerate wrappers to reach the nn.Module."""
    for attr in ("module", "_orig_mod"):
        inner = getattr(model, attr, None)
        if inner is not None and hasattr(inner, "named_modules"):
            return unwrap_model(inner)
    return model


def get_layer_list(model: Any) -> Any:
    """The `nn.ModuleList` of transformer blocks (Llama, Mistral, Qwen, ...)."""
    base = unwrap_model(model)
    for path in ("model.layers", "model.model.layers", "transformer.h", "gpt_neox.layers"):
        obj: Any = base
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__"):
            return obj
    raise ValueError(
        f"Could not locate the transformer block list on {type(base).__name__}. "
        "Pass explicit module names instead."
    )


def num_layers(model: Any) -> int:
    return len(get_layer_list(model))


def layer_module_name(model: Any, layer: int) -> str:
    """Fully-qualified name of block `layer`, e.g. `model.layers.7`."""
    base = unwrap_model(model)
    blocks = get_layer_list(model)
    target = blocks[layer]
    for name, module in base.named_modules():
        if module is target:
            return name
    raise ValueError(f"Layer {layer} not found in {type(base).__name__}")


def get_layer_modules(model: Any, layers: Sequence[int]) -> List[Any]:
    blocks = get_layer_list(model)
    n = len(blocks)
    out = []
    for layer in layers:
        index = layer if layer >= 0 else n + layer
        if not 0 <= index < n:
            raise IndexError(f"Layer {layer} out of range for a {n}-layer model.")
        out.append(blocks[index])
    return out


def get_modules_by_regex(model: Any, pattern: str) -> Dict[str, Any]:
    base = unwrap_model(model)
    matched = {
        name: module for name, module in base.named_modules() if re.fullmatch(pattern, name)
    }
    if not matched:
        raise ValueError(f"No module matched `{pattern}`.")
    return matched


def down_proj_regex(model: Any, layers: Sequence[int]) -> List[str]:
    """Regexes selecting the MLP down-projection of the given layers (LUNAR's knob)."""
    names = [layer_module_name(model, layer) for layer in layers]
    return [rf"{re.escape(name)}\.mlp\.down_proj\.weight" for name in names]


# --------------------------------------------------------------------------
# hooks
# --------------------------------------------------------------------------
@contextmanager
def capture_activations(modules: Sequence[Any]):
    """Capture the hidden states produced by `modules` during a forward pass.

    Yields a dict `{module_position: tensor}` filled in on each forward call;
    the tensors keep their graph, so the trainer can back-propagate through them.
    """
    cache: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(position: int):
        def hook(_module, _inputs, output):
            cache[position] = output[0] if isinstance(output, tuple) else output
            return None
        return hook

    try:
        for position, module in enumerate(modules):
            handles.append(module.register_forward_hook(make_hook(position)))
        yield cache
    finally:
        for handle in handles:
            handle.remove()


def forward_with_activations(
    model: Any,
    batch: Dict[str, torch.Tensor],
    modules: Sequence[Any],
    grad: bool = True,
    ignore_labels: bool = True,
) -> Tuple[Dict[int, torch.Tensor], Any]:
    """Forward `batch` and return `({position: hidden_states}, outputs)`."""
    drop = ("index", "labels") if ignore_labels else ("index",)
    inputs = {k: v for k, v in batch.items() if k not in drop and isinstance(v, torch.Tensor)}
    with capture_activations(modules) as cache:
        with torch.set_grad_enabled(grad):
            outputs = model(**inputs)
        return dict(cache), outputs


# --------------------------------------------------------------------------
# position selection & pooling
# --------------------------------------------------------------------------
def position_mask(
    labels: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    strategy: str = "last_prompt",
    is_target: bool = False,
) -> torch.Tensor:
    """Boolean mask `[b, s]` of the positions the redirection acts on.

    Args:
        labels: `[b, s]` with `IGNORE_INDEX` on prompt (and padding) positions.
        attention_mask: `[b, s]`; assumed all-ones when omitted.
        strategy: one of `POSITION_STRATEGIES`.
        is_target: neighbour side, which has no answer tokens; `answer` then
            degrades to `last_prompt`.
    """
    if strategy not in POSITION_STRATEGIES:
        raise ValueError(f"Unknown position strategy `{strategy}` (expected {POSITION_STRATEGIES}).")
    if labels is None and attention_mask is None:
        raise ValueError("position_mask needs labels or attention_mask.")

    if attention_mask is None:
        attention_mask = torch.ones_like(labels)  # type: ignore[arg-type]
    attended = attention_mask.bool()

    if strategy == "all":
        return attended
    if strategy == "last":
        return _last_true(attended)

    if labels is None:
        # prompt-only batch: every attended token belongs to the prompt
        if strategy == "prompt_mean":
            return attended
        return _last_true(attended)

    answer = (labels != IGNORE_INDEX) & attended
    if strategy == "answer" and not is_target:
        return answer if answer.any() else _last_true(attended)

    prompt = (labels == IGNORE_INDEX) & attended
    if strategy == "prompt_mean":
        return prompt if prompt.any() else attended

    # last_prompt (and the target-side fallback of `answer`)
    first_answer = _first_true(answer)
    mask = _shift_left(first_answer)
    # rows without an answer token (or whose answer starts at 0): last prompt token
    empty = ~mask.any(dim=1)
    if empty.any():
        fallback = _last_true(prompt if prompt.any() else attended)
        mask = torch.where(empty.unsqueeze(1), fallback, mask)
    return mask


def _first_true(mask: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(mask)
    has = mask.any(dim=1)
    index = torch.argmax(mask.int(), dim=1)
    out[has, index[has]] = True
    return out


def _last_true(mask: torch.Tensor) -> torch.Tensor:
    flipped = torch.flip(mask.int(), dims=[1])
    index = mask.shape[1] - 1 - torch.argmax(flipped, dim=1)
    out = torch.zeros_like(mask)
    has = mask.any(dim=1)
    out[has, index[has]] = True
    return out


def _shift_left(mask: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(mask)
    out[:, :-1] = mask[:, 1:]
    return out


def pool_activations(
    hidden: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    strategy: str = "last_prompt",
    is_target: bool = False,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean-pool `hidden` `[b, s, d]` over the selected positions -> `[b, d]`."""
    if mask is None:
        mask = position_mask(labels, attention_mask, strategy=strategy, is_target=is_target)
    weights = mask.to(hidden.dtype).unsqueeze(-1)
    total = (hidden * weights).sum(dim=1)
    count = weights.sum(dim=1).clamp(min=1.0)
    return total / count


# --------------------------------------------------------------------------
# batched extraction (evaluation / layer selection)
# --------------------------------------------------------------------------
@torch.no_grad()
def extract_activations(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    layers: Sequence[int],
    strategy: str = "last_prompt",
    batch_size: int = 8,
    max_length: int = 512,
    apply_chat_template: bool = True,
    system_prompt: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = torch.float32,
) -> Dict[int, torch.Tensor]:
    """Pooled activations of `prompts` at each layer -> `{layer: [N, d]}`.

    Prompts are formatted exactly like a question at inference time (chat
    template + generation prompt), which is the representation an attacker sees.
    """
    modules = get_layer_modules(model, layers)
    device = device or str(getattr(model, "device", "cpu"))
    was_training = model.training
    model.eval()

    collected: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}
    side = getattr(tokenizer, "padding_side", "right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        for start in range(0, len(prompts), batch_size):
            chunk = list(prompts[start : start + batch_size])
            texts = [
                _format_prompt(tokenizer, p, apply_chat_template, system_prompt) for p in chunk
            ]
            tokenizer.padding_side = "right"
            batch = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True,
                max_length=max_length, add_special_tokens=False,
            ).to(device)
            cache, _ = forward_with_activations(model, dict(batch), modules, grad=False)
            mask = position_mask(None, batch["attention_mask"], strategy=strategy, is_target=True)
            for position, layer in enumerate(layers):
                pooled = pool_activations(cache[position], mask=mask)
                collected[layer].append(pooled.to(dtype or pooled.dtype).cpu())
    finally:
        tokenizer.padding_side = side
        if was_training:
            model.train()

    return {layer: torch.cat(chunks, dim=0) for layer, chunks in collected.items()}


def _format_prompt(
    tokenizer: Any,
    prompt: str,
    apply_chat_template: bool,
    system_prompt: Optional[str],
) -> str:
    if not apply_chat_template or not getattr(tokenizer, "chat_template", None):
        return prompt
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def iter_batches(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
