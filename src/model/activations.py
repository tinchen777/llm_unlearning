"""Per-layer activation extraction via forward hooks (Trainer-independent).

Hook points of decoder layer `l` (Llama / Qwen2 / Phi layouts):

- `block_out`    : residual stream after block `l` (= LUNAR's pre-hook on block `l+1`),
                   i.e. the residual right after the MLP output of layer `l` is added.
                   This is the activation LUNAR's unlearning vector r_UV lives in.
- `mlp_out`      : output of `layer.mlp` (the MLP's contribution, down_proj output).
- `down_proj_in` : input of the MLP output projection (`down_proj` / `fc2`), the
                   input LUNAR trains the modified down_proj on.
- `resid_mid`    : residual after attention, before the MLP (input of
                   `post_attention_layernorm`); Llama/Qwen2 only, Phi runs
                   attention and MLP in parallel so it has no such point.

For sequential blocks: `block_out == resid_mid + mlp_out`.

Token positions are negative offsets from each row's LAST REAL token, located via
`attention_mask`, so extraction is correct under both left and right padding.
LUNAR uses the post-instruction template tokens, see `get_eoi_positions` in `METHOD/NEAR/template.py`.

Reference: LUNAR (arXiv:2502.07218), https://github.com/facebookresearch/LUNAR
"""

from __future__ import annotations
import logging
import torch
from torch import nn
from contextlib import contextmanager
from tqdm import tqdm
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .utils import batch_to_device

logger = logging.getLogger(__name__)

HOOK_POINTS = ("block_out", "mlp_out", "down_proj_in", "resid_mid")
_MLP_OUT_PROJ_NAMES = ("down_proj", "fc2", "c_proj", "dense_4h_to_h")


def get_decoder_layers(model: Any) -> nn.ModuleList:
    """The decoder-layer ModuleList, found structurally (works through PEFT/DDP wrappers)."""
    n_layers = model.config.num_hidden_layers
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) == n_layers and name.endswith("layers"):
            return module
    raise ValueError(f"Cannot find the decoder layers (ModuleList of length {n_layers}) in {type(model).__name__}.")


def _get_hook_module(layer: nn.Module, point: str) -> Tuple[nn.Module, bool]:
    """Returns (module, is_pre_hook) to capture `point` of a decoder layer."""
    if point == "block_out":
        return layer, False
    if point == "mlp_out":
        return layer.mlp, False
    if point == "down_proj_in":
        for name in _MLP_OUT_PROJ_NAMES:
            if hasattr(layer.mlp, name):
                return getattr(layer.mlp, name), True
        raise ValueError(f"No MLP output projection {_MLP_OUT_PROJ_NAMES} in {type(layer.mlp).__name__}.")
    if point == "resid_mid":
        if not hasattr(layer, "post_attention_layernorm"):
            raise ValueError(
                f"`resid_mid` needs a sequential block with `post_attention_layernorm`; "
                f"{type(layer).__name__} is a parallel attention/MLP block."
            )
        return layer.post_attention_layernorm, True
    raise ValueError(f"Unknown hook point `{point}`, choose from {HOOK_POINTS}.")


def get_out_proj(model: Any, layer_idx: int) -> nn.Linear:
    """The MLP output projection (`down_proj` / `fc2`) of decoder layer `layer_idx`."""
    return _get_hook_module(get_decoder_layers(model)[layer_idx], "down_proj_in")[0]  # type: ignore


def _as_tensor(x: Any) -> torch.Tensor:
    return x[0] if isinstance(x, (tuple, list)) else x


@contextmanager
def capture_activations(model: Any, layers: Sequence[int], point: str = "block_out"):
    """Context manager caching `point` activations of the given layers during forward.

    Yields a dict `{layer_idx: Tensor[bsz, seq_len, d]}`, refilled on every forward pass.
    """
    decoder_layers = get_decoder_layers(model)
    cache: Dict[int, torch.Tensor] = {}
    handles = []
    for layer_idx in layers:
        module, is_pre = _get_hook_module(decoder_layers[layer_idx], point)
        if is_pre:
            def pre_hook(mod, args, kwargs, _l=layer_idx):
                cache[_l] = _as_tensor(args[0] if args else kwargs["hidden_states"]).detach()
            handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        else:
            def fwd_hook(mod, args, output, _l=layer_idx):
                cache[_l] = _as_tensor(output).detach()
            handles.append(module.register_forward_hook(fwd_hook))
    try:
        yield cache
    finally:
        for handle in handles:
            handle.remove()


def gather_positions(
    acts: torch.Tensor,
    attention_mask: torch.Tensor,
    positions: Sequence[int]
) -> torch.Tensor:
    """Select token `positions` (negative offsets from each row's last real token).

    acts: [bsz, seq_len, d]; attention_mask: [bsz, seq_len] -> returns [bsz, n_pos, d].
    """
    if any(p >= 0 for p in positions):
        raise ValueError(f"positions must be negative offsets from the last real token, got {list(positions)}.")
    mask = attention_mask.to(acts.device).bool()
    seq_idx = torch.arange(mask.shape[1], device=acts.device)
    last = torch.where(mask, seq_idx, -1).max(dim=1).values  # [bsz], last real token (any padding side)
    first = torch.where(mask, seq_idx, mask.shape[1]).min(dim=1).values  # [bsz], first real token
    offsets = torch.tensor(list(positions), device=acts.device)
    idx = last.unsqueeze(1) + 1 + offsets.unsqueeze(0)  # [bsz, n_pos]
    if (idx < first.unsqueeze(1)).any():
        raise ValueError(f"positions {list(positions)} reach beyond the start of a sequence.")
    return acts.gather(1, idx.unsqueeze(-1).expand(-1, -1, acts.shape[-1]))


@torch.no_grad()
def collect_mean_activations(
    model: Any,
    dataloader: Iterable[Mapping[str, Any]],
    positions: Sequence[int] = (-1,),
    layers: Optional[Sequence[int]] = None,
    point: str = "block_out",
    desc: str = "",
):
    """Collect `point` activations at `positions` for every layer in `layers`, returning the dataset-mean activations.

    Args:
        positions: sequence of positions to gather activations from (default: `(-1,)` for the last position).
        layers: sequence of layer indices to collect activations from (default: all layers).
        point: the point in the model to capture activations from (default: `"block_out"`).
        desc: description for the progress bar (default: empty string).

    Returns:
        (mean-activations, layers), where:
            - `mean-activations`: Tensor[n_pos, n_layers, d] (float64, CPU), the dataset-mean activations.
            - `layers`: list of layer indices.
    """
    layers = list(range(model.config.num_hidden_layers)) if layers is None else list(layers)
    was_training = model.training
    model.eval()

    total: Optional[torch.Tensor] = None
    n_samples = 0
    with capture_activations(model, layers, point) as cache:
        for batch in tqdm(dataloader, desc=f"Collecting `{point}` activations [{desc}]", unit="batch(es)", colour="green"):
            if "input_ids" not in batch:
                raise ValueError(f"Expected a flat batch with `input_ids`, got keys {list(batch)}.")
            inputs = batch_to_device({k: batch[k] for k in ("input_ids", "attention_mask")}, model.device)
            model(**inputs)
            # [bsz, n_pos, n_layers, d]
            acts = torch.stack(
                [gather_positions(cache[l], inputs["attention_mask"], positions).cpu() for l in layers],
                dim=2
            )
            batch_sum = acts.to(torch.float64).sum(dim=0)
            total = batch_sum if total is None else total + batch_sum
            # update
            n_samples += acts.shape[0]

    if was_training:
        model.train()
    if n_samples == 0 or total is None:
        raise ValueError("The dataloader yielded no samples.")

    return total / n_samples, layers


@torch.no_grad()
def collect_sample_activations(
    model: Any,
    dataloader: Iterable[Mapping[str, Any]],
    positions: Sequence[int] = (-1,),
    layers: Optional[Sequence[int]] = None,
    point: str = "block_out",
    desc: str = "",
    meta_keys: Optional[Sequence[str]] = None,
):
    """Collect `point` activations at `positions` for every layer in `layers`, one tensor PER SAMPLE (no averaging).

    Args:
        positions: sequence of positions to gather activations from (default: `(-1,)` for the last position).
        layers: sequence of layer indices to collect activations from (default: all layers).
        point: the point in the model to capture activations from (default: `"block_out"`).
        desc: description for the progress bar (default: empty string).
        meta_keys: batch metadata keys to record per sample (default: `None`).

    Returns:
        (activations, layers, per_meta), where:
            - `activations`: `{dataset_index: Tensor[n_pos, n_layers, d]}` (model dtype, CPU), per-sample activations.
            - `layers`: list of layer indices.
            - `per_meta`: `{dataset_index: {key: value}}` containing the requested metadata for each sample.
    """
    layers = list(range(model.config.num_hidden_layers)) if layers is None else list(layers)
    was_training = model.training
    model.eval()

    n_samples = 0
    per_sample: Dict[int, torch.Tensor] = {}
    per_meta: Dict[int, Dict[str, Any]] = {}
    with capture_activations(model, layers, point) as cache:
        for batch in tqdm(dataloader, desc=f"Collecting `{point}` activations [{desc}]", unit="batch(es)", colour="green"):
            if "input_ids" not in batch:
                raise ValueError(f"Expected a flat batch with `input_ids`, got keys {list(batch)}.")
            indices = batch.get("index")
            inputs = batch_to_device({k: batch[k] for k in ("input_ids", "attention_mask")}, model.device)
            model(**inputs)
            # [bsz, n_pos, n_layers, d]
            acts = torch.stack(
                [gather_positions(cache[l], inputs["attention_mask"], positions).cpu() for l in layers],
                dim=2
            )
            if indices is None:
                raise ValueError("Collecting sample-wise activations needs the dataset `index` in each batch.")
            for i, idx in enumerate(torch.as_tensor(indices).tolist()):
                per_sample[int(idx)] = acts[i]
                if meta_keys:
                    per_meta[int(idx)] = {
                        k: (batch[k][i].item() if isinstance(batch[k], torch.Tensor) else batch[k][i])
                        for k in meta_keys if k in batch
                    }
            n_samples += acts.shape[0]

    if was_training:
        model.train()
    if n_samples == 0:
        raise ValueError("The dataloader yielded no samples.")

    return per_sample, layers, per_meta
