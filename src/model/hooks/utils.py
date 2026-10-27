
from __future__ import annotations
import torch
from contextlib import contextmanager
from typing import Any, Dict, Sequence

from ..arch import get_decoder_layers, resolve_point


@contextmanager
def capture_activations(model: Any, layers: Sequence[int], point: str = "block_out"):
    """Context manager caching `point` activations of the given layers during forward.

    Yields a dict `{layer_idx: Tensor[bsz, seq_len, d]}`, refilled on every forward pass.
    """
    def _as_tensor(x: Any) -> torch.Tensor:
        return x[0] if isinstance(x, (tuple, list)) else x

    decoder_layers = get_decoder_layers(model)
    cache: Dict[int, torch.Tensor] = {}
    handles = []
    for layer_idx in layers:
        module, side = resolve_point(layer=decoder_layers[layer_idx], point=point)
        if side == "input":
            # input side
            def pre_hook(mod, args, kwargs, _l=layer_idx):
                cache[_l] = _as_tensor(args[0] if args else kwargs["hidden_states"]).detach()
            handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        else:
            # output side
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
