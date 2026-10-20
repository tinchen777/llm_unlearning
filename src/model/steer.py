"""Inference-time steering: add coeff * r_UV to the residual stream after one decoder layer, i.e. the effect a
perfect W_down edit (`model.edit`) of that layer would have. Used by the LUNAR-style layer selection.

Reference: LUNAR (arXiv:2502.07218), https://github.com/facebookresearch/LUNAR
"""

from __future__ import annotations
import torch
from typing import Any, Optional, Sequence


class Steerer:
    """Add `coeff * r` to the residual stream after decoder layer `layer` (= `block_out`, where r_UV lives),
    i.e. what a perfectly trained W_down of that layer would do, without training anything.

    Batches must be LEFT padded prompt-only batches (as for generation), so the end-of-instruction tokens of
    every row are the last columns of the prompt. Per batch, call `set(r, prompt_len)` with
    `r` [bsz, n_pos, d] (row i -> r_UV of the author of sample i) and the prompt width; `set(None)` disables.

    - `mode="eoi"` : position p of the prompt gets `r[:, p]` (prompt / prefill pass only); the generated tokens and
                     any appended answer tokens are untouched. Same targets as `build_shift(mode="eoi")`.
    - `mode="all"` : every token (prompt, answer, generated) gets `r[:, ref_position]`, as LUNAR's released code.
    """

    def __init__(
        self,
        model: Any,
        layer: int,
        positions: Sequence[int],
        mode: str = "eoi",
        ref_position: int = -1,
        coeff: float = 1.0
    ):
        if mode not in ("eoi", "all"):
            raise ValueError(f"steering mode must be `eoi` or `all`, got `{mode}`.")
        if mode == "all" and ref_position not in positions:
            raise ValueError(f"ref_position={ref_position} is not one of the positions {list(positions)}.")
        from model.activations import get_decoder_layers

        self.module = get_decoder_layers(model)[layer]
        self.positions = list(positions)
        self.mode, self.coeff = mode, float(coeff)
        self.ref_idx = self.positions.index(ref_position) if mode == "all" else None
        self.r: Optional[torch.Tensor] = None
        self.prompt_len = 0
        self._handle = None

    def set(self, r: Optional[torch.Tensor], prompt_len: int = 0):
        self.r, self.prompt_len = r, int(prompt_len)

    def _hook(self, module, args, output):
        if self.r is None:
            return None
        hidden = output[0] if isinstance(output, tuple) else output
        r = self.r.to(device=hidden.device, dtype=hidden.dtype)
        if r.shape[0] != hidden.shape[0]:
            raise ValueError(f"r has {r.shape[0]} rows for a batch of {hidden.shape[0]}.")
        hidden = hidden.clone()
        if self.mode == "all":
            hidden += self.coeff * r[:, self.ref_idx].unsqueeze(1)
        elif hidden.shape[1] >= self.prompt_len:  # prompt (prefill) pass; decoding steps have seq_len 1
            cols = [self.prompt_len + p for p in self.positions]
            hidden[:, cols] += self.coeff * r
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    def __enter__(self):
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.set(None)
