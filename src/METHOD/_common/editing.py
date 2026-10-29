"""Pieces of the W_down editing pipeline shared by LUNAR and NEAR (the methods differ in r_UV and in the layer selection).

- `train_w_down_layers`          : train (AdamW or closed form) W_down of every layer on the recorded forget / retain tokens
- `make_response_writer`         : write the responses of fixed samples of some splits (before / after the edit)
- `resolve_candidate_layers`     : candidate layers of a layer selection (explicit list / window of relative depths / all)
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from model.arch import get_out_proj
from model.intervene.edit import solve_w_down_closed_form, train_w_down, w_down_mse

logger = logging.getLogger(__name__)


def train_w_down_layers(
    model: Any,
    layers: Sequence[int],
    forget_tokens: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    retain_tokens: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    wd: Any,
    seed: int,
    solver: str = "adam",
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, torch.Tensor]]:
    """Train W_down of every layer on its recorded (x, target) tokens (layers are independent of each other).

    `wd` is the `wdown` config block (lr, lr_gamma, epochs, batch_size, weight_decay, forget_weight, retain_weight, ridge).
    Returns `(per_layer, new_weights)`: the MSE of the forget / retain tokens before / after (also after the cast to the
    model dtype) and the training history per layer, and the new W_down per layer (NOT yet written into the model).
    """
    if solver not in ("adam", "closed_form"):
        raise ValueError(f"solver must be `adam` or `closed_form`, got `{solver}`.")
    device = model.device
    per_layer, new_weights = {}, {}
    for _l in layers:
        w0 = get_out_proj(model, _l).weight.detach()
        ft, rt = forget_tokens[_l], retain_tokens[_l]
        logger.info(f"layer {_l}: W_down {tuple(w0.shape)} ({solver})")
        if solver == "closed_form":
            w, history = solve_w_down_closed_form(
                w0, ft, rt, ridge=float(wd["ridge"]),
                forget_weight=float(wd["forget_weight"]), retain_weight=float(wd["retain_weight"]), device=device,
            )
        else:
            w, history = train_w_down(
                w0, ft, rt, lr=float(wd["lr"]), epochs=int(wd["epochs"]), batch_size=int(wd["batch_size"]),
                lr_gamma=float(wd["lr_gamma"]), weight_decay=float(wd["weight_decay"]),
                forget_weight=float(wd["forget_weight"]), retain_weight=float(wd["retain_weight"]),
                device=device, seed=seed,
            )
        w_dev = w.to(device)  # the MSEs of the (large) token sets are evaluated on the device, not on the cpu
        w_round = w_dev.to(w0.dtype).float()  # what the model will really use after the cast
        w0_f = w0.float().to(device)
        per_layer[_l] = {
            "forget_mse_before": w_down_mse(w0_f, *ft), "forget_mse_after": w_down_mse(w_dev, *ft),
            "forget_mse_after_rounded": w_down_mse(w_round, *ft),
            "retain_mse_before": w_down_mse(w0_f, *rt), "retain_mse_after": w_down_mse(w_dev, *rt),
            "retain_mse_after_rounded": w_down_mse(w_round, *rt),
            "history": history,
        }
        new_weights[_l] = w
    return per_layer, new_weights


def resolve_candidate_layers(sel: Any, available: Sequence[int]) -> List[int]:
    """Candidate layers of a layer selection: `select.layers` (explicit), else `select.layer_fraction` (window of relative
    depths, e.g. [0.4, 0.8]: LUNAR picks layers at 50-75 % of the depth, appendix E), else every available layer."""
    candidates = sel.get("layers", None)
    if candidates is None and sel.get("layer_fraction", None) is not None:
        lo, hi = (float(x) for x in sel["layer_fraction"])
        n_total = max(available) + 1
        candidates = [_l for _l in available if lo * n_total <= _l <= hi * n_total]
        if not candidates:
            raise ValueError(f"select.layer_fraction={[lo, hi]} selects no layer of {list(available)}.")
    candidates = list(available) if candidates is None else [int(_l) for _l in candidates]
    if not set(candidates) <= set(available):
        raise ValueError(f"select.layers={candidates} not in the available layers: {list(available)}.")
    return candidates
