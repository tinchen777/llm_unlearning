"""Editing the MLP output projection (W_down) of one decoder layer, LUNAR style (no hydra, no Trainer).

For one decoder layer, with
    x = input of the MLP output projection (down_proj), y = its ORIGINAL output, per token:
        forget tokens : target = y + coeff * shift      (shift built from the unlearning vector r_UV)
        retain tokens : target = y                      (the layer must keep behaving the same)
and `W_down` is trained (MSE) to map x -> target, starting from the original weights: AdamW like LUNAR's code
(`train_w_down`), or the closed form of Eq. 9 (`solve_w_down_closed_form`). Inputs/targets are recorded ONCE from the
original model (`collect_mlp_tokens`); training itself runs no model forward.

Reference: LUNAR (arXiv:2502.07218), https://github.com/facebookresearch/LUNAR
"""

from __future__ import annotations
import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..inference.utils import batch_to_device, get_model_device
from ..hooks.utils import capture_activations
from ..arch import get_out_proj, in_eval_mode

logger = logging.getLogger(__name__)


def build_shift(
    r: torch.Tensor,
    attention_mask: torch.Tensor,
    positions: Sequence[int],
    mode: str,
    ref_position: int = -1,
) -> torch.Tensor:
    """Per-token shift [bsz, seq, d] that is added to the MLP output of forget tokens.

    Args:
        r: [bsz, n_pos, d], the r_UV of each row (row i -> the author of sample i), saved position order.
        positions: negative offsets (from the last real token) the n_pos axis of `r` refers to.
        mode: - "all": every token gets `r[:, ref_position]` (my reading of LUNAR's code: a single direction
                broadcast over the whole sequence);
              - "eoi": position p gets `r[:, p]`, all other tokens get 0 (one vector per end-of-instruction token).
    """
    bsz, seq = attention_mask.shape
    mask = attention_mask.bool().to(r.device)
    cols = torch.arange(seq, device=r.device)
    last = torch.where(mask, cols, -1).max(dim=1).values  # last real token of every row (any padding side)
    shift = r.new_zeros(bsz, seq, r.shape[-1])
    if mode == "all":
        shift[:] = r[:, list(positions).index(ref_position)].unsqueeze(1)
    elif mode == "eoi":
        rows = torch.arange(bsz, device=r.device)
        for j, p in enumerate(positions):
            shift[rows, last + 1 + p] = r[:, j]
    else:
        raise ValueError(f"shift mode must be `all` or `eoi`, got `{mode}`.")
    return shift


@torch.no_grad()
@in_eval_mode
def collect_mlp_tokens(
    model: Any,
    loader: Iterable[Mapping[str, Any]],
    layers: Sequence[int],
    shift_fn: Optional[Callable[[Mapping[str, Any], int, torch.Tensor], torch.Tensor]] = None,
    desc: str = "",
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Record, for every NON-padding token of every sample and every layer in `layers`:

        x : input of the MLP output projection (model dtype)      -> [N_tokens, d_ff]
        t : the MLP output (float32), plus `shift_fn(batch, layer, attention_mask)` if given -> [N_tokens, d]

    One forward pass records all layers. Everything is returned on CPU.
    """
    layers = list(layers)
    xs: Dict[int, List[torch.Tensor]] = {_l: [] for _l in layers}
    ts: Dict[int, List[torch.Tensor]] = {_l: [] for _l in layers}
    with capture_activations(model, layers, "down_proj_in") as cache_in, \
            capture_activations(model, layers, "mlp_out") as cache_out:
        for batch in tqdm(loader, desc=f"Recording MLP inputs/outputs [{desc}]", unit="batch(es)", colour="blue"):
            inputs = batch_to_device({k: batch[k] for k in ("input_ids", "attention_mask")}, get_model_device(model))
            model(**inputs)
            mask = inputs["attention_mask"].bool()
            for _l in layers:
                t = cache_out[_l].float()
                if shift_fn is not None:
                    t = t + shift_fn(batch, _l, inputs["attention_mask"]).to(t)
                xs[_l].append(cache_in[_l][mask].cpu())
                ts[_l].append(t[mask].cpu())
    return {_l: (torch.cat(xs[_l]), torch.cat(ts[_l])) for _l in layers}


@torch.no_grad()
def w_down_mse(w: torch.Tensor, x: torch.Tensor, t: torch.Tensor, chunk: int = 8192) -> float:
    """mean squared error (mean over tokens AND features, like `F.mse_loss`) of `x @ w.T` against `t`."""
    total = 0.0
    for s in range(0, len(x), chunk):
        pred = F.linear(x[s:s + chunk].to(w.device).float(), w)
        total += F.mse_loss(pred, t[s:s + chunk].to(w.device), reduction="sum").item()
    return total / t.numel()


def train_w_down(
    w0: torch.Tensor,
    forget: Tuple[torch.Tensor, torch.Tensor],
    retain: Tuple[torch.Tensor, torch.Tensor],
    lr: float,
    epochs: int,
    batch_size: int,
    lr_gamma: float = 0.9,
    weight_decay: float = 0.0,
    forget_weight: float = 1.0,
    retain_weight: float = 1.0,
    device: Any = "cpu",
    seed: int = 0,
) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
    """Train W_down (starting from `w0` [d, d_ff]) so that `W x` matches the targets.

        loss = forget_weight * MSE(forget tokens) + retain_weight * MSE(retain tokens)

    Every step uses `batch_size` forget tokens (one epoch = one pass over the forget tokens) and
    `batch_size` randomly drawn retain tokens. fp32 master weights; AdamW + ExponentialLR(lr_gamma) per epoch.
    Returns (trained weights [d, d_ff] float32 on CPU, per-epoch history).
    """
    (fx, ft), (rx, rt) = forget, retain
    if len(fx) == 0 or len(rx) == 0:
        raise ValueError(f"Need forget and retain tokens, got {len(fx)} and {len(rx)}.")
    gen = torch.Generator().manual_seed(seed)
    fx, ft, rx, rt = (a.to(device) for a in (fx, ft, rx, rt))
    w = w0.detach().to(device=device, dtype=torch.float32).clone().requires_grad_(True)
    optimizer = torch.optim.AdamW([w], lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

    history: List[Dict[str, float]] = []
    for epoch in range(epochs):
        perm = torch.randperm(len(fx), generator=gen)
        sum_f = sum_r = 0.0
        steps = 0
        for start in range(0, len(fx), batch_size):
            fi = perm[start:start + batch_size].to(device)
            ri = torch.randint(0, len(rx), (len(fi),), generator=gen).to(device)
            loss_f = F.mse_loss(F.linear(fx[fi].float(), w), ft[fi])
            loss_r = F.mse_loss(F.linear(rx[ri].float(), w), rt[ri])
            loss = forget_weight * loss_f + retain_weight * loss_r
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_f += loss_f.item()
            sum_r += loss_r.item()
            steps += 1
        history.append({"epoch": epoch + 1, "forget_mse": sum_f / steps, "retain_mse": sum_r / steps,
                        "lr": scheduler.get_last_lr()[0]})
        scheduler.step()
        logger.info(f"epoch {epoch + 1}/{epochs}  forget_mse={sum_f / steps:.4g}  retain_mse={sum_r / steps:.4g}")
    return w.detach().cpu(), history


@torch.no_grad()
def solve_w_down_closed_form(
    w0: torch.Tensor,
    forget: Tuple[torch.Tensor, torch.Tensor],
    retain: Tuple[torch.Tensor, torch.Tensor],
    ridge: float = 1e-3,
    forget_weight: float = 1.0,
    retain_weight: float = 1.0,
    device: Any = "cpu",
    chunk: int = 8192,
) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
    """Closed-form W_down (LUNAR Eq. 9), no learning rate / epochs to tune.

        min_W   forget_weight * mean_f ||W x - t||^2 + retain_weight * mean_r ||W x - t||^2 + lambda ||W - W0||^2

    is a ridge regression on the residual targets R = t - W0 x (zero for retain tokens, `coeff * shift` for forget
    tokens), solved for the CHANGE of the weights:  dW = R^T X (X^T X + lambda I)^-1.
    Every token has weight `w_set / N_set`, so the two sets weigh like the two MSE terms of `train_w_down`
    (mean over each set), not like their token counts.

    Args:
        ridge: lambda as a FRACTION of mean(diag(X^T X)), i.e. scale free; 0 gives plain least squares (needs more
            tokens than input features, else the Gram matrix is singular). Shrinks the edit towards W0.
    Returns:
        (W [d, d_ff] float32 on CPU, one-entry history with the final per-set MSE)
    """
    (fx, ft), (rx, rt) = forget, retain
    if len(fx) == 0 or len(rx) == 0:
        raise ValueError(f"Need forget and retain tokens, got {len(fx)} and {len(rx)}.")
    w0 = w0.detach().to(device=device, dtype=torch.float32)
    p = w0.shape[1]
    gram = torch.zeros(p, p, device=device, dtype=torch.float32)
    xr = torch.zeros(p, w0.shape[0], device=device, dtype=torch.float32)
    for (x, t), weight in (((fx, ft), forget_weight / len(fx)), ((rx, rt), retain_weight / len(rx))):
        scale = weight ** 0.5
        for s in range(0, len(x), chunk):
            xb = x[s:s + chunk].to(device).float() * scale
            rb = (t[s:s + chunk].to(device).float() - F.linear(x[s:s + chunk].to(device).float(), w0)) * scale
            gram += xb.T @ xb
            xr += xb.T @ rb
    gram = gram.double()
    gram += ridge * gram.diagonal().mean() * torch.eye(p, device=device, dtype=torch.float64)
    delta = torch.linalg.solve(gram, xr.double()).T.float()  # [d, d_ff]
    w = w0 + delta  # still on `device`: the MSE below is evaluated there
    history = [{"epoch": 0, "forget_mse": w_down_mse(w, fx, ft), "retain_mse": w_down_mse(w, rx, rt), "lr": 0.0}]
    logger.info(f"closed form (ridge={ridge:g}): forget_mse={history[0]['forget_mse']:.4g}  retain_mse={history[0]['retain_mse']:.4g}")
    return w.detach().cpu(), history


@torch.no_grad()
def apply_w_down(model: Any, layer_idx: int, weight: torch.Tensor):
    """Write the trained weights into the MLP output projection of `layer_idx` (cast to the model dtype)."""
    proj = get_out_proj(model, layer_idx)
    if proj.bias is not None:
        raise ValueError("W_down training assumes an output projection without bias (as Llama / Qwen2).")
    proj.weight.copy_(weight.to(proj.weight))
