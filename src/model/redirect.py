"""Building blocks of the neighbour-redirection experiment (no hydra, no Trainer).

Part 1 - r_UV ablation over the number of neighbours (used by src/r_uv.py):
    forget question  -> author          (`forget_authors`)
    neighbour rows   -> (author, neighbour)
    per author a fixed random order of its neighbours; the K-neighbour reference of author a is the
    FIRST K of that order, so the sets are nested (K=1 in K=5 in K=15) and the ablation is controlled.
        r_UV[K][a] = mean_{n in first K neighbours of a} mean(act | neighbour n)  -  mean(act | forget of a)

Part 2 - training W_down, LUNAR style (used by src/w_down.py; AdamW like LUNAR's code, or the closed form of Eq. 9). For one decoder layer, with
    x = input of the MLP output projection (down_proj), y = its ORIGINAL output, per token:
        forget tokens : target = y + coeff * shift      (shift built from r_UV of the sample's author)
        retain tokens : target = y                      (the layer must keep behaving the same)
    and `W_down` is trained (MSE) to map x -> target, starting from the original weights.
    Inputs/targets are recorded ONCE from the original model; training itself runs no model forward.

Part 3 - inference-time steering (`Steerer`, used by src/select_layer.py): add coeff * r_UV to the residual
    stream after one layer, i.e. the effect a perfect W_down edit of that layer would have.

Reference: LUNAR (arXiv:2502.07218), https://github.com/facebookresearch/LUNAR
"""

from __future__ import annotations
import logging
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from model.activations import capture_activations, get_out_proj, unlearning_vector
from utils.common import to_device

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------------------
# Part 1: r_UV ablation over the number of neighbours
# ---------------------------------------------------------------------------------------------


def selection_file_name(k: int, coeff: float, shift_mode: str) -> str:
    """Name of the layer-selection file of src/select_layer.py: the selected layer depends on the reference size K,
    the coefficient and the shift mode, so W_down training (src/w_down.py) must read the file of the SAME ones."""
    return f"layer_selection_k{k}_c{coeff:g}_{shift_mode}.json"


def forget_authors(indices: Iterable[Any], qa_per_author: int, author_offset: int = 0) -> List[int]:
    """Author of each forget row. TOFU stores the QAs of one author contiguously (20 per author),
    so `author = row_index // qa_per_author + author_offset`. `author_offset` aligns this numbering
    with the `author_id` column of the neighbour csv."""
    return [int(i) // qa_per_author + author_offset for i in indices]


def neighbors_by_author(author_ids: Iterable[Any], neighbor_ids: Iterable[Any]) -> Dict[Any, List[Any]]:
    """{author: sorted list of distinct neighbour ids} from the two metadata columns of the neighbour rows."""
    out: Dict[Any, Set[Any]] = {}
    for a, n in zip(author_ids, neighbor_ids):
        out.setdefault(a, set()).add(n)
    return {a: sorted(ns) for a, ns in out.items()}


def check_neighbors(forget_author_set: Set[Any], nb_by_author: Mapping[Any, Sequence[Any]], ks: Sequence[int]):
    """Fail early (before any model forward) when the forget authors and the neighbour csv disagree."""
    missing = forget_author_set - set(nb_by_author)
    if missing:
        raise ValueError(
            f"{len(missing)} forget author(s) have no neighbours: {sorted(missing)}.\n"
            f"  forget authors (row_index // qa_per_author + author_offset): {sorted(forget_author_set)}\n"
            f"  `author_id` values in the neighbour data                  : {sorted(nb_by_author)}\n"
            f"  -> fix `ruv.qa_per_author` / `ruv.author_offset` so that both numberings match."
        )
    extra = set(nb_by_author) - forget_author_set
    if extra:
        logger.warning(f"Neighbour data has authors that are not in the forget split (ignored): {sorted(extra)}")
    short = {a: len(nb_by_author[a]) for a in forget_author_set if len(nb_by_author[a]) < max(ks)}
    if short:
        raise ValueError(f"Need at least max(ks)={max(ks)} neighbours per author, but got {short} (author: #neighbours).")


def nested_neighbor_order(nb_by_author: Mapping[Any, Sequence[Any]], seed: int) -> Dict[Any, List[Any]]:
    """One random permutation of the neighbours of every author (deterministic given `seed`,
    independent of the order of the csv rows). The K-neighbour reference is `order[a][:K]`."""
    rng = random.Random(seed)
    order = {}
    for author in sorted(nb_by_author):
        ids = sorted(nb_by_author[author])  # sort first -> the order does not depend on how the ids are listed
        rng.shuffle(ids)
        order[author] = ids
    return order


def compute_r_uv(
    forget_means: Mapping[Any, torch.Tensor],
    neighbor_means: Mapping[Tuple[Any, Any], torch.Tensor],
    order: Mapping[Any, Sequence[Any]],
    ks: Sequence[int],
) -> Dict[int, Dict[Any, torch.Tensor]]:
    """r_UV[K][author] = mean over the first K neighbours (each neighbour weighs the same, whatever
    its number of questions) - mean forget activation of the author. Shapes [n_pos, n_layers, d]."""
    r_uv: Dict[int, Dict[Any, torch.Tensor]] = {}
    for k in ks:
        r_uv[k] = {}
        for author, forget_mean in forget_means.items():
            reference = torch.stack([neighbor_means[(author, n)] for n in order[author][:k]]).mean(dim=0)
            r_uv[k][author] = unlearning_vector(reference, forget_mean)
    return r_uv


def summarize_r_uv(
    r_uv: Mapping[int, Mapping[Any, torch.Tensor]],
    forget_means: Mapping[Any, torch.Tensor],
    layers: Sequence[int],
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Per K and layer (averaged over authors and positions), to read the ablation at a glance:

    - `norm`      : ||r_UV||
    - `rel_norm`  : ||r_UV|| / ||mean forget activation||   (how far the redirection moves the activation)
    - `cos_to_kmax`: cosine(r_UV[K][a], r_UV[max K][a]) over (positions, d): is a K-neighbour reference
                    already pointing where the largest reference points?
    """
    kmax = max(r_uv)
    summary: Dict[int, Dict[str, Dict[str, float]]] = {}
    for k, per_author in r_uv.items():
        authors = list(per_author)
        r = torch.stack([per_author[a] for a in authors])  # [A, n_pos, n_layers, d]
        base = torch.stack([forget_means[a] for a in authors])
        ref = torch.stack([r_uv[kmax][a] for a in authors])
        norm = r.norm(dim=-1).mean(dim=1).mean(dim=0)  # [n_layers]
        rel = (r.norm(dim=-1) / base.norm(dim=-1)).mean(dim=1).mean(dim=0)
        cos = F.cosine_similarity(r.movedim(2, 1).flatten(2), ref.movedim(2, 1).flatten(2), dim=-1).mean(dim=0)  # [n_layers]
        summary[k] = {
            str(layer): {"norm": norm[i].item(), "rel_norm": rel[i].item(), "cos_to_kmax": cos[i].item()}
            for i, layer in enumerate(layers)
        }
    return summary


# ---------------------------------------------------------------------------------------------
# Part 2: W_down training
# ---------------------------------------------------------------------------------------------


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
    was_training = model.training
    model.eval()
    xs: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
    ts: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
    with capture_activations(model, layers, "down_proj_in") as cache_in, \
            capture_activations(model, layers, "mlp_out") as cache_out:
        for batch in tqdm(loader, desc=f"Recording MLP inputs/outputs [{desc}]", unit="batch(es)", colour="blue"):
            inputs = to_device({k: batch[k] for k in ("input_ids", "attention_mask")}, model.device)
            model(**inputs)
            mask = inputs["attention_mask"].bool()
            for l in layers:
                t = cache_out[l].float()
                if shift_fn is not None:
                    t = t + shift_fn(batch, l, inputs["attention_mask"]).to(t)
                xs[l].append(cache_in[l][mask].cpu())
                ts[l].append(t[mask].cpu())
    if was_training:
        model.train()
    return {l: (torch.cat(xs[l]), torch.cat(ts[l])) for l in layers}


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
def apply_w_down(model: Any, layer_idx: int, w: torch.Tensor):
    """Write the trained weights into the MLP output projection of `layer_idx` (cast to the model dtype)."""
    proj = get_out_proj(model, layer_idx)
    if proj.bias is not None:
        raise ValueError("W_down training assumes an output projection without bias (as Llama / Qwen2).")
    proj.weight.copy_(w.to(proj.weight))


# ---------------------------------------------------------------------------------------------
# Part 3: inference-time redirection (steering), used by the LUNAR-style layer selection
# ---------------------------------------------------------------------------------------------


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

    def __init__(self, model: Any, layer: int, positions: Sequence[int], mode: str = "eoi",
                 ref_position: int = -1, coeff: float = 1.0):
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
