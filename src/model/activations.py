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
LUNAR uses the post-instruction template tokens, see `get_eoi_positions`.

Reference: LUNAR (arXiv:2502.07218), https://github.com/facebookresearch/LUNAR
"""

from __future__ import annotations
import logging
import torch
from torch import nn
from contextlib import contextmanager
from tqdm import tqdm
from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

from utils.common import to_device

logger = logging.getLogger(__name__)

HOOK_POINTS = ("block_out", "mlp_out", "down_proj_in", "resid_mid")
_MLP_OUT_PROJ_NAMES = ("down_proj", "fc2", "c_proj", "dense_4h_to_h")

_EOI_SENTINEL = "Where was the author of this book born?"


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
def collect_activations(
    model: Any,
    dataloader: Iterable[Mapping[str, Any]],
    positions: Sequence[int] = (-1,),
    layers: Optional[Sequence[int]] = None,
    point: str = "block_out",
    reduce: str = "mean",
    desc: str = "",
    meta_keys: Optional[Sequence[str]] = None,
):
    """Collect `point` activations at `positions` for every layer in `layers`.

    Args:
        reduce: - `"mean"`: dataset-mean activations, Tensor[n_pos, n_layers, d] (float64, CPU),
                  accumulated on the fly (no per-sample storage), as in LUNAR.
                - `"none"`: per-sample activations, `{dataset_index: Tensor[n_pos, n_layers, d]}`
                  (model dtype, CPU), e.g. to build per-sample redirection targets.
        meta_keys: only with `reduce="none"`: batch metadata (e.g. `author_id`) to record per sample as well.
    Returns:
        (activations, layers); with `meta_keys` additionally `{dataset_index: {key: value}}` as a third element.
    """
    if reduce not in ("mean", "none"):
        raise ValueError(f"reduce must be `mean` or `none`, got `{reduce}`.")
    layers = list(range(model.config.num_hidden_layers)) if layers is None else list(layers)
    was_training = model.training
    model.eval()

    total: Optional[torch.Tensor] = None
    n_samples = 0
    per_sample: Dict[int, torch.Tensor] = {}
    per_meta: Dict[int, Dict[str, Any]] = {}
    with capture_activations(model, layers, point) as cache:
        for batch in tqdm(dataloader, desc=f"Collecting `{point}` activations [{desc}]", unit="batch(es)", colour="blue"):
            if "input_ids" not in batch:
                raise ValueError(f"Expected a flat batch with `input_ids`, got keys {list(batch)}.")
            indices = batch.get("index")
            inputs = to_device({k: batch[k] for k in ("input_ids", "attention_mask")}, model.device)
            model(**inputs)
            # [bsz, n_pos, n_layers, d]
            acts = torch.stack(
                [gather_positions(cache[l], inputs["attention_mask"], positions).cpu() for l in layers],
                dim=2
            )
            if reduce == "mean":
                batch_sum = acts.to(torch.float64).sum(dim=0)
                total = batch_sum if total is None else total + batch_sum
            else:
                if indices is None:
                    raise ValueError("reduce='none' needs the dataset `index` in each batch.")
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
    if reduce == "mean":
        return total / n_samples, layers  # type: ignore
    if meta_keys:
        return per_sample, layers, per_meta
    return per_sample, layers


@torch.no_grad()
def collect_group_means(
    model: Any,
    dataloader: Iterable[Mapping[str, Any]],
    group_fn: Callable[[Mapping[str, Any]], Sequence[Hashable]],
    positions: Sequence[int] = (-1,),
    layers: Optional[Sequence[int]] = None,
    point: str = "block_out",
    desc: str = "",
):
    """Like `collect_activations(reduce="mean")`, but one mean per GROUP (e.g. per author).

    `group_fn(batch)` returns one hashable group key per row of the batch. Only running sums are
    kept (no per-sample storage), so memory is `n_groups * n_pos * n_layers * d` float32.

    Returns:
        (means, counts, layers): `means[key]` is Tensor[n_pos, n_layers, d] (float32, CPU),
        `counts[key]` the number of samples in the group.
    """
    layers = list(range(model.config.num_hidden_layers)) if layers is None else list(layers)
    was_training = model.training
    model.eval()

    sums: Dict[Hashable, torch.Tensor] = {}
    counts: Dict[Hashable, int] = {}
    with capture_activations(model, layers, point) as cache:
        for batch in tqdm(dataloader, desc=f"Collecting `{point}` group means [{desc}]", unit="batch(es)", colour="blue"):
            keys = list(group_fn(batch))
            inputs = to_device({k: batch[k] for k in ("input_ids", "attention_mask")}, model.device)
            if len(keys) != inputs["input_ids"].shape[0]:
                raise ValueError(f"group_fn returned {len(keys)} keys for a batch of {inputs['input_ids'].shape[0]} rows.")
            model(**inputs)
            # [bsz, n_pos, n_layers, d]
            acts = torch.stack(
                [gather_positions(cache[l], inputs["attention_mask"], positions).cpu() for l in layers],
                dim=2
            ).float()
            for i, key in enumerate(keys):
                sums[key] = sums[key] + acts[i] if key in sums else acts[i].clone()
                counts[key] = counts.get(key, 0) + 1

    if was_training:
        model.train()
    if not sums:
        raise ValueError("The dataloader yielded no samples.")
    return {key: sums[key] / counts[key] for key in sums}, counts, layers


def unlearning_vector(reference_mean: torch.Tensor, forget_mean: torch.Tensor) -> torch.Tensor:
    """LUNAR unlearning vector: r_UV = mean(a | D_ref) - mean(a | D_forget), per position/layer."""
    if reference_mean.shape != forget_mean.shape:
        raise ValueError(f"Shape mismatch: {tuple(reference_mean.shape)} vs {tuple(forget_mean.shape)}.")
    return reference_mean - forget_mean


def get_eoi_positions(tokenizer: Any, template_args: Any, chat_header: Any = None) -> List[int]:
    """Negative offsets of the end-of-instruction (post-question template) tokens.

    E.g. Llama-3: `<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n` -> [-5..-1].
    Found as the prompt tokens whose decoded text lies entirely after the question, so it
    follows the same templating (system prompt, date string, tags) as the dataset.
    """
    from data.datasets.utils import prepare_chat_header, tok_chat_sample

    if chat_header is None:
        chat_header = prepare_chat_header(template_args)
    prompt_ids = tok_chat_sample(
        _EOI_SENTINEL, "", 0,
        tokenizer=tokenizer,
        chat_header=chat_header,
        template_args=template_args,
        max_length=4096,
        predict_with_generate=True,
    )["input_ids"][0]
    # smallest prefix that still contains the whole question
    prefix_len = len(prompt_ids)
    while prefix_len > 0 and _EOI_SENTINEL in tokenizer.decode(prompt_ids[:prefix_len - 1]):
        prefix_len -= 1
    n_eoi = len(prompt_ids) - prefix_len
    if n_eoi == 0:
        raise ValueError("The prompt template has no tokens after the question; use explicit positions.")
    return list(range(-n_eoi, 0))


def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """ROC-AUC = P(score(pos) > score(neg)) (ties count 1/2), via the Mann-Whitney U statistic."""
    scores = torch.cat([pos, neg]).double()
    order = scores.argsort()
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=scores.dtype)
    for value in scores.unique():  # average ranks of ties
        tie = scores == value
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    n_pos, n_neg = len(pos), len(neg)
    return ((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()


def _group_folds(
    groups_a: Sequence[Hashable],
    groups_b: Sequence[Hashable],
    folds: int,
    g: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fold of every sample of both classes, all samples of a group in ONE fold, so a test sample never has a sibling
    (same author / same neighbour) in the fold the direction is fitted on. A group present in BOTH classes (e.g. a
    forget author and the neighbours generated for him) gets the same fold in both; the other groups are spread so
    that every fold holds samples of every class."""
    keys_a, keys_b = set(groups_a), set(groups_b)
    shared = sorted(keys_a & keys_b, key=str)
    fold_of: Dict[Hashable, int] = {k: r % folds for r, k in enumerate(
        [shared[i] for i in torch.randperm(len(shared), generator=g).tolist()])}
    out = []
    for groups, keys in ((groups_a, keys_a), (groups_b, keys_b)):
        local = dict(fold_of)  # class-only groups are balanced per class
        counts = [0] * folds
        for k in groups:
            if k in fold_of:
                counts[fold_of[k]] += 1
        own = sorted(keys - set(fold_of), key=str)
        size = {k: sum(1 for x in groups if x == k) for k in own}
        for i in torch.randperm(len(own), generator=g).tolist():
            k = own[i]
            f = min(range(folds), key=lambda j: counts[j])
            local[k] = f
            counts[f] += size[k]
        ids = torch.tensor([local[k] for k in groups])
        if any(not bool((ids == f).any()) for f in range(folds)):
            raise ValueError(f"Cannot split {len(keys)} group(s) over {folds} folds with every fold non-empty.")
        out.append(ids)
    return out[0], out[1]


def mean_diff_auc(
    a: torch.Tensor,
    b: torch.Tensor,
    folds: int = 2,
    seed: int = 0,
    groups_a: Optional[Sequence[Hashable]] = None,
    groups_b: Optional[Sequence[Hashable]] = None,
    min_groups: int = 4,
) -> float:
    """How linearly separable are the activations `a` [n, d] and `b` [m, d]? Cross-validated AUC of the projection
    on the difference of class means, the direction being fitted on the OTHER folds (0.5 = indistinguishable,
    1.0 = perfectly separated along a single direction, like r_UV).

    `groups_a` / `groups_b` (one key per row): split the folds by group instead of by sample (see `_group_folds`).
    Without them questions of one author land on both sides of the split and the author's name alone separates the
    classes; with them only what generalises across authors counts. Both must be given, or neither. With fewer than
    `min_groups` groups in a class (e.g. forget01 has 2 authors) the direction fitted on the other group(s) cannot
    generalise and the AUC is ~0 or ~1 by chance (seen on a toy: 0.10 vs 0.80 for two near-identical models), so a
    ValueError is raised instead of returning a number."""
    if (groups_a is None) != (groups_b is None):
        raise ValueError("Give groups for both classes or for neither.")
    if groups_a is not None:
        n_groups = min(len(set(groups_a)), len(set(groups_b)))  # type: ignore[arg-type]
        if n_groups < min_groups:
            raise ValueError(f"group-wise AUC needs at least {min_groups} groups per class, got {n_groups}.")
    if len(a) < folds or len(b) < folds:
        raise ValueError(f"Need at least {folds} samples per class, got {len(a)} and {len(b)}.")
    g = torch.Generator().manual_seed(seed)
    if groups_a is None:
        fa = torch.randperm(len(a), generator=g) % folds
        fb = torch.randperm(len(b), generator=g) % folds
    else:
        fa, fb = _group_folds(list(groups_a), list(groups_b), folds, g)  # type: ignore[arg-type]
    a, b = a.double(), b.double()
    scores = []
    for f in range(folds):
        direction = a[fa != f].mean(0) - b[fb != f].mean(0)
        scores.append(auc(a[fa == f] @ direction, b[fb == f] @ direction))
    return sum(scores) / folds
