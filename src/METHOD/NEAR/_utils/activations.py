"""Activation statistics of the NEAR experiment: means per GROUP (author), separability of two question sets.

`collect_group_means` is used by r_uv.py / w_down.py, `mean_diff_auc` (+ `auc`, `_group_folds`) by probe.py.
The plain hooks (`capture_activations`, `collect_mean_activations`, `collect_sample_activations`, ...) are generic and live in `model.activations`.
"""

from __future__ import annotations
import logging
import torch
from tqdm import tqdm
from typing import Any, Callable, Dict, Hashable, Iterable, Mapping, Optional, Sequence, Tuple

from model.activations import capture_activations, gather_positions
from model.utils import batch_to_device

logger = logging.getLogger(__name__)


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
    """Like `collect_mean_activations`, but one mean per GROUP (e.g. per author).

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
            inputs = batch_to_device({k: batch[k] for k in ("input_ids", "attention_mask")}, model.device)
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
