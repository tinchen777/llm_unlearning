"""Activation statistics of the NEAR experiment: means per GROUP (author), separability of two question sets.

The plain hooks (`capture_activations`, `collect_mean_activations`, `collect_sample_activations`, ...) are generic and live in `model.activations`.
"""

from __future__ import annotations
import logging
import torch
from typing import Dict, Hashable, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


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
    """
    How linearly separable are the activations `a` [n, d] and `b` [m, d]?
    Cross-validated AUC of the projection on the difference of class means, the direction being fitted on the OTHER folds (0.5 = indistinguishable, 1.0 = perfectly separated along a single direction, like r_UV).

    `groups_a` / `groups_b` (one key per row): split the folds by group instead of by sample (see `_group_folds`).
    Without them questions of one author land on both sides of the split and the author's name alone separates the classes; with them only what generalises across authors counts. Both must be given, or neither. With fewer than `min_groups` groups in a class (e.g. forget01 has 2 authors) the direction fitted on the other group(s) cannot generalise and the AUC is ~0 or ~1 by chance (seen on a toy: 0.10 vs 0.80 for two near-identical models), so a ValueError is raised instead of returning a number.
    """
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
