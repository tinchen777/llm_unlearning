"""Diagnostics computed on pooled activations, used to score candidate layers."""

from __future__ import annotations
from typing import Dict, Optional, Sequence

import torch


def _as_float(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(torch.float32)


def probe_auc(
    positives: torch.Tensor,
    negatives: torch.Tensor,
    n_splits: int = 5,
    seed: int = 42,
    max_iter: int = 2000,
) -> float:
    """Cross-validated AUC of a linear probe separating two activation sets.

    This is the layer-selection analogue of the paper's identifiability attack:
    a high AUC means the layer linearly encodes the distinction, a value near
    0.5 means the two sets are indistinguishable there.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = torch.cat([_as_float(positives), _as_float(negatives)], dim=0).numpy()
    y = torch.cat([
        torch.ones(len(positives)), torch.zeros(len(negatives))
    ]).numpy()
    if len(positives) < 2 or len(negatives) < 2:
        return float("nan")

    splits = max(2, min(n_splits, len(positives), len(negatives)))
    folds = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    scores = []
    for train_idx, test_idx in folds.split(X, y):
        clf = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=max_iter, C=1.0)
        )
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict_proba(X[test_idx])[:, 1]
        if len(set(y[test_idx].tolist())) < 2:
            continue
        scores.append(roc_auc_score(y[test_idx], preds))
    return float(sum(scores) / len(scores)) if scores else float("nan")


def relative_distance(source: torch.Tensor, target: torch.Tensor) -> float:
    """Mean `||h_src - h_tgt|| / ||h_src||` over paired activations.

    How far the redirection has to move each forget activation, in units of the
    activation's own norm. Large values mean a big, destructive edit.
    """
    source, target = _as_float(source), _as_float(target)
    diff = (source - target).norm(dim=-1)
    norm = source.norm(dim=-1).clamp(min=1e-6)
    return float((diff / norm).mean())


def mean_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = _as_float(a), _as_float(b)
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())


def centroid_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((_as_float(a).mean(0) - _as_float(b).mean(0)).norm())


def spread(x: torch.Tensor) -> float:
    """Mean distance to the centroid."""
    x = _as_float(x)
    return float((x - x.mean(0, keepdim=True)).norm(dim=-1).mean())


def target_cohesion(grouped: Sequence[torch.Tensor]) -> float:
    """How well-defined a per-entity mean target is, in `[0, 1]`.

    `1 - within/(within + between)`: high when the M neighbours of an entity sit
    close together relative to how far apart different entities' neighbours are.
    A low value means the "mean neighbour" is an average face rather than a
    location any real stranger occupies.
    """
    groups = [_as_float(g) for g in grouped if len(g) > 0]
    if len(groups) < 2:
        return float("nan")
    within = sum(spread(g) for g in groups) / len(groups)
    centroids = torch.stack([g.mean(0) for g in groups])
    between = spread(centroids)
    total = within + between
    return float(1.0 - within / total) if total > 0 else float("nan")


def layer_diagnostics(
    forget: torch.Tensor,
    neighbor: torch.Tensor,
    retain: Optional[torch.Tensor] = None,
    neighbor_groups: Optional[Sequence[torch.Tensor]] = None,
    seed: int = 42,
) -> Dict[str, float]:
    """All raw diagnostics for one layer.

    Args:
        forget: `[N, d]` activations of the forget questions.
        neighbor: `[N, d]` activations of the paired neighbour questions
            (same order as `forget`), used for the paired distance.
        retain: `[R, d]` activations of retain-entity questions.
        neighbor_groups: per-entity neighbour activations, for cohesion.
    """
    known = forget if retain is None else torch.cat([_as_float(forget), _as_float(retain)], dim=0)
    diagnostics = {
        "known_unknown_auc": probe_auc(known, neighbor, seed=seed),
        "forget_neighbor_auc": probe_auc(forget, neighbor, seed=seed),
        "redirect_cost": relative_distance(forget, neighbor),
        "forget_neighbor_cosine": mean_cosine(forget, neighbor),
        "neighbor_cohesion": (
            target_cohesion(neighbor_groups) if neighbor_groups else float("nan")
        ),
        "forget_spread": spread(forget),
        "neighbor_spread": spread(neighbor),
    }
    if retain is not None:
        gap = centroid_distance(forget, retain)
        diagnostics["forget_retain_centroid_gap"] = gap
        diagnostics["retain_separation"] = gap / max(spread(retain), 1e-6)
        diagnostics["retain_neighbor_auc"] = probe_auc(retain, neighbor, seed=seed)
    return diagnostics
