"""Loss terms of the neighbour-redirection objective."""

from __future__ import annotations
from typing import Optional

import torch
from torch.nn import functional as F


def masked_redirect_loss(
    activations: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "mse",
) -> torch.Tensor:
    """Pull the selected positions of `activations` onto `target`.

    Args:
        activations: `[b, s, d]` activations of the forget batch.
        target: `[b, d]` neighbour target, broadcast over the selected positions.
        mask: `[b, s]` boolean, the positions the redirection acts on.
        loss_type: `mse` (LUNAR-style regression) or `cosine` (direction only).

    Returns:
        Scalar loss, averaged per sample so that samples with many selected
        positions do not dominate.
    """
    target = target.to(dtype=activations.dtype, device=activations.device)
    expanded = target.unsqueeze(1).expand_as(activations)
    weights = mask.to(activations.dtype)
    counts = weights.sum(dim=1).clamp(min=1.0)

    if loss_type == "mse":
        per_position = F.mse_loss(activations, expanded, reduction="none").mean(dim=-1)
    elif loss_type == "cosine":
        per_position = 1.0 - F.cosine_similarity(activations, expanded, dim=-1)
    else:
        raise ValueError(f"Unknown redirect loss type `{loss_type}` (expected mse|cosine).")

    per_sample = (per_position * weights).sum(dim=1) / counts
    return per_sample.mean()


def masked_activation_mse(
    activations: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Keep `activations` where the reference model put them (retain term)."""
    reference = reference.to(dtype=activations.dtype, device=activations.device)
    per_position = F.mse_loss(activations, reference, reduction="none").mean(dim=-1)
    weights = mask.to(activations.dtype)
    counts = weights.sum(dim=1).clamp(min=1.0)
    return ((per_position * weights).sum(dim=1) / counts).mean()


def info_nce(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.1,
    negatives_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """InfoNCE with the neighbour as the positive.

    Args:
        anchor: `[b, d]` pooled forget activations of the model being trained.
        positive: `[b, d]` neighbour target.
        negatives: `[b, k, d]` activations the forget sample must move away from
            (its own pre-unlearning activation, other entities' targets, ...).
        temperature: softmax temperature.
        negatives_mask: `[b, k]` boolean; `False` entries are ignored.

    Returns:
        Scalar loss. Minimising it makes the forget activation closer to its
        neighbour than to any negative.
    """
    anchor = F.normalize(anchor.float(), dim=-1)
    positive = F.normalize(positive.float().to(anchor.device), dim=-1)
    negatives = F.normalize(negatives.float().to(anchor.device), dim=-1)

    positive_logit = (anchor * positive).sum(dim=-1, keepdim=True) / temperature  # [b, 1]
    negative_logits = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1) / temperature  # [b, k]
    if negatives_mask is not None:
        negative_logits = negative_logits.masked_fill(~negatives_mask.to(anchor.device), float("-inf"))

    logits = torch.cat([positive_logit, negative_logits], dim=-1)
    labels = torch.zeros(len(anchor), dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, labels)
