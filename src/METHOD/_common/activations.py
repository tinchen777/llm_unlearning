"""Unlearning vector shared by the activation-redirection methods (LUNAR, NEAR)."""

from __future__ import annotations
import torch


def unlearning_vector(reference_mean: torch.Tensor, forget_mean: torch.Tensor) -> torch.Tensor:
    """LUNAR unlearning vector: r_UV = mean(a | D_ref) - mean(a | D_forget), per position/layer."""
    if reference_mean.shape != forget_mean.shape:
        raise ValueError(f"Shape mismatch: {tuple(reference_mean.shape)} vs {tuple(forget_mean.shape)}.")
    return reference_mean - forget_mean
