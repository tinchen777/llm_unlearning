
from __future__ import annotations
import torch
from torch.nn import functional as F
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    pass


def per_token_CE(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = 100
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-token cross-entropy loss.

    Parameters
    ----------
        logits : torch.Tensor
            The logits from the model of shape `[batch_size, seq_len, vocab_size]`.

        labels : torch.Tensor
            The ground truth labels of shape `[batch_size, seq_len]`.

    Returns
    -------
    losses : torch.Tensor
        The per-token cross-entropy loss of shape `[batch_size, seq_len-1]`.
    shift_labels_mask : torch.BoolTensor
        A mask indicating valid positions (not equal to `ignore_index`) of shape `[batch_size, seq_len-1]`.
    """
    shift_logits = logits[..., :-1, :]  # shape as [batch_size, seq_len-1, vocab_size]
    shift_labels = labels[..., 1:].to(logits.device)  # shape as [batch_size, seq_len-1]
    losses = F.cross_entropy(
        shift_logits.transpose(-1, -2),
        shift_labels,
        ignore_index=ignore_index,
        reduction="none",
    )  # shape as [batch_size, seq_len-1]

    return losses, shift_labels != ignore_index
