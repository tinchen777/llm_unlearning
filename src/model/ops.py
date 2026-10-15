
from __future__ import annotations
import torch
from torch.nn import functional as F
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    pass


def per_token_CE(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
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


def mean_answer_logprob(
    logits: torch.Tensor,
    answer_ids: torch.Tensor,
    answer_mask: torch.Tensor
) -> torch.Tensor:
    """
    Mean per-token log-prob of the answers, i.e. the log of the length-normalised answer probability.

    Parameters
    ----------
        logits : torch.Tensor
            Logits `[batch_size, n_context + n_answer, vocab_size]` of the sequences `[context | answer]`
            (the answers are the LAST `n_answer` columns, right padded).

        answer_ids : torch.Tensor
            `[batch_size, n_answer]` answer tokens.

        answer_mask : torch.Tensor
            `[batch_size, n_answer]`, 1 at the real answer tokens (0 at the padding).

    Returns
    -------
    torch.Tensor
        `[batch_size]`, <= 0.
    """
    n_answer = answer_ids.shape[1]
    # the token at column n_context + j is predicted by column n_context + j - 1
    logp = logits[:, -(n_answer + 1):-1].float().log_softmax(-1)
    answer_ids, mask = answer_ids.to(logits.device), answer_mask.to(logits.device).float()
    token_lp = logp.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1) * mask
    return token_lp.sum(-1) / mask.sum(-1).clamp(min=1)
