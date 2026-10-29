
from __future__ import annotations
import torch
from torch.nn import functional as F
from typing import Tuple, Mapping


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


def build_prompt_answer_batch(
    prompt_batch: Mapping[str, torch.Tensor],
    pad_token_id: int,
    ignore_index: int = -100
):
    """Build a full sequence batch by appending the answer tokens to the prompt batch.

    Parameters
    ----------
    prompt_batch : Mapping[str, torch.Tensor]
        A batch containing `input_ids`, `attention_mask`, and `labels` for the prompt.
    pad_token_id : int
        The token ID used for padding the answer sequences.
    ignore_index : int, optional
        The index to ignore in the labels, by default -100

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary containing `input_ids`, `attention_mask`, `position_ids`, and `labels` for the full sequence.
    """
    prompt_ids, prompt_mask, labels = prompt_batch["input_ids"], prompt_batch["attention_mask"], prompt_batch["labels"]
    if not bool(prompt_mask[:, -1].all()):
        raise ValueError("build_prompt_answer_batch needs LEFT padded prompt-only batches (collator padding_side: left).")
    answers = [label[label != ignore_index] for label in labels]
    bsz, n_prompt, n_ans = prompt_ids.shape[0], prompt_ids.shape[1], max(len(a) for a in answers)

    answer_ids = prompt_ids.new_full((bsz, n_ans), pad_token_id)
    answer_labels = prompt_ids.new_full((bsz, n_ans), ignore_index)
    answer_mask = prompt_mask.new_zeros((bsz, n_ans))
    for i, a in enumerate(answers):
        answer_ids[i, :len(a)] = a
        answer_labels[i, :len(a)] = a
        answer_mask[i, :len(a)] = 1
    full_mask = torch.cat([prompt_mask, answer_mask], dim=1)

    return {
        "input_ids": torch.cat([prompt_ids, answer_ids], dim=1),
        "attention_mask": full_mask,
        "position_ids": (full_mask.long().cumsum(-1) - 1).clamp(min=0),
        "labels": torch.cat([prompt_ids.new_full((bsz, n_prompt), ignore_index), answer_labels], dim=1),
    }
