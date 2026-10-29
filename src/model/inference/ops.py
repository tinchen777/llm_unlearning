
from __future__ import annotations
import torch
from torch.nn import functional as F
from typing import Tuple, Dict, Any, List, Mapping

from .utils import batch_to_device


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

# TODO switch to metric
@torch.no_grad()
def answer_logprobs(model: Any, batch: Mapping[str, Any], pad_token_id: int, ignore_index: int = -100) -> List[float]:
    """Mean per-token log-prob of the ground-truth answer of every row, given its prompt (teacher forcing).

    `batch` is a LEFT padded prompt-only batch (`predict_with_generate`); the answer tokens are read from
    `labels` and appended after the prompt columns (`append_answers`), so the prompt keeps the same columns as in
    generation (a `Steerer` hooked on the model therefore redirects exactly the same tokens).
    """
    inputs, answer_ids, answer_mask = append_answers(batch, pad_token_id, ignore_index)
    logits = model(**batch_to_device(inputs, model.device)).logits
    return mean_answer_logprob(logits, answer_ids, answer_mask).tolist()


def append_answers(
    batch: Mapping[str, Any],
    pad_token_id: int,
    ignore_index: int = -100
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Turn a LEFT padded prompt-only batch (generation layout, `predict_with_generate`) into full sequences by appending the
    answer tokens (the non-`ignore_index` part of `labels`) after the prompt columns.

    The prompt keeps exactly its columns of the generation pass, so a hook set up for generation (e.g. a `Steerer`)
    acts on the same tokens; the answers are RIGHT padded after it.

    Returns
    -------
    inputs : dict
        `input_ids`, `attention_mask`, `position_ids` of shape `[batch_size, n_prompt + n_answer]`. The positions are
        counted from the first real token (left padding shifts the columns, plain `model(...)` would use the column).
    answer_ids : torch.Tensor
        `[batch_size, n_answer]`, right padded with `pad_token_id`.
    answer_mask : torch.Tensor
        `[batch_size, n_answer]`, 1 at the real answer tokens.
    """
    ids, am, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
    if not bool(am[:, -1].all()):
        raise ValueError("append_answers needs LEFT padded prompt-only batches (collator padding_side: left).")
    answers = [label[label != ignore_index] for label in labels]
    bsz, n_ans = ids.shape[0], max(len(a) for a in answers)
    answer_ids = torch.full((bsz, n_ans), pad_token_id, dtype=ids.dtype)
    answer_mask = torch.zeros((bsz, n_ans), dtype=am.dtype)
    for i, a in enumerate(answers):
        answer_ids[i, :len(a)] = a
        answer_mask[i, :len(a)] = 1
    full_mask = torch.cat([am, answer_mask], dim=1)
    inputs = {
        "input_ids": torch.cat([ids, answer_ids], dim=1),
        "attention_mask": full_mask,
        "position_ids": (full_mask.long().cumsum(-1) - 1).clamp(min=0),
    }
    return inputs, answer_ids, answer_mask
