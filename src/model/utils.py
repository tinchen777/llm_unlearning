"""
Utility functions for `model` module.
"""
from __future__ import annotations
import torch
from transformers import BatchEncoding
from typing import Any, Dict, Mapping, Literal, overload, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers.modeling_outputs import CausalLMOutputWithPast


def batch_to_device(batch: Any, device: torch.device):
    """Move a batch of data to the specified device."""
    if isinstance(batch, dict):
        return {k: v.to(device) for k, v in batch.items()}
    elif isinstance(batch, BatchEncoding):
        return batch.to(device)
    else:
        raise ValueError(
            f"Expected batch to be a `dict` or `BatchEncoding`, but got {type(batch)}."
        )


@overload
def forward_batch(model: Any, batch: Mapping[str, torch.Tensor], ignore_labels: Literal[False] = ..., grad: bool = ...) -> Tuple[torch.Tensor, torch.Tensor, CausalLMOutputWithPast]: ...
@overload
def forward_batch(model: Any, batch: Mapping[str, torch.Tensor], ignore_labels: Literal[True], grad: bool = ...) -> Tuple[torch.Tensor, CausalLMOutputWithPast]: ...
def forward_batch(model: Any, batch: Mapping[str, torch.Tensor], ignore_labels: bool = False, grad: bool = True):
    """
    Forward a batch through the model.

    Parameters
    ----------
        model : Any
            The model to forward the batch through.

        batch : Mapping[str, torch.Tensor]
            The batch of data to forward through the model.

        ignore_labels : bool, default `False`
            - `True`, the 'labels' key in the batch will be ignored, which means the model will not compute the loss.
            - `False`, the 'labels' key in the batch will be used to compute the loss.

        grad : bool, default `True`
            - `True`, gradients will be computed.

    Returns
    -------
        For `ignore_labels`：
            - `True`: Returns a tuple of (`logits`, `outputs`) of the model.
            - `False`: Returns a tuple of (`loss`, `logits`, `outputs`)
        `logits` : torch.Tensor
            Shape as `[batch_size, seq_len, vocab_size]`.
        `loss` : torch.Tensor
            `Scalar` tensor representing the loss.
        `outputs` : CausalLMOutputWithPast
            The output of the model, which includes `logits`, `loss`, etc.
    """
    ignore_keys = ("index", "labels") if ignore_labels else ("index",)
    batch = {k: v for k, v in batch.items() if k not in ignore_keys}

    with torch.set_grad_enabled(grad):
        outputs = model(**batch)
    # logits
    logits = outputs.logits  # bsz x seq_len x vocab_size
    if logits is None:
        raise ValueError("Model output `logits` is `None`. Ensure the model returns logits.")
    if ignore_labels:
        return logits, outputs
    else:
        # loss
        loss = outputs.loss  # scalar
        if loss is None:
            raise ValueError("Model output `loss` is `None`. Ensure the batch contains `labels` and the model returns loss.")
        return loss, logits, outputs


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
