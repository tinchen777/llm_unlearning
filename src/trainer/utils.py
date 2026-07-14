
from __future__ import annotations
from torch.nn import functional as F
from typing import Any, Mapping, TYPE_CHECKING

from utils.common import forward_batch, per_token_CE

if TYPE_CHECKING:
    import torch


def compute_kl_divergence(
    model: Any,
    target_model: Any,
    inputs: Mapping[str, torch.Tensor]
):
    ref_logits, _ = forward_batch(target_model, inputs, ignore_labels=True, grad=False)

    ref_probs = ref_logits.log_softmax(dim=-1).reshape(-1, ref_logits.shape[-1])

    logits, outputs = forward_batch(model, inputs, ignore_labels=True)
    current_probs = logits.log_softmax(dim=-1).reshape(-1, logits.shape[-1])

    # minimum KL divergence
    return F.kl_div(
        current_probs, ref_probs, reduction="batchmean", log_target=True
    ), outputs


def compute_batch_nll(model: Any, inputs: Mapping[str, torch.Tensor], grad: bool = True):
    # get the sum loss for each sequence in a batch
    # NOTE: not same as `model(**inputs).loss` but has sum loss for each seq in a batch
    logits, outputs = forward_batch(model, inputs, ignore_labels=True, grad=grad)

    losses, shift_labels_mask = per_token_CE(logits, inputs["labels"])
    losses = losses.sum(dim=-1)  # sum loss for each sequence in a batch

    return losses, outputs, shift_labels_mask
