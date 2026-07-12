
from __future__ import annotations
import torch
from torch.nn import functional as F
from typing import Any, Mapping, Optional, Tuple

from utils.common import forward_batch, per_token_CE


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

    loss, _ = per_token_CE(logits, inputs["labels"])
    loss = loss.sum(dim=-1)  # sum loss for each sequence in a batch

    return loss, outputs


def compute_dpo_loss(
    model: Any,
    ref_model: Any,
    beta: float,
    win_inputs: Optional[Mapping[str, torch.Tensor]] = None,
    lose_inputs: Optional[Mapping[str, torch.Tensor]] = None
):
    if win_inputs is not None:
        win_loss, win_outputs = compute_batch_nll(model, win_inputs)
        win_ref_loss, _ = compute_batch_nll(ref_model, win_inputs, grad=False)
        win_log_ratio = -(win_loss - win_ref_loss)
    else:
        win_log_ratio = None
        win_outputs = None

    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
        lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs, grad=False)
        lose_log_ratio = -(lose_loss - lose_ref_loss)
    else:
        lose_log_ratio = None
        lose_outputs = None
    # diff_log_ratio = win_log_ratio - lose_log_ratio
    if win_log_ratio is not None:
        if lose_log_ratio is not None:
            diff_log_ratio = win_log_ratio - lose_log_ratio
        else:
            diff_log_ratio = win_log_ratio
    else:
        if lose_log_ratio is not None:
            diff_log_ratio = -lose_log_ratio
        else:
            raise ValueError("Both win_log_ratio and lose_log_ratio can't be None")

    loss = -2 / beta * F.logsigmoid(beta * diff_log_ratio).mean()
    return loss, (win_outputs, lose_outputs)


def compute_undial_loss(
    model: Any,
    ref_model: Any,
    inputs: Mapping[str, torch.Tensor],
    beta: float
):
    # Forward pass on the student (trainable) model
    logits, outputs = forward_batch(model, inputs, ignore_labels=True)
    
    
    # outputs = model(**inputs)
    # logits = outputs.logits
    labels = inputs["labels"]

    shift_labels = labels[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()

    # Forward pass on the teacher model (no grad)
    teacher_logits, _ = forward_batch(ref_model, inputs, ignore_labels=True, grad=False)
    
    # with torch.no_grad():
    #     teacher_logits = ref_model(**inputs).logits
    shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()

    # Build the mask that identifies the tokens need to be unlearned
    mask = torch.zeros_like(shift_teacher_logits)
    batch_idx = torch.arange(mask.shape[0]).view(-1, 1, 1)
    seq_idx = torch.arange(mask.shape[1]).view(1, -1, 1)
    mask[batch_idx, seq_idx, shift_labels.unsqueeze(-1)] = 1.0

    # Adjust teacher logits: subtract di_strength on the correct token
    # pre_softmax = shift_teacher_logits - mask * beta
    # soft_label = F.softmax(pre_softmax, dim=-1)
    soft_label = (shift_teacher_logits - mask * beta).softmax(dim=-1)

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        soft_label.view(-1, soft_label.size(-1)),
    )
    return loss.mean(), outputs


def compute_wga_loss(
    model: Any,
    inputs: Mapping[str, torch.Tensor],
    beta: float
):
    logits, outputs = forward_batch(model, inputs, ignore_labels=True)

    loss, shift_mask = per_token_CE(logits, inputs["labels"])
    loss = loss.view(-1)

    # outputs = model(**inputs)
    # labels = inputs["labels"]
    # labels = labels.to(outputs.logits.device)

    # shift_logits = outputs.logits[..., :-1, :].contiguous()
    # shift_labels = labels[..., 1:].contiguous()

    # lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
    #     shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    # )
    weight_ce = ((-loss).exp().detach()) ** beta
    forget_loss = -(weight_ce * loss)[shift_mask].mean()
    return forget_loss, outputs


def compute_satimp_loss(
    model: Any,
    inputs: Mapping[str, torch.Tensor],
    beta1: float,
    beta2: float
):
    logits, outputs = forward_batch(model, inputs, ignore_labels=True)
    
    loss, shift_mask = per_token_CE(logits, inputs["labels"])
    loss = loss.view(-1)
    
    
    # outputs = model(**inputs)
    # labels = inputs["labels"]
    # labels = labels.to(outputs.logits.device)

    # shift_logits = outputs.logits[..., :-1, :].contiguous()
    # shift_labels = labels[..., 1:].contiguous()

    # lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
    #     shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    # )
    weight_sat = ((-loss).exp().detach()) ** beta1
    weight_imp = (1 - (-loss).exp().detach()) ** beta2
    forget_loss = -((weight_sat * weight_imp) * loss)[shift_mask].mean()
    return forget_loss, outputs
