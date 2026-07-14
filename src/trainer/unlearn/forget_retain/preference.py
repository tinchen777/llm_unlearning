
from __future__ import annotations
from torch.nn import functional as F

from ...utils import compute_batch_nll
from .base import ForgetRetainTrainer


class NPO(ForgetRetainTrainer):
    requires_ref_model = True

    def __init__(self, beta: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def compute_forget_loss(self, model, forget_inputs, **kwargs):
        lose_loss, lose_outputs, _ = compute_batch_nll(model, forget_inputs)
        lose_ref_loss, _, _ = compute_batch_nll(self.ref_model, forget_inputs, grad=False)
        delta = lose_loss - lose_ref_loss

        loss = -2 / self.beta * F.logsigmoid(self.beta * delta).mean()
        return loss, lose_outputs


class DPO(ForgetRetainTrainer):
    requires_ref_model = True

    def __init__(self, beta: float = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def compute_forget_loss(self, model, forget_inputs, **kwargs):
        # win
        win_inputs = forget_inputs["alternate"]
        win_loss, _, _ = compute_batch_nll(model, win_inputs)
        win_ref_loss, _, _ = compute_batch_nll(self.ref_model, win_inputs, grad=False)
        win_log_ratio = win_ref_loss - win_loss  # = log π_θ(y_w) - log π_ref(y_w)
        # lose
        lose_inputs = forget_inputs["original"]
        lose_loss, lose_outputs, _ = compute_batch_nll(model, lose_inputs)
        lose_ref_loss, _, _ = compute_batch_nll(self.ref_model, lose_inputs, grad=False)
        lose_log_ratio = lose_ref_loss - lose_loss  # = log π_θ(y_l) - log π_ref(y_l)

        diff_log_ratio = win_log_ratio - lose_log_ratio

        loss = -F.logsigmoid(self.beta * diff_log_ratio).mean()
        return loss, lose_outputs


class SimNPO(ForgetRetainTrainer):
    def __init__(self, delta: float = 0.0, beta: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.beta = beta

    def compute_forget_loss(self, model, forget_inputs, **kwargs):
        losses, outputs, shift_labels_mask = compute_batch_nll(model, forget_inputs)
        diff = losses / shift_labels_mask.sum(-1).clamp(min=1) - self.delta

        loss = -2 / self.beta * F.logsigmoid(self.beta * diff).mean()
        return loss, outputs
