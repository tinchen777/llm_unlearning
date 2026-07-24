
from __future__ import annotations

from utils.common import forward_batch
from .base import ForgetRetainTrainer


class GradDiff(ForgetRetainTrainer):
    def compute_forget_loss(self, model, forget_inputs, **kwargs):
        loss, _, outputs = forward_batch(model, forget_inputs)
        loss = -loss  # maximize the loss for forget set
        return loss, outputs


class BoundedGradDiff(ForgetRetainTrainer):
    def __init__(self, forget_loss_bound: float = 4.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forget_loss_bound = forget_loss_bound  # tau: forget NLL 的上界

    def compute_forget_loss(self, model, forget_inputs, **kwargs):
        loss, _, outputs = forward_batch(model, forget_inputs)
        loss = -loss.clamp(max=self.forget_loss_bound)
        return loss, outputs
