
from __future__ import annotations

from utils.common import forward_batch, per_token_CE
from .base import ForgetRetainTrainer


class WGA(ForgetRetainTrainer):
    def __init__(
        self,
        beta: float = 1.0,
        gamma: float = 1.0,
        alpha: float = 1.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, gamma=gamma, alpha=alpha, **kwargs)
        self.beta = beta

    def compute_forget_loss(self, model, forget_inputs, **kwargs):
        logits, outputs = forward_batch(model, forget_inputs, ignore_labels=True)

        losses, shift_mask = per_token_CE(logits, forget_inputs["labels"])
        losses = losses.view(-1)

        neg_losses_exp = (-losses).exp().detach()
        weight_ce = neg_losses_exp ** self.beta
        forget_loss = -(weight_ce * losses)[shift_mask.view(-1)].mean()
        return forget_loss, outputs
