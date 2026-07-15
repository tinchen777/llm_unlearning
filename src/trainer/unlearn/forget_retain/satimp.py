
from __future__ import annotations

from utils.common import forward_batch, per_token_CE
from .base import ForgetRetainTrainer


class SatImp(ForgetRetainTrainer):
    def __init__(
        self,
        beta1: float = 5.0,
        beta2: float = 1.0,
        gamma: float = 0.1,
        alpha: float = 1.0,
        *args,
        **kwargs
    ):
        # attention, satimp requires two beta!!!!
        super().__init__(*args, gamma=gamma, alpha=alpha, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2

    def compute_forget_loss(self, model, forget_inputs, **kwargs):
        logits, outputs = forward_batch(model, forget_inputs, ignore_labels=True)

        losses, shift_mask = per_token_CE(logits, forget_inputs["labels"])
        losses = losses.view(-1)

        neg_losses_exp = (-losses).exp().detach()
        weight_sat = neg_losses_exp ** self.beta1
        weight_imp = (1 - neg_losses_exp) ** self.beta2
        loss = -((weight_sat * weight_imp) * losses)[shift_mask.view(-1)].mean()

        return loss, outputs
