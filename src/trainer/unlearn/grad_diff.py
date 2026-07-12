
from __future__ import annotations
from typing import Any, Mapping, Optional, TYPE_CHECKING

from utils.common import forward_batch
from ..utils import compute_kl_divergence
from .base import UnlearnTrainer

if TYPE_CHECKING:
    import torch


class GradDiff(UnlearnTrainer):
    def __init__(self, gamma: float = 1.0, alpha: float = 1.0, retain_loss_type: str = "NLL", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gamma = gamma
        self.alpha = alpha
        self.retain_loss_type = retain_loss_type

        self.ref_model = None
        if retain_loss_type == "KL":
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_retain_loss(self, model: Any, retain_inputs: Mapping[str, torch.Tensor]):
        if self.retain_loss_type == "NLL":
            retain_loss, _, _ = forward_batch(model, retain_inputs)
        elif self.retain_loss_type == "KL":
            retain_loss, _ = compute_kl_divergence(
                self.model, self.ref_model, retain_inputs
            )
        else:
            raise NotImplementedError(f"{self.retain_loss_type} not implemented for retain set.")
        return retain_loss

    def compute_loss(
        self,
        model: Any,
        inputs: Mapping[str, Mapping[str, torch.Tensor]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
        **kwargs
    ):
        # forget loss
        forget_loss, _, forget_outputs = forward_batch(model, inputs["forget"])
        forget_loss = -forget_loss  # maximize the loss for forget set
        # retain loss
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=inputs["retain"])

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        return (loss, forget_outputs) if return_outputs else loss
