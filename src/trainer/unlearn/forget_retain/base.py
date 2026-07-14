
from __future__ import annotations
from typing import Any, Mapping, Optional, Tuple, TYPE_CHECKING

from utils.common import forward_batch
from ...utils import compute_kl_divergence
from ..base import UnlearnTrainer

if TYPE_CHECKING:
    import torch
    from transformers.modeling_outputs import CausalLMOutputWithPast


class ForgetRetainTrainer(UnlearnTrainer):
    requires_ref_model: bool = False

    def __init__(
        self,
        gamma: float = 1.0,
        alpha: float = 1.0,
        retain_loss_type: str = "NLL",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.gamma = gamma
        self.alpha = alpha
        self.retain_loss_type = retain_loss_type

        self.ref_model = None
        if retain_loss_type == "KL" or self.requires_ref_model:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_forget_loss(self, model: Any, forget_inputs: Mapping[str, Any], **kwargs) -> Tuple[torch.Tensor, CausalLMOutputWithPast]:
        raise NotImplementedError("compute_forget_loss must be implemented in subclasses.")

    def compute_retain_loss(self, model: Any, retain_inputs: Mapping[str, torch.Tensor], **kwargs):
        if self.retain_loss_type == "NLL":
            loss, _, _ = forward_batch(model, retain_inputs)
        elif self.retain_loss_type == "KL":
            loss, _ = compute_kl_divergence(model, self.ref_model, retain_inputs)
        else:
            raise NotImplementedError(f"{self.retain_loss_type} not implemented for retain set.")
        return loss

    def aggregate_loss(self, forget_loss: torch.Tensor, retain_loss: torch.Tensor) -> torch.Tensor:
        return self.gamma * forget_loss + self.alpha * retain_loss

    def compute_loss(
        self,
        model: Any,
        inputs: Mapping[str, Mapping[str, torch.Tensor]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
        **kwargs
    ):
        # forget loss
        forget_loss, forget_outputs = self.compute_forget_loss(model=model, forget_inputs=inputs["forget"])
        # retain loss
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=inputs["retain"])

        loss = self.aggregate_loss(forget_loss, retain_loss)
        return (loss, forget_outputs) if return_outputs else loss
