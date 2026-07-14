
from __future__ import annotations
from typing import Any, Mapping, Optional, TYPE_CHECKING

from utils.common import forward_batch
from .base import UnlearnTrainer

if TYPE_CHECKING:
    import torch


class GradAscent(UnlearnTrainer):
    def compute_loss(
        self,
        model: Any,
        inputs: Mapping[str, Mapping[str, torch.Tensor]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
        **kwargs
    ):
        loss, _, outputs = forward_batch(model, inputs["forget"])
        loss = -loss  # maximize the loss for forget set

        return (loss, outputs) if return_outputs else loss
