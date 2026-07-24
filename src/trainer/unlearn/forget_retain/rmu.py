"""Borrowed implementation from https://github.com/centerforaisafety/wmdp/blob/main/rmu/unlearn.py"""

from __future__ import annotations
import re
import torch
from torch.nn import functional as F
import deepspeed
from typing import List, Any, Optional, Mapping

from utils.common import forward_batch, IGNORE_INDEX
from .base import ForgetRetainTrainer

MODULE_REGEX = r"model\.layers\.7"
TRAINABLE_PARAMS_REGEX = [r"model\.layers\.(5|6|7)\.mlp\.down_proj\.weight"]


class RMU(ForgetRetainTrainer):
    requires_ref_model = True

    def __init__(
        self,
        module_regex: str = MODULE_REGEX,
        trainable_params_regex: Optional[List[str]] = None,
        steering_coeff: float = 20,
        *args,
        **kwargs,
    ):
        """
        RMU Trainer that fine-tunes only specific layers and parameters using regex-based filtering.

        Args:
            module_path (str): Regex pattern to match module names.
            trainable_param_paths (list of str): List of regex patterns for trainable parameters.
        """
        super().__init__(*args, **kwargs)

        # Unfreeze only the selected parameters
        self.trainable_params_regex = trainable_params_regex or TRAINABLE_PARAMS_REGEX
        # Get actual module references
        self.module_regex = module_regex  # Regex for selecting modules

        self.model_module = self._get_matching_module(self.model)
        self.ref_module = self._get_matching_module(self.ref_model)
        self.steering_coeff = steering_coeff
        self.control_vec = None

    def create_optimizer(self, model=None):
        self._freeze_all_params(self.model, False)
        # This makes the optimizer to select only trainable params
        self._set_trainable_params(self.model, True)

        optimizer = super().create_optimizer()

        self._freeze_all_params(self.model, True)

        return optimizer

    def _get_matching_module(self, model: Any):
        """Returns a single module matching the given regex from a DeepSpeed/DDP-wrapped model."""
        # Handle DeepSpeed and DDP-wrapped models by accessing the underlying module
        if isinstance(model, deepspeed.DeepSpeedEngine):
            model = model.module  # Extract the actual PyTorch model inside

        matched_modules = {
            name: module
            for name, module in model.named_modules()
            if re.fullmatch(self.module_regex, name)
        }

        if len(matched_modules) > 1:
            raise ValueError(
                f"More than one module matched with {self.module_regex}: {list(matched_modules.keys())}"
            )
        elif not matched_modules:
            raise ValueError(f"No module matched with {self.module_regex}")

        return next(iter(matched_modules.values()))  # Return the single matched module

    @staticmethod
    def _freeze_all_params(model: Any, requires_grad: bool):
        """Freeze all parameters in the model initially."""
        for param in model.parameters():
            param.requires_grad = requires_grad

    def _set_trainable_params(self, model: Any, requires_grad: bool):
        """Unfreeze specific parameters that match the regex patterns."""
        for name, param in model.named_parameters():
            if any(re.fullmatch(pattern, name) for pattern in self.trainable_params_regex):
                param.requires_grad = requires_grad

    def forward_with_cache(
        self,
        model: Any,
        inputs: Mapping[str, torch.Tensor],
        module: Any,
        grad: bool = True
    ):
        """Performs a forward pass while caching the output of a specified module."""
        cache = []

        def hook(module, input, output):
            if isinstance(output, tuple):
                cache.append(output[0])
            else:
                cache.append(output)
            return None

        hook_handle = module.register_forward_hook(hook)
        _, outputs = forward_batch(model, inputs, ignore_labels=True, grad=grad)
        hook_handle.remove()
        return cache[0], outputs

    def get_control_vector(self, dim: int):
        if self.control_vec is None:
            random_vector = torch.rand(1, 1, dim)
            self.control_vec = random_vector / random_vector.norm() * self.steering_coeff
        return self.control_vec

    def compute_activation_loss(
        self,
        activation1: torch.Tensor,
        activation2: torch.Tensor,
        mask: torch.Tensor
    ):
        squared_diff = F.mse_loss(activation1, activation2, reduction="none")  # Shape (b, s, d)
        expanded_mask = mask.unsqueeze(-1).expand_as(squared_diff)  # Shape: [b, s, d]
        squared_diff_sum = (squared_diff * expanded_mask).mean(dim=2).sum(dim=1)  # Shape: [b, 1]
        num_tokens = mask.sum(dim=-1, keepdim=True).clamp(min=1)  # Sum over seq_len, Shape: [b, 1]
        return (squared_diff_sum / num_tokens).mean()

    def compute_forget_loss(self, model, forget_inputs, **kwargs):
        model_activations, outputs = self.forward_with_cache(
            model, forget_inputs, module=self.model_module
        )
        # If multiple datasets or concepts need unlearning, pass the control vector during processing; otherwise, default to a random vector during training.
        control_vec = forget_inputs.get("control_vec")
        if control_vec is None:
            control_vec = self.get_control_vector(model_activations.shape[-1])
        control_vec = control_vec.to(
            dtype=model_activations.dtype,
            device=model_activations.device
        )
        control_vec = control_vec.expand_as(model_activations)
        mask = forget_inputs["labels"] != IGNORE_INDEX  # Shape: [b, s]
        loss = self.compute_activation_loss(
            model_activations, control_vec, mask
        )
        return loss, outputs

    def compute_retain_loss(self, model, retain_inputs, **kwargs):
        if self.retain_loss_type == "EMBED_DIFF":
            model_activations, _ = self.forward_with_cache(
                model, retain_inputs, module=self.model_module
            )
            ref_activations, _ = self.forward_with_cache(
                self.ref_model, retain_inputs, module=self.ref_module, grad=False
            )
            mask = retain_inputs["labels"] != IGNORE_INDEX  # Shape: [b, s]
            return self.compute_activation_loss(
                model_activations,
                ref_activations.to(model_activations.device),
                mask,
            )
        return super().compute_retain_loss(model, retain_inputs)
