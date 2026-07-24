
from __future__ import annotations
from copy import deepcopy
from accelerate.utils import is_deepspeed_available
from typing import Any

from ..base import FinetuneTrainer

if is_deepspeed_available():
    import deepspeed
else:
    deepspeed = None


class UnlearnTrainer(FinetuneTrainer):

    def _prepare_ref_model(self, model: Any):
        ref_model = deepcopy(model).to(self.accelerator.device)
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        ref_model.eval()

        return ref_model

    def _prepare_deepspeed(self, model: Any):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        if deepspeed_plugin is None:
            raise ValueError(
                "DeepSpeed is not enabled. Please ensure that the DeepSpeed plugin is properly configured."
            )
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        hidden_size = (
            max(model.config.hidden_sizes)
            if getattr(model.config, "hidden_sizes", None)
            else getattr(model.config, "hidden_size", None)
        )
        if (
            hidden_size is not None
            and config_kwargs["zero_optimization"]["stage"] == 3
        ):
            # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
            # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
            config_kwargs.update({
                "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
            })

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0

        if deepspeed is None:
            raise ImportError(
                "DeepSpeed is not installed. Please install DeepSpeed to use this feature."
            )
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)

        return model

    def prediction_step(self, *args, **kwargs):
        # as compute_loss often overridden by unlearning methods, and we want to maintain the Trainer's evaluation setup.
        self.compute_loss = super().compute_loss
        try:
            return super().prediction_step(*args, **kwargs)
        finally:
            del self.compute_loss
