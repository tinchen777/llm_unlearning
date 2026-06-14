# Modified from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py

import logging
import os
from typing import Any, Dict, List, Optional, Union
from xml.parsers.expat import model
from omegaconf import OmegaConf

from torch.utils.data import Dataset
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from .base import FinetuneTrainer

# 需要安装 peft 库：pip install peft
# from peft import PeftModel, get_peft_model, PeftConfig
from peft import LoraConfig, get_peft_model, PeftModel

logger = logging.getLogger(__name__)

_EVAL_PLACEHOLDER = "_EVAL_PLACEHOLDER"


class LoraFinetuneTrainer(FinetuneTrainer):
    """
    Trainer for LoRA fine-tuning.
    Expects `lora_config` dict inside `method_args` (from config).
    """

    def __init__(
        self,
        lora_config: Optional[Dict] = None,
        *args,
        **kwargs
    ):
        # 1. 提取 model 并注入 LoRA
        model = kwargs.get("model")
        if model is not None and lora_config is not None:
            lora_config = OmegaConf.to_container(lora_config, resolve=True)
            peft_config = LoraConfig(**lora_config)
            # 只对模型主体注入 LoRA，lm_head 等保持原样
            model = get_peft_model(model, peft_config)
            # 手动调用，保证 checkpointing 下梯度链不断
            model.enable_input_require_grads()

            kwargs["model"] = model
            logger.info("LoRA adapters injected into the model.")

        self.lora_config = lora_config
        if isinstance(model, PeftModel):
            model.print_trainable_parameters()
        logger.info("Attention implementation: %s", model.config._attn_implementation)
        super().__init__(*args, **kwargs)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Override to save only the trainable adapter weights.
        If output_dir is not provided, use the one from args.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        os.makedirs(output_dir, exist_ok=True)

        # 保存 adapter 权重（仅可训练参数）
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(output_dir)
            # 同时保存 tokenizer 以便后续加载
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)
            logger.info(f"LoRA adapter saved to {output_dir}")
        else:
            # fallback 到原始保存逻辑（安全兜底）
            super().save_model(output_dir, _internal_call)

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        # 保持父类的日志和评估逻辑不变
        return super()._maybe_log_save_evaluate(*args, **kwargs)