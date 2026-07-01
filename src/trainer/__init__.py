
from __future__ import annotations
import torch
import logging
from transformers import Trainer, TrainingArguments
from typing import Dict, Any, Optional, TYPE_CHECKING

from .base import FinetuneTrainer
from .unlearn.grad_ascent import GradAscent
from .unlearn.grad_diff import GradDiff
from .unlearn.npo import NPO
from .unlearn.dpo import DPO
from .unlearn.simnpo import SimNPO
from .unlearn.rmu import RMU
from .unlearn.undial import UNDIAL
from .unlearn.ceu import CEU
from .unlearn.satimp import SatImp
from .unlearn.wga import WGA
from .unlearn.pdu import PDU
from .unlearn.bounded_grad_diff import BoundedGradDiff

if TYPE_CHECKING:
    from utils.config import TrackingConfig

logger = logging.getLogger(__name__)

TRAINER_REGISTRY: Dict[str, type[FinetuneTrainer]] = {}


def _register_trainer(trainer_class):
    TRAINER_REGISTRY[trainer_class.__name__] = trainer_class


def load_trainer(
    trainer_cfg: TrackingConfig,
    model: Any,
    evaluators: Any = None,
    train_dataset: Optional[Any] = None,
    eval_dataset: Optional[Any] = None,
    processing_class: Optional[Any] = None,
    data_collator: Optional[Any] = None
) -> FinetuneTrainer:
    args = _load_trainer_args(
        trainer_cfg.get("args", {}),
        len(train_dataset) if train_dataset else 0
    )
    trainer_name = trainer_cfg["handler"]
    trainer_cls = TRAINER_REGISTRY[trainer_name]
    trainer = trainer_cls(
        evaluators=evaluators,
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
        **trainer_cfg.get("method_args", {}),
    )
    logger.info(
        f"Trainer `{trainer_name}` loaded, output_dir: {args.output_dir}"
    )
    return trainer


def _load_trainer_args(trainer_args: TrackingConfig, dataset_len: int):
    warmup_epochs = trainer_args.pop("warmup_epochs", None, allow_none=True)
    if warmup_epochs:
        batch_size = trainer_args["per_device_train_batch_size"]
        grad_accum_steps = trainer_args["gradient_accumulation_steps"]
        num_devices = torch.cuda.device_count()
        trainer_args["warmup_steps"] = int(
            (warmup_epochs * dataset_len) // (batch_size * grad_accum_steps * num_devices)
        )
    return TrainingArguments(**trainer_args)


# Register Finetuning Trainer
_register_trainer(Trainer)
_register_trainer(FinetuneTrainer)

# Register Unlearning Trainer
_register_trainer(GradAscent)
_register_trainer(GradDiff)
_register_trainer(NPO)
_register_trainer(DPO)
_register_trainer(SimNPO)
_register_trainer(RMU)
_register_trainer(UNDIAL)
_register_trainer(CEU)
_register_trainer(SatImp)
_register_trainer(WGA)
_register_trainer(PDU)

# Register custom example unlearning trainer (see demos/)
_register_trainer(BoundedGradDiff)
