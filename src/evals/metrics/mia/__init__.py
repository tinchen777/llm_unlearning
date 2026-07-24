
from __future__ import annotations
import logging
from transformers import AutoModelForCausalLM
from typing import Any, TYPE_CHECKING

from .base import MetricMIAFunc
from .loss import LOSSAttack
from .min_k import MinKProbAttack, MinKPlusPlusAttack
from .gradnorm import GradNormAttack
from .zlib import ZLIBAttack
from .reference import ReferenceAttack

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from .base import Attack

logger = logging.getLogger("eval.metric.mia")


@MetricMIAFunc
def mia_loss(model: Any, forget_dl: DataLoader, holdout_dl: DataLoader, **kwargs):
    return dict(attack_cls=LOSSAttack)


@MetricMIAFunc
def mia_min_k(model: Any, forget_dl: DataLoader, holdout_dl: DataLoader, k: float, **kwargs):
    return dict(attack_cls=MinKProbAttack, k=k)


@MetricMIAFunc
def mia_min_k_plus_plus(model: Any, forget_dl: DataLoader, holdout_dl: DataLoader, k: float, **kwargs):
    return dict(attack_cls=MinKPlusPlusAttack, k=k)


@MetricMIAFunc
def mia_gradnorm(model: Any, forget_dl: DataLoader, holdout_dl: DataLoader, p: float, **kwargs):
    return dict(attack_cls=GradNormAttack, p=p)


@MetricMIAFunc
def mia_zlib(model: Any, forget_dl: DataLoader, holdout_dl: DataLoader, tokenizer: Any, **kwargs):
    return dict(attack_cls=ZLIBAttack, tokenizer=tokenizer)


@MetricMIAFunc
def mia_reference(model: Any, forget_dl: DataLoader, holdout_dl: DataLoader, reference_model_path: str, **kwargs):
    logger.info(f"Loading reference model from {reference_model_path}")
    reference_model = AutoModelForCausalLM.from_pretrained(
        reference_model_path,
        dtype=model.dtype,  # transformers>=4.56 renamed `torch_dtype` -> `dtype`
        device_map={"": model.device},
    )
    return dict(attack_cls=ReferenceAttack, reference_model=reference_model)
