
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
    from data import Data

logger = logging.getLogger("eval.metric.mia")


@MetricMIAFunc
def mia_loss(model: Any, data: Data, **kwargs):
    return dict(attack_cls=LOSSAttack)


@MetricMIAFunc
def mia_min_k(model: Any, data: Data, k: float, **kwargs):
    return dict(attack_cls=MinKProbAttack, k=k)


@MetricMIAFunc
def mia_min_k_plus_plus(model: Any, data: Data, k: float, **kwargs):
    return dict(attack_cls=MinKPlusPlusAttack, k=k)


@MetricMIAFunc
def mia_gradnorm(model: Any, data: Data, p: float, **kwargs):
    return dict(attack_cls=GradNormAttack, p=p)


@MetricMIAFunc
def mia_zlib(model: Any, data: Data, tokenizer: Any, **kwargs):
    return dict(attack_cls=ZLIBAttack, tokenizer=tokenizer)


@MetricMIAFunc
def mia_reference(model: Any, data: Data, reference_model_path: str, **kwargs):
    logger.info(f"Loading reference model from `{reference_model_path}`")
    reference_model = AutoModelForCausalLM.from_pretrained(
        reference_model_path,
        dtype=model.dtype,
        device_map={"": model.device},
    )
    return dict(attack_cls=ReferenceAttack, reference_model=reference_model)
