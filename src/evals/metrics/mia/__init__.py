
from __future__ import annotations
import numpy as np
import logging
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM
from typing import Any, TYPE_CHECKING

from ..base import MetricFunc
from .loss import LOSSAttack
from .min_k import MinKProbAttack
from .min_k_plus_plus import MinKPlusPlusAttack
from .gradnorm import GradNormAttack
from .zlib import ZLIBAttack
from .reference import ReferenceAttack

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from .base import Attack

logger = logging.getLogger("metrics")


@MetricFunc
def mia_loss(model: Any, forget_dl: DataLoader, holdout_dl: DataLoader, **kwargs):
    
    
    
    
    return mia_auc(
        LOSSAttack,
        model,
        forget_dl=forget_dl,
        holdout_dl=holdout_dl,
    )

# TODO
@MetricFunc
def mia_min_k(model: Any, forget_dl: DataLoader, holdout_dl: DataLoader, k: float, **kwargs):
    return mia_auc(
        MinKProbAttack,
        model,
        forget_dl=forget_dl,
        holdout_dl=holdout_dl,
        k=k,
    )


@MetricFunc
def mia_min_k_plus_plus(model: Any, **kwargs):
    return mia_auc(
        MinKPlusPlusAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
        k=kwargs["k"],
    )


@MetricFunc
def mia_gradnorm(model: Any, **kwargs):
    return mia_auc(
        GradNormAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
        p=kwargs["p"],
    )


@MetricFunc
def mia_zlib(model: Any, **kwargs):
    return mia_auc(
        ZLIBAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
        tokenizer=kwargs.get("tokenizer"),
    )


@MetricFunc
def mia_reference(model: Any, **kwargs):
    if "reference_model_path" not in kwargs:
        raise ValueError("Reference model must be provided in kwargs")
    logger.info(f"Loading reference model from {kwargs['reference_model_path']}")
    reference_model = AutoModelForCausalLM.from_pretrained(
        kwargs["reference_model_path"],
        dtype=model.dtype,  # transformers>=4.56 renamed `torch_dtype` -> `dtype`
        device_map={"": model.device},
    )
    return mia_auc(
        ReferenceAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
        reference_model=reference_model,
    )


def mia_auc(
    attack_cls: type[Attack],
    model: Any,
    forget_dl: DataLoader,
    holdout_dl: DataLoader,
    **kwargs
):
    """
    Compute the MIA AUC and accuracy.

    Parameters:
      - attack_cls: the attack class to use.
      - model: the target model.
      - data: a dict with keys "forget" and "holdout".
      - collator: data collator.
      - batch_size: batch size.
      - kwargs: additional optional parameters (e.g. k, p, tokenizer, reference_model).

    Returns a dict containing the attack outputs, including "acc" and "auc".

    Note on convention: auc is 1 when the forget data is much more likely than the holdout data
    """
    forget = attack_cls(model=model, dataloader=forget_dl, **kwargs).attack()
    holdout = attack_cls(model=model, dataloader=holdout_dl, **kwargs).attack()
    forget_scores = [elem["score"] for elem in forget["value_by_index"].values()]
    holdout_scores = [elem["score"] for elem in holdout["value_by_index"].values()]

    scores = np.array(forget_scores + holdout_scores)
    labels = np.array([0] * len(forget_scores) + [1] * len(holdout_scores))
    auc_value = roc_auc_score(labels, scores)

    return {
        "forget": forget,
        "holdout": holdout,
        "auc": auc_value,
        "agg_value": auc_value
    }
