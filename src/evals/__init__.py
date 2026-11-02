
from __future__ import annotations
from typing import Dict, Any, TYPE_CHECKING

from .tofu import TOFUEvaluator
from .muse import MUSEEvaluator
from .lm_eval import LMEvalEvaluator

if TYPE_CHECKING:
    from .base import Evaluator
    from utils.config import TrackingConfig

EVALUATOR_REGISTRY: Dict[str, type[Evaluator]] = {}


def _register_evaluator(evaluator_cls: type[Evaluator]):
    EVALUATOR_REGISTRY[evaluator_cls.__name__] = evaluator_cls


def get_evaluators(eval_cfgs: TrackingConfig, **metrics_kwargs: Any):
    evaluators: Dict[str, Evaluator] = {}
    for name, evaluator_cfg in eval_cfgs.items():
        try:
            evaluators[name] = _get_evaluator(evaluator_cfg, **metrics_kwargs)
        except Exception as e:
            raise RuntimeError(f"Error loading `{name}` evaluator in `@{evaluator_cfg.loc_choices}`") from e
    return evaluators


def _get_evaluator(evaluator_cfg: TrackingConfig, **metrics_kwargs):
    eval_cls = EVALUATOR_REGISTRY[evaluator_cfg["handler"]]
    return eval_cls(evaluator_cfg=evaluator_cfg, **metrics_kwargs)


# Register Your benchmark evaluators
_register_evaluator(TOFUEvaluator)
_register_evaluator(MUSEEvaluator)
_register_evaluator(LMEvalEvaluator)
