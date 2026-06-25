
from __future__ import annotations
from typing import Dict, Any, TYPE_CHECKING

from .tofu import TOFUEvaluator
from .muse import MUSEEvaluator
from .lm_eval import LMEvalEvaluator

if TYPE_CHECKING:
    from utils.config import TrackingConfig

EVALUATOR_REGISTRY: Dict[str, Any] = {}


def _register_evaluator(evaluator_cls):
    EVALUATOR_REGISTRY[evaluator_cls.__name__] = evaluator_cls


def get_evaluators(eval_cfgs: TrackingConfig, **kwargs):
    evaluators = {}
    for eval_name, eval_cfg in eval_cfgs.items():
        try:
            evaluators[eval_name] = _get_evaluator(eval_cfg, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Error loading evaluator `{eval_name}` in `@{eval_cfg.loc_choices}` with {eval_cfg}") from e
    return evaluators


def _get_evaluator(eval_cfg: TrackingConfig, **kwargs):
    eval_cls = EVALUATOR_REGISTRY[eval_cfg["handler"]]
    return eval_cls(eval_cfg, **kwargs)


# Register Your benchmark evaluators
_register_evaluator(TOFUEvaluator)
_register_evaluator(MUSEEvaluator)
_register_evaluator(LMEvalEvaluator)
