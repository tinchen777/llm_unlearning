
from __future__ import annotations
from typing import TYPE_CHECKING

from .base import Evaluator

if TYPE_CHECKING:
    from utils.config import TrackingConfig


class TOFUEvaluator(Evaluator):
    def __init__(self, eval_cfg: TrackingConfig, **kwargs):
        super().__init__("TOFU", eval_cfg, **kwargs)
