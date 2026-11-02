
from __future__ import annotations
from typing import TYPE_CHECKING

from .base import Evaluator

if TYPE_CHECKING:
    from utils.config import TrackingConfig


class TOFUEvaluator(Evaluator):
    def __init__(self, evaluator_cfg: TrackingConfig, **metrics_kwargs):
        super().__init__("TOFU", evaluator_cfg, **metrics_kwargs)
