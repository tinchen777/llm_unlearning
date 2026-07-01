
from __future__ import annotations
from typing import TYPE_CHECKING

from .base import Evaluator

if TYPE_CHECKING:
    from utils.config import TrackingConfig


class MUSEEvaluator(Evaluator):
    def __init__(self, eval_cfg: TrackingConfig, **kwargs):
        super().__init__("MUSE", eval_cfg, **kwargs)
