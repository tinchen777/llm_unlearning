"""
Straight-forward LOSS attack, as described in https://ieeexplore.ieee.org/abstract/document/8429311
"""

from __future__ import annotations
from typing import Dict

from .base import Attack
from ..metric_utils import evaluate_probability


class LOSSAttack(Attack):
    def compute_batch_values(self, batch):
        """Compute probabilities and losses for the batch."""
        return evaluate_probability(self.model, batch)

    def compute_score(self, sample_stats: Dict[str, float]):
        """Return the average loss for the sample."""
        return sample_stats["avg_loss"]
