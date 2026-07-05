"""
Reference-based attacks.
"""

from __future__ import annotations
from typing import Any, Mapping, Dict, TYPE_CHECKING

from .base import Attack
from ..metric_utils import evaluate_probability

if TYPE_CHECKING:
    import torch


class ReferenceAttack(Attack):
    def setup(self, reference_model: Any, **kwargs):
        """Setup reference model."""
        self.reference_model = reference_model

    def compute_batch_values(self, batch: Mapping[str, torch.Tensor]):
        """Compute loss scores for both target and reference models."""
        ref_results = evaluate_probability(self.reference_model, batch)
        target_results = evaluate_probability(self.model, batch)
        return [
            {"target_loss": t["avg_loss"], "ref_loss": r["avg_loss"]}
            for t, r in zip(target_results, ref_results)
        ]

    def compute_score(self, sample_stats: Dict[str, float]):
        """Score using difference between target and reference model losses."""
        return sample_stats["target_loss"] - sample_stats["ref_loss"]
