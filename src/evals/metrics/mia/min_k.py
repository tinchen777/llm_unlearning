"""
Min-k % Prob Attack: https://arxiv.org/pdf/2310.16789.pdf
"""

from __future__ import annotations
import numpy as np
from typing import Mapping, Dict, TYPE_CHECKING

from .base import Attack
from ..metric_utils import tokenwise_logprobs

if TYPE_CHECKING:
    import torch


class MinKProbAttack(Attack):
    def setup(self, k: float = 0.2, **kwargs):
        self.k = k

    def compute_batch_values(self, batch: Mapping[str, torch.Tensor]):
        """Get token-wise log probabilities for the batch."""
        _, target_logprobs_batch, _ = tokenwise_logprobs(self.model, batch)
        return target_logprobs_batch

    def compute_score(self, sample_stats: torch.Tensor):
        """Score single sample using min-k negative log probs scores attack."""
        lp = sample_stats.float().cpu().numpy()
        if lp.size == 0:
            return 0.0

        num_k = max(1, int(len(lp) * self.k))
        sorted_vals = np.sort(lp)
        return float(-np.mean(sorted_vals[:num_k]))


class MinKPlusPlusAttack(Attack):
    def setup(self, k: float = 0.2, **kwargs):
        self.k = k

    def compute_batch_values(self, batch: Mapping[str, torch.Tensor]):
        """Get both token-wise and vocab-wise log probabilities for the batch."""
        vocab_logprobs_batch, target_logprobs_batch, _ = tokenwise_logprobs(self.model, batch)
        return [
            {"vocab_logprobs": vlp, "target_logprobs": tlp}
            for vlp, tlp in zip(vocab_logprobs_batch, target_logprobs_batch)
        ]

    def compute_score(self, sample_stats: Dict[str, torch.Tensor]):
        """Score using min-k negative log probs scores with vocab-wise normalization."""
        vocab_logprobs = sample_stats["vocab_logprobs"]
        target_logprobs = sample_stats["target_logprobs"]

        if len(target_logprobs) == 0:
            return 0.0

        # Compute normalized scores using vocab distribution
        mu = (vocab_logprobs.exp() * vocab_logprobs).sum(-1)
        sigma = (vocab_logprobs.exp() * vocab_logprobs.square()).sum(-1) - mu.square()

        # Handle numerical stability
        sigma = sigma.clamp(min=1e-6)
        scores = (
            target_logprobs.float().cpu().numpy() - mu.float().cpu().numpy()
        ) / sigma.sqrt().cpu().numpy()

        # Take bottom k% as the attack score
        num_k = max(1, int(len(scores) * self.k))
        return float(-np.mean(sorted(scores)[:num_k]))
