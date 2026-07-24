"""
Min-k % Prob Attack: https://arxiv.org/pdf/2310.16789.pdf
"""

from __future__ import annotations
from typing import Dict, TYPE_CHECKING

from .base import Attack
from ..metric_utils import tokenwise_logprobs
from ..utils import topk_mean

if TYPE_CHECKING:
    import torch


class MinKProbAttack(Attack):
    def setup(self, k: float = 0.2, **kwargs):
        self.k = k

    def compute_batch_values(self, batch):
        """Get token-wise log probabilities for the batch."""
        _, target_logprobs_batch, _ = tokenwise_logprobs(self.model, batch)
        return target_logprobs_batch

    def compute_score(self, sample_stats: torch.Tensor):
        """Score single sample using min-k negative log probs scores attack.
        sample_stats shape as [seq_len]
        """
        # Take bottom k% as the attack score
        return -topk_mean(sample_stats, self.k, largest=False)


class MinKPlusPlusAttack(Attack):
    def setup(self, k: float = 0.2, **kwargs):
        self.k = k

    def compute_batch_values(self, batch):
        """Get both token-wise and vocab-wise log probabilities for the batch."""
        vocab_logprobs_batch, target_logprobs_batch, _ = tokenwise_logprobs(self.model, batch)
        return [
            {"vocab_logprobs": vlp, "target_logprobs": tlp}
            for vlp, tlp in zip(vocab_logprobs_batch, target_logprobs_batch)
        ]

    def compute_score(self, sample_stats: Dict[str, torch.Tensor]):
        """Score using min-k negative log probs scores with vocab-wise normalization."""
        vocab_logprobs = sample_stats["vocab_logprobs"]  # shape as [seq_len, vocab_size]
        target_logprobs = sample_stats["target_logprobs"]  # shape as [seq_len]

        if len(target_logprobs) == 0:
            return 0.0

        # Compute normalized scores using vocab distribution
        mu = (vocab_logprobs.exp() * vocab_logprobs).sum(-1)  # shape as [seq_len]
        sigma = (vocab_logprobs.exp() * vocab_logprobs.square()).sum(-1) - mu.square()  # shape as [seq_len]

        # Handle numerical stability
        sigma = sigma.clamp(min=1e-6)
        scores = (target_logprobs - mu) / sigma.sqrt()  # shape as [seq_len]

        # Take bottom k% as the attack score
        return -topk_mean(scores, self.k, largest=False)
