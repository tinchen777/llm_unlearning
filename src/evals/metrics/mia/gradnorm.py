"""
Gradient-norm attack. Proposed for MIA in multiple settings, and particularly
experimented for pre-training data and LLMs in https://arxiv.org/abs/2402.17012
"""

from __future__ import annotations
import torch
from typing import Union, Mapping, List

from .base import Attack
from ..metric_utils import tokenwise_logprobs
from ..utils import to_np


# DO NOT use gradnorm in a way so that it runs when your accumulated gradients during training aren't used yet
# gradnorm zeros out the gradients of the model during its computation
class GradNormAttack(Attack):
    def setup(self, p: Union[int, float], **kwargs):
        if p not in [1, 2, float("inf")]:
            raise ValueError(f"Invalid p-norm value: {p}")
        self.p = p

    def compute_batch_values(self, batch: Mapping[str, torch.Tensor]):
        """Compute gradients of examples w.r.t model parameters. More grad norm => more loss."""
        self.model.train()
        _, target_logprobs_batch, _ = tokenwise_logprobs(self.model, batch, grad=True)
        batch_loss = [-lps.mean() for lps in target_logprobs_batch]
        batch_grad_norms: List[torch.Tensor] = []
        for sample_loss in batch_loss:
            sample_grad_norms = []
            self.model.zero_grad()
            sample_loss.backward()
            for param in self.model.parameters():
                if param.grad is not None:
                    sample_grad_norms.append(param.grad.detach().norm(p=self.p))
            batch_grad_norms.append(torch.stack(sample_grad_norms).mean())
        self.model.eval()
        return batch_grad_norms

    def compute_score(self, sample_stats: torch.Tensor):
        return float(to_np(sample_stats))
