"""
zlib-normalization Attack: https://www.usenix.org/system/files/sec21-carlini-extracting.pdf
"""

from __future__ import annotations
import zlib
from typing import Any, Optional, Mapping, Dict, TYPE_CHECKING

from .base import Attack
from ..metric_utils import (
    evaluate_probability,
    get_decoded_target_texts
)

if TYPE_CHECKING:
    import torch


class ZLIBAttack(Attack):
    def setup(self, tokenizer: Optional[Any] = None, **kwargs):
        """Setup tokenizer."""
        self.tokenizer = tokenizer or self.model.tokenizer

    def compute_batch_values(self, batch: Mapping[str, torch.Tensor]):
        """Get loss and text for batch."""
        texts = get_decoded_target_texts(self.tokenizer, batch)
        eval_results = evaluate_probability(self.model, batch)
        return [{"loss": r["avg_loss"], "text": t} for r, t in zip(eval_results, texts)]

    def compute_score(self, sample_stats: Dict[str, Any]):
        """Score using loss normalized by compressed text length."""
        text = sample_stats["text"]
        zlib_entropy = len(zlib.compress(text.encode("utf-8")))
        return float(sample_stats["loss"] / zlib_entropy)
