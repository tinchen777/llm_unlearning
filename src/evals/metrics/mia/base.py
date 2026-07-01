
from __future__ import annotations
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class Attack(ABC):
    def __init__(self, model: Any, **kwargs):
        """Initialize attack with model and create dataloader."""
        self.model = model
        self.setup(**kwargs)

    def setup(self, *args, **kwargs): ...

    @abstractmethod
    def compute_batch_values(self, batch) -> List[Any]:
        """Process a batch through model to get needed statistics."""
        ...

    @abstractmethod
    def compute_score(self, sample_stats):
        """Compute MIA score for a single sample."""
        ...

    def attack(self, dataloader: DataLoader):
        """Run full MIA attack."""
        all_scores = []
        all_indices = []

        for batch in tqdm(dataloader, total=len(dataloader)):
            indices = batch.pop("index").cpu().numpy().tolist()
            batch_values = self.compute_batch_values(batch)
            scores = [self.compute_score(values) for values in batch_values]

            all_scores.extend(scores)
            all_indices.extend(indices)

        scores_by_index = {
            str(idx): {"score": float(score)}
            for idx, score in zip(all_indices, all_scores)
        }

        return {
            "agg_value": float(np.mean(all_scores)),
            "value_by_index": scores_by_index,
        }
