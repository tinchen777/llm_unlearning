
from __future__ import annotations
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from abc import ABC, abstractmethod
from typing import Any, List, Mapping, TYPE_CHECKING

from ..base import MetricFunc

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader


class Attack(ABC):
    def __init__(self, model: Any, **kwargs):
        """Initialize attack with model and create dataloader."""
        self.model = model
        self.setup(**kwargs)

    def setup(self, *args, **kwargs): ...

    @abstractmethod
    def compute_batch_values(self, batch: Mapping[str, torch.Tensor]) -> List[Any]:
        """Process a batch through model to get needed statistics."""
        ...

    @abstractmethod
    def compute_score(self, sample_stats: Any) -> float:
        """Compute MIA score for a single sample."""
        ...

    def attack(self, dataloader: DataLoader):
        """Run full MIA attack."""
        all_scores: List[float] = []
        all_indices: List[int] = []

        for batch in tqdm(dataloader, total=len(dataloader)):
            indices = batch.pop("index")
            batch_values = self.compute_batch_values(batch)
            scores = [self.compute_score(values) for values in batch_values]

            all_scores.extend(scores)
            all_indices.extend(indices)

        scores_by_index = {
            str(idx): {"score": score}
            for idx, score in zip(all_indices, all_scores)
        }

        return {
            "agg_value": float(np.mean(all_scores)),
            "value_by_index": scores_by_index,
        }, all_scores


class MetricMIAFunc(MetricFunc):
    def __call__(self, model, forget_dl, holdout_dl, **kwargs):
        cls_and_kwargs = self.func(model, forget_dl, holdout_dl, **kwargs)
        return mia_auc(
            model=model,
            forget_dl=forget_dl,
            holdout_dl=holdout_dl,
            **cls_and_kwargs
        )


def mia_auc(
    attack_cls: type[Attack],
    model: Any,
    forget_dl: DataLoader,
    holdout_dl: DataLoader,
    **kwargs
):
    """
    Compute the MIA AUC and accuracy.

    Parameters:
    - attack_cls: The class of the attack to be used (e.g., LOSSAttack, MinKProbAttack).
    - model: The model to be attacked.
    - forget_dl: DataLoader for the forget dataset.
    - holdout_dl: DataLoader for the holdout dataset.
    - kwargs: Additional optional parameters (e.g., k, p, tokenizer, reference_model).

    Returns a dict containing the attack outputs, including "acc" and "auc".

    Note on convention: auc is 1 when the forget data is much more likely than the holdout data
    """
    attacker = attack_cls(model=model, **kwargs)

    forget, forget_scores = attacker.attack(forget_dl)
    holdout, holdout_scores = attacker.attack(holdout_dl)

    scores = np.array(forget_scores + holdout_scores)
    labels = np.array([0] * len(forget_scores) + [1] * len(holdout_scores))
    auc_value = roc_auc_score(labels, scores)

    return {
        "forget": forget,
        "holdout": holdout,
        "auc": auc_value,
        "agg_value": auc_value
    }
