"""
Utility functions for `data` module.
"""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader, Subset
from typing import Any, Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import Dataset

IGNORE_INDEX = -100


def fixed_subset(
    dataset: Dataset,
    max_samples: Optional[int] = None,
    seed: int = 42
) -> Dataset:
    """
    A FIXED random subset of `max_samples` samples (sorted indices; the same for every call / model given `seed`),
    or the dataset itself when `max_samples` is None or not smaller than the dataset.
    """
    if max_samples is None or int(max_samples) >= len(dataset):  # type: ignore
        return dataset
    idx = torch.randperm(
        len(dataset),  # type: ignore
        generator=torch.Generator().manual_seed(seed)
    )[:int(max_samples)]
    return Subset(dataset, sorted(idx.tolist()))


def one_dataset(datasets: Dict[str, Dataset]) -> Dataset:
    if len(datasets) != 1:
        raise ValueError(f"Expected exactly one dataset for `dataloader`, but got {len(datasets)}.")
    return next(iter(datasets.values()))


def get_subloader(
    dataset: Dataset,
    batch_size: int,
    collator: Optional[Any] = None,
    max_samples: Optional[int] = None,
    seed: int = 42,
    **loader_kwargs
) -> DataLoader:
    """Unshuffled loader of the single dataset on a fixed subset."""
    return DataLoader(
        fixed_subset(dataset, max_samples=max_samples, seed=seed),
        batch_size=batch_size,
        collate_fn=collator,
        **loader_kwargs
    )


def one_loader(loaders: Dict[str, DataLoader]) -> DataLoader:
    if len(loaders) != 1:
        raise ValueError(f"Expected exactly one loader, but got {len(loaders)}.")
    return next(iter(loaders.values()))


def randidx(high: int) -> int:
    """Returns a random integer in the range [0, high)."""
    return int(torch.randint(high, ()).item())
