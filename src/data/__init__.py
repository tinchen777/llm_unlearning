
from __future__ import annotations
import logging
import json
from pprint import pprint
import torch
from torch.utils.data import DataLoader, Subset
from typing import Dict, Any, Optional, Sequence, Tuple, TYPE_CHECKING

from .datasets.qa import QADataset, QAwithIdkDataset, QAwithAlternateDataset
from .datasets.pretraining import PretrainingDataset, CompletionDataset
from .collators.nested_data import DataCollatorForNestedData
from .combined_dataset import BalancedUnlearnDataset

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from utils.config import TrackingConfig

logger = logging.getLogger("data")

DATASET_REGISTRY: Dict[str, Any] = {}
COMBINED_DATASET_REGISTRY: Dict[str, Tuple[Any, Sequence[str], Sequence[str]]] = {}
COLLATOR_REGISTRY: Dict[str, Any] = {}


def _register_dataset(dataset_cls):
    DATASET_REGISTRY[dataset_cls.__name__] = dataset_cls


def _register_combined_dataset(
    combined_cls,
    combined_keys: Sequence[str],
    other_keys: Sequence[str] = ()
):
    COMBINED_DATASET_REGISTRY[combined_cls.__name__] = (combined_cls, combined_keys, other_keys)


def _register_collator(collator_cls):
    COLLATOR_REGISTRY[collator_cls.__name__] = collator_cls

# === Data Loaders ===

def get_split_loaders(
    data_cfg: TrackingConfig,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    **data_kwargs
) -> Dict[str, Dict[str, DataLoader]]:
    split_data = get_split_data(data_cfg, **data_kwargs)
    return {
        split_name: get_loaders(
            datasets,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collator=collator
        )
        for split_name, (datasets, collator) in split_data.items()
    }


def get_loaders(
    datasets: Dict[str, Dataset],
    batch_size: int,
    collator: Optional[Any] = None,
    shuffle: bool = False,
    num_workers: int = 0,
    **loader_kwargs
) -> Dict[str, DataLoader]:
    return {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collator,
            **loader_kwargs
        )
        for name, dataset in datasets.items()
    }


def fixed_subset(dataset: Dataset, max_samples: Optional[int] = None, seed: int = 0) -> Dataset:
    """A FIXED random subset of `max_samples` samples (sorted indices; the same for every call / model given `seed`),
    or the dataset itself when `max_samples` is None or not smaller than the dataset."""
    if max_samples is None or int(max_samples) >= len(dataset):  # type: ignore
        return dataset
    idx = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:int(max_samples)]  # type: ignore
    return Subset(dataset, sorted(idx.tolist()))


def split_loader(
    split_data: Dict[str, Tuple[Dict[str, Dataset], Optional[Any]]],
    split: str,
    batch_size: int,
    max_samples: Optional[int] = None,
    seed: int = 0,
) -> DataLoader:
    """Unshuffled loader of the single dataset of `split` (output of `get_split_data`), on a fixed subset."""
    datasets, collator = split_data[split]
    dataset = fixed_subset(one_dataset(datasets), max_samples, seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)


def one_loader(loaders: Dict[str, DataLoader]) -> DataLoader:
    if len(loaders) != 1:
        raise ValueError(f"Expected exactly one loader, but got {len(loaders)}.")
    return next(iter(loaders.values()))


def get_split_data(
    data_cfg: TrackingConfig,
    **kwargs
) -> Dict[str, Tuple[Dict[str, Dataset], Optional[Any]]]:
    split_data = {
        split_name: (
            get_datasets(split_cfg["datasets"], **kwargs),
            get_collator(split_cfg.get("collator", None), **kwargs)
        )
        for split_name, split_cfg in data_cfg.items()
    }
    logger.info(f"Loaded split data: {list(split_data)}")
    pprint(split_data)

    return split_data

# === Datasets ===

def get_datasets(datasets_cfg: TrackingConfig, **kwargs):
    # combine args
    combine_cls, combine_keys, other_keys = COMBINED_DATASET_REGISTRY.get(
        datasets_cfg.pop("handler", None), (None, (), ())
    )
    other_args = {key: datasets_cfg.pop(key) for key in other_keys} if other_keys else {}

    datasets: Dict[str, Dataset] = {}
    for dataset_name, item_cfg in datasets_cfg.items():
        # get dataset configuration
        dataset_cfg = item_cfg if "handler" in item_cfg else next(iter(item_cfg.values()))
        # Load the dataset
        try:
            datasets[dataset_name] = _load_dataset(dataset_cfg, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Error loading dataset `{dataset_name}` in `@{dataset_cfg.loc_choices}` with {dataset_cfg}") from e
    # Combine datasets if necessary
    if combine_cls is not None and combine_keys:
        datasets[combine_cls.__name__] = combine_cls(
            **{key: datasets.pop(key) for key in combine_keys},
            **other_args
        )

    return datasets


def _load_dataset(dataset_cfg: TrackingConfig, **kwargs) -> Dataset:
    dataset_cls = DATASET_REGISTRY[dataset_cfg["handler"]]
    return dataset_cls(**dataset_cfg.get("args", {}, check_none=True), **kwargs)


def one_dataset(datasets: Dict[str, Dataset]) -> Dataset:
    if len(datasets) != 1:
        raise ValueError(f"Expected exactly one dataset for `dataloader`, but got {len(datasets)}.")
    return next(iter(datasets.values()))

# === Collator ===

def get_collator(collator_cfg: Optional[TrackingConfig], **kwargs):
    if collator_cfg is None:
        return None
    try:
        collator_cls = COLLATOR_REGISTRY[collator_cfg["handler"]]
        return collator_cls(**collator_cfg.get("args", {}, check_none=True), **kwargs)
    except Exception as e:
        raise RuntimeError(f"Error loading collator in `@{collator_cfg.loc_choices}` with {collator_cfg}") from e


# Register dataset
_register_dataset(QADataset)
_register_dataset(QAwithIdkDataset)
_register_dataset(PretrainingDataset)
_register_dataset(CompletionDataset)
_register_dataset(QAwithAlternateDataset)

_register_combined_dataset(BalancedUnlearnDataset, ("forget", "retain"), ("anchor",))

# Register collator
_register_collator(DataCollatorForNestedData)
