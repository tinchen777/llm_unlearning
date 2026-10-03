
from __future__ import annotations
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Sequence, Tuple, TYPE_CHECKING

from .datasets.qa import QADataset, QAwithIdkDataset, QAwithAlternateDataset
from .datasets.pretraining import PretrainingDataset, CompletionDataset
from .collators.nested_data import DataCollatorForNestedData
from .combined_dataset import BalancedUnlearnDataset

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from utils.config import TrackingConfig

DATASET_REGISTRY: Dict[str, Any] = {}
COMBINED_DATASET_REGISTRY: Dict[str, Any] = {}
COLLATOR_REGISTRY: Dict[str, Any] = {}


def _register_dataset(dataset_cls):
    DATASET_REGISTRY[dataset_cls.__name__] = dataset_cls


def _register_combined_dataset(combined_dataset_cls, combined_keys: Sequence[str]):
    COMBINED_DATASET_REGISTRY[combined_dataset_cls.__name__] = (combined_dataset_cls, combined_keys)


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


def one_loader(loaders: Dict[str, DataLoader]) -> DataLoader:
    if len(loaders) != 1:
        raise ValueError(f"Expected exactly one loader, but got {len(loaders)}.")
    return next(iter(loaders.values()))


def get_split_data(
    data_cfg: TrackingConfig,
    **kwargs
) -> Dict[str, Tuple[Dict[str, Dataset], Optional[Any]]]:
    return {
        split_name: (
            get_datasets(split_cfg["datasets"], **kwargs),
            get_collator(split_cfg.get("collator", None), **kwargs)
        )
        for split_name, split_cfg in data_cfg.items()
    }

# === Datasets ===

def get_datasets(datasets_cfg: TrackingConfig, **kwargs):
    # extract combine args
    combine_handler = datasets_cfg.pop("handler", None)
    combine_anchor = datasets_cfg.pop("anchor", None)

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
    if combine_handler is not None and combine_anchor is not None:
        combine_dataset_cls, combine_keys = COMBINED_DATASET_REGISTRY[combine_handler]
        datasets[combine_handler] = combine_dataset_cls(
            **{key: datasets.pop(key) for key in combine_keys},
            anchor=combine_anchor
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

_register_combined_dataset(BalancedUnlearnDataset, ("forget", "retain"))

# Register collator
_register_collator(DataCollatorForNestedData)
