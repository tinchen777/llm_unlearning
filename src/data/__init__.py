
from __future__ import annotations
import logging
from pprint import pprint
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING

from .datasets.qa import QADataset, QAwithIdkDataset, QAwithAlternateDataset
from .datasets.pretraining import PretrainingDataset, CompletionDataset
from .collators.nested_data import DataCollatorForNestedData
from .combined_dataset import BalancedUnlearnDataset
from .utils import get_subloader

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from utils.config import TrackingConfig

logger = logging.getLogger("data")

DATASET_REGISTRY: Dict[str, Any] = {}
COMBINED_DATASET_REGISTRY: Dict[str, Any] = {}
COLLATOR_REGISTRY: Dict[str, Any] = {}


def _register_dataset(dataset_cls):
    DATASET_REGISTRY[dataset_cls.__name__] = dataset_cls


def _register_combined_dataset(combined_cls):
    COMBINED_DATASET_REGISTRY[combined_cls.__name__] = combined_cls


def _register_collator(collator_cls):
    COLLATOR_REGISTRY[collator_cls.__name__] = collator_cls

# === Data Loaders ===

def get_split_loaders(
    data_cfg: TrackingConfig,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
    seed: int = 42,
    **data_kwargs
) -> Dict[str, Dict[str, DataLoader]]:
    """
    Returns
    -------
        {
            `split_name`: {`loader_name`: `DataLoader`, ...},
            ...
        }
    """
    split_data = get_split_data(data_cfg, **data_kwargs)
    return {
        split_name: get_loaders(
            datasets,
            batch_size=batch_size,
            collator=collator,
            max_samples=max_samples,
            seed=seed,
            shuffle=shuffle,
            num_workers=num_workers
        )
        for split_name, (datasets, collator) in split_data.items()
    }


def get_loaders(
    datasets: Dict[str, Dataset],
    batch_size: int,
    collator: Optional[Any] = None,
    max_samples: Optional[int] = None,
    seed: int = 42,
    **loader_kwargs
) -> Dict[str, DataLoader]:
    """
    Returns
    -------
        {`loader_name`: `DataLoader`, ...}
    """
    return {
        name: get_subloader(
            dataset,
            batch_size=batch_size,
            collator=collator,
            max_samples=max_samples,
            seed=seed,
            **loader_kwargs
        )
        for name, dataset in datasets.items()
    }

# === Split Data ===

def get_split_data(
    data_cfg: TrackingConfig,
    **kwargs
) -> Dict[str, Tuple[Dict[str, Dataset], Optional[Any]]]:
    """
    Returns
    -------
        {
            `split_name`: ({`dataset_name`: `Dataset`, ...}, `collator`),
            ...
        }
    """
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
    """
    Returns
    -------
        {`dataset_name`: `Dataset`, ...}
    """
    # combine
    combine_cfg = datasets_cfg.pop("COMBINE", None)

    dataset_cfg_dict = {datasets_cfg["name"]: datasets_cfg} if "handler" in datasets_cfg else datasets_cfg
    datasets: Dict[str, Dataset] = {}
    # Load the datasets
    for dataset_name, item_cfg in dataset_cfg_dict.items():
        try:
            datasets[dataset_name] = _load_dataset(item_cfg, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Error loading dataset `{dataset_name}` in `@{item_cfg.loc_choices}` with {item_cfg}") from e
    # Combine datasets if necessary
    if combine_cfg is not None:
        combine_cls = COMBINED_DATASET_REGISTRY[combine_cfg["handler"]]
        datasets = {
            combine_cfg.get("name", combine_cls.__name__): combine_cls(
                **datasets,
                **combine_cfg.get("args", {}, check_none=True)
            )
        }

    return datasets


def _load_dataset(dataset_cfg: TrackingConfig, **kwargs) -> Dataset:
    dataset_cls = DATASET_REGISTRY[dataset_cfg["handler"]]
    return dataset_cls(**dataset_cfg.get("args", {}, check_none=True), **kwargs)

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

_register_combined_dataset(BalancedUnlearnDataset)

# Register collator
_register_collator(DataCollatorForNestedData)
