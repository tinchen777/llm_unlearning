
from __future__ import annotations
import logging
from pprint import pprint
from typing import Dict, Any, Optional, TYPE_CHECKING

from .datasets.qa import QADataset, QAwithIdkDataset, QAwithAlternateDataset
from .datasets.pretraining import PretrainingDataset, CompletionDataset
from .collators.nested_data import DataCollatorForNestedData
from .combined_dataset import BalancedUnlearnDataset
from .utils import make_subloader

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


class Data:
    @classmethod
    def from_cfg(
        cls,
        datasets_cfg: TrackingConfig,
        collator_cfg: Optional[TrackingConfig] = None,
        **kwargs
    ):
        return cls(
            datasets=get_datasets(datasets_cfg, **kwargs),
            collator=get_collator(collator_cfg, **kwargs)
        )

    def __init__(self, datasets: Dict[str, Dataset], collator: Optional[Any] = None):
        self.datasets = datasets
        self.collator = collator

    def set_loader(
        self,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
        max_samples: Optional[int] = None,
        seed: int = 42
    ):
        """Set the loader configuration for the data."""
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.max_samples = max_samples
        self.seed = seed
        return self

    def one_loader(self, **loader_kwargs):
        """Get a data loader for the single dataset."""
        return make_subloader(
            self.one_dataset,
            batch_size=self.batch_size,
            collator=self.collator,
            max_samples=self.max_samples,
            seed=self.seed,
            **loader_kwargs
        )

    def loaders(self, **loader_kwargs):
        """Get data loaders for all datasets."""
        return {
            name: make_subloader(
                dataset,
                batch_size=self.batch_size,
                collator=self.collator,
                max_samples=self.max_samples,
                seed=self.seed,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                **loader_kwargs
            )
            for name, dataset in self.datasets.items()
        }

    @property
    def names(self):
        """Get the names of all datasets."""
        return list(self.datasets)

    @property
    def one_dataset(self):
        """Get the single dataset, ensuring there is exactly one."""
        if len(self.datasets) != 1:
            raise ValueError(f"Expected exactly one dataset for `Data`, but got {len(self.datasets)}.")
        return next(iter(self.datasets.values()))


class SplitData:
    def __init__(self, data_cfg: TrackingConfig, **kwargs):
        self.split_data = {
            split_name: Data.from_cfg(
                datasets_cfg=split_cfg["datasets"],
                collator_cfg=split_cfg.get("collator", None),
                **kwargs
            )
            for split_name, split_cfg in data_cfg.items()
        }
        logger.info(f"Loaded split data: {list(self.split_data)}")
        pprint(self.split_data)

    def set_split_loader(
        self,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
        max_samples: Optional[int] = None,
        seed: int = 42
    ):
        """Set the loader configuration for all splits."""
        for split_data in self.split_data.values():
            split_data.set_loader(
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                max_samples=max_samples,
                seed=seed
            )
        return self

    def __iter__(self):
        return iter(self.split_data)

    def __getitem__(self, split: str):
        return self.split_data[split]

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

    def _load_dataset(dataset_cfg: TrackingConfig) -> Dataset:
        """Load a single dataset based on its configuration."""
        dataset_cls = DATASET_REGISTRY[dataset_cfg["handler"]]
        return dataset_cls(**dataset_cfg.get("args", {}, check_none=True), **kwargs)

    # Load the datasets
    for dataset_name, item_cfg in dataset_cfg_dict.items():
        try:
            datasets[dataset_name] = _load_dataset(item_cfg)
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
