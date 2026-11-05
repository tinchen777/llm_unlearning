
from __future__ import annotations
import copy
import logging
from pprint import pprint
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING

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
        dataset_cfg: TrackingConfig,
        collator_cfg: Optional[TrackingConfig] = None,
        split: str = "???",
        **kwargs
    ):
        dataset, name = get_dataset(dataset_cfg, **kwargs)
        return cls(
            name=name,
            split=split,
            dataset=dataset,
            collator=get_collator(collator_cfg, **kwargs)
        )

    def __init__(self, name: str, dataset: Dataset, collator: Optional[Any] = None, split: str = "???"):
        self.name = name
        self.split = split
        self.dataset = dataset
        self.collator = collator
        self.set_loader(batch_size=1)

    def set_loader(
        self,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
        max_samples: Optional[int] = None,
        sampling_seed: int = 42
    ):
        """Set the loader configuration for the data."""
        self.default_args: Dict[str, Any] = dict(
            collator=self.collator,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            max_samples=max_samples,
            sampling_seed=sampling_seed
        )
        return self

    def with_generation_mode(self, generation_mode: bool) -> "Data":
        """The same samples, tokenised as the prompt only (`generation_mode=True`) or as prompt + answer (`False`); same collator and
        loader settings. Only for the QA datasets (they keep the tokenisation settings in `tok_kwargs`)."""
        tok_kwargs = getattr(self.dataset, "tok_kwargs", None)
        if tok_kwargs is None or "generation_mode" not in tok_kwargs:
            raise TypeError(f"{type(self.dataset).__name__} has no `generation_mode`: cannot change what it tokenises.")
        dataset = copy.copy(self.dataset)  # shares the raw data; the tokenised data is rebuilt on first use
        dataset.tok_kwargs = {**tok_kwargs, "generation_mode": generation_mode}
        dataset._data = None
        other = Data(self.name, dataset, self.collator, self.split)
        other.default_args = dict(self.default_args)
        return other

    def loader(self, **loader_kwargs: Any):
        """Get a data loader for the dataset."""
        args = self.default_args | loader_kwargs
        return make_subloader(self.dataset, **args)

    def __repr__(self) -> str:
        return f"Data<{self.name}>(dataset={self.dataset}, collator={self.collator})"


class SplitData:
    def __init__(self, data_cfg: TrackingConfig, **kwargs):
        self.split_data = {
            split_name: Data.from_cfg(
                split=split_name,
                dataset_cfg=split_cfg["dataset"],
                collator_cfg=split_cfg.get("collator", None),
                **kwargs
            )
            for split_name, split_cfg in data_cfg.items()
        }
        if not self.split_data:
            raise ValueError("No split data found.")
        else:
            logger.info(f"Loaded [{len(self)}] split data: {list(self.split_data)}")
            pprint(self.split_data)

    def set_split_loader(
        self,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
        max_samples: Optional[int] = None,
        sampling_seed: int = 42
    ):
        """Set the loader configuration for all splits."""
        for split_data in self.split_data.values():
            split_data.set_loader(
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                max_samples=max_samples,
                sampling_seed=sampling_seed
            )
        return self

    @property
    def splits(self):
        """Get the names of all splits."""
        return list(self.split_data)

    def __iter__(self):
        return iter(self.split_data)

    def __len__(self):
        return len(self.split_data)

    def values(self):
        """The `Data` of every split."""
        return self.split_data.values()

    def items(self):
        """`(split_name, Data)` pairs."""
        return self.split_data.items()

    def __getitem__(self, split: str):
        return self.split_data[split]

# === Dataset ===


def get_dataset(dataset_cfg: TrackingConfig, **kwargs):
    """
    Returns
    -------
        (_Dataset_, _str_)
    """
    def _load_dataset(dataset_cfg: TrackingConfig) -> Tuple[Dataset, str]:
        """Load a single dataset based on its configuration and return it along with its name."""
        name = dataset_cfg.get("name", "???")
        try:
            dataset_cls = DATASET_REGISTRY[dataset_cfg["handler"]]
            return dataset_cls(**dataset_cfg.get("args", {}, check_none=True), **kwargs), name
        except Exception as e:
            raise RuntimeError(f"Error loading dataset `{name}` in `@{dataset_cfg.loc_choices}` with {dataset_cfg}") from e

    if "handler" in dataset_cfg:
        # single dataset case
        return _load_dataset(dataset_cfg)

    # combined datasets case
    combine_cfg = dataset_cfg.pop("COMBINE", None)
    if combine_cfg is None:
        raise RuntimeError(f"Missing COMBINE configuration in `@{dataset_cfg.loc_choices}` with {dataset_cfg}")

    datasets = {
        dataset_name: _load_dataset(item_cfg)[0]
        for dataset_name, item_cfg in dataset_cfg.items()
    }
    combine_cls = COMBINED_DATASET_REGISTRY[combine_cfg["handler"]]
    combine_dataset = combine_cls(
        **datasets,
        **combine_cfg.get("args", {}, check_none=True)
    )
    return combine_dataset, combine_cfg.get("name", combine_cls.__name__)

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
