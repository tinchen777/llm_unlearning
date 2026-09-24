
from __future__ import annotations
import logging
from torch.utils.data import DataLoader
from typing import Dict, Any, Union, Optional, TYPE_CHECKING

from .datasets.qa import QADataset, QAwithIdkDataset, QAwithAlternateDataset
from .datasets.pretraining import PretrainingDataset, CompletionDataset
from .collators import DataCollatorForSupervisedDataset
from .unlearn import ForgetRetainDataset

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from utils.config import TrackingConfig

logger = logging.getLogger(__name__)

DATASET_REGISTRY: Dict[str, Any] = {}
COLLATOR_REGISTRY: Dict[str, Any] = {}


def _register_data(data_cls):
    DATASET_REGISTRY[data_cls.__name__] = data_cls


def _register_collator(collator_cls):
    COLLATOR_REGISTRY[collator_cls.__name__] = collator_cls


def get_dataloader(
    data_cfg: TrackingConfig,
    mode: str,
    batch_size: int,
    split: Optional[str] = None,
    shuffle: bool = False,
    num_workers: int = 0,
    collator_cfgs: Optional[TrackingConfig] = None,
    **kwargs
):
    """Build DataLoader(s) from a data config, independent of any Trainer.

    `get_data` returns `{split: Dataset | {name: Dataset}}`, so the dataset
    must be picked out of that mapping before wrapping (a DataLoader over the
    dict itself sees `len == #splits` and indexes it with ints -> KeyError).

    Returns:
        - `split` given: the loader of that split (a `{name: DataLoader}` dict
          if the split holds several datasets).
        - `split=None` with a single split: that split's loader(s).
        - `split=None` with several splits: `{split: loader(s)}`.
    """
    # get data
    data = get_data(data_cfg, mode=mode, **kwargs)
    # get collator
    if collator_cfgs is not None:
        collator = get_collators(collator_cfgs, **kwargs)
        if isinstance(collator, dict):
            logger.warning(f"Got multiple({len(collator)}) collators, using the first one for dataloader.")
            collator = next(iter(collator.values()))
    else:
        collator = None

    def _wrap(dataset: Union[Dataset, Dict[str, Dataset]]):
        if isinstance(dataset, dict):
            return {name: _wrap(ds) for name, ds in dataset.items()}
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collator
        )

    if split is not None:
        if split not in data:
            raise KeyError(f"Split `{split}` not found in data, available: {list(data)}.")
        return _wrap(data[split])
    if len(data) == 1:
        return _wrap(next(iter(data.values())))
    return {split_name: _wrap(split_data) for split_name, split_data in data.items()}


def get_data(data_cfg: TrackingConfig, mode: str, **kwargs):
    data: Dict[str, Union[Dataset, Dict[str, Dataset]]] = {}
    for split_name, split_cfg in data_cfg.items():
        if split_name != "anchor":
            data[split_name] = get_datasets(split_cfg, **kwargs)

    if mode == "train":
        return data
    elif mode == "unlearn":
        data["train"] = ForgetRetainDataset(
            forget=data["forget"],
            retain=data["retain"],
            anchor=data_cfg.get("anchor", "forget", check_none=True)
        )
        for split_name in [k for k in data if k not in ("train", "eval", "test")]:
            data.pop(split_name)
    return data


def get_datasets(dataset_cfgs: TrackingConfig, **kwargs):
    datasets: Dict[str, Dataset] = {}
    for dataset_name, dataset_cfg in dataset_cfgs.items():
        try:
            dataset = _load_single_dataset(dataset_cfg, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Error loading dataset `{dataset_name}` in `@{dataset_cfg.loc_choices}` with {dataset_cfg}") from e
        if len(dataset_cfgs) == 1:
            # if only one dataset, return it directly
            return dataset
        access_name = dataset_cfg.get("access_key", dataset_name, check_none=True)
        datasets[str(access_name)] = dataset
    return datasets


def _load_single_dataset(dataset_cfg: TrackingConfig, **kwargs) -> Dataset:
    dataset_cls = DATASET_REGISTRY[dataset_cfg["handler"]]
    return dataset_cls(**dataset_cfg.get("args", {}, check_none=True), **kwargs)


def get_collators(collator_cfgs: TrackingConfig, **kwargs):
    collators: Dict[str, Any] = {}
    for collator_name, collator_cfg in collator_cfgs.items():
        try:
            collator = _get_single_collator(collator_cfg, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Error loading collator `{collator_name}` in `@{collator_cfg.loc_choices}` with {collator_cfg}") from e
        if len(collator_cfgs) == 1:
            # if only one collator, return it directly
            return collator
        collators[collator_name] = collator
    return collators


def _get_single_collator(collator_cfg: TrackingConfig, **kwargs):
    collator_cls = COLLATOR_REGISTRY[collator_cfg["handler"]]
    return collator_cls(**collator_cfg.get("args", {}, check_none=True), **kwargs)


# Register datasets
_register_data(QADataset)
_register_data(QAwithIdkDataset)
_register_data(PretrainingDataset)
_register_data(CompletionDataset)
_register_data(QAwithAlternateDataset)

# Register collators
_register_collator(DataCollatorForSupervisedDataset)
