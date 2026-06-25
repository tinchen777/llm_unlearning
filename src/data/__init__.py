
from __future__ import annotations
from typing import Dict, Any, Union, TYPE_CHECKING

from .qa import QADataset, QAwithIdkDataset, QAwithAlternateDataset
from .collators import DataCollatorForSupervisedDataset
from .unlearn import ForgetRetainDataset
from .pretraining import PretrainingDataset, CompletionDataset

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from utils.config import TrackingConfig

DATASET_REGISTRY: Dict[str, Any] = {}
COLLATOR_REGISTRY: Dict[str, Any] = {}


def _register_data(data_cls):
    DATASET_REGISTRY[data_cls.__name__] = data_cls


def _register_collator(collator_cls):
    COLLATOR_REGISTRY[collator_cls.__name__] = collator_cls


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
            anchor=data_cfg.get("anchor", "forget")
        )
        for split_name in [k for k in data if k not in ("train", "eval", "test")]:
            data.pop(split_name)
    return data


def get_datasets(dataset_cfgs: TrackingConfig, **kwargs):
    dataset: Dict[str, Dataset] = {}
    for dataset_name, dataset_cfg in dataset_cfgs.items():
        access_name = dataset_cfg.get("access_key", dataset_name)
        try:
            dataset[str(access_name)] = _load_single_dataset(dataset_cfg, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Error loading dataset `{dataset_name}` in `@{dataset_cfg.loc_choices}` with {dataset_cfg}") from e
    if len(dataset) == 1:
        # return a single dataset
        return next(iter(dataset.values()))
    # return mapping to multiple datasets
    return dataset


def _load_single_dataset(dataset_cfg: TrackingConfig, **kwargs) -> Dataset:
    dataset_cls = DATASET_REGISTRY[dataset_cfg["handler"]]
    return dataset_cls(**dataset_cfg.get("args", {}), **kwargs)


def get_collators(collator_cfgs: TrackingConfig, **kwargs):
    collators = {}
    for collator_name, collator_cfg in collator_cfgs.items():
        try:
            collators[collator_name] = _get_single_collator(collator_cfg, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Error loading collator `{collator_name}` in `@{collator_cfg.loc_choices}` with {collator_cfg}") from e
    if len(collators) == 1:
        # return a single collator
        return next(iter(collators.values()))
    # return collators in a dict
    return collators


def _get_single_collator(collator_cfg: TrackingConfig, **kwargs):
    collator_cls = COLLATOR_REGISTRY[collator_cfg["handler"]]
    return collator_cls(**collator_cfg.get("args", {}), **kwargs)


# Register datasets
_register_data(QADataset)
_register_data(QAwithIdkDataset)
_register_data(PretrainingDataset)
_register_data(CompletionDataset)
_register_data(QAwithAlternateDataset)

# Register composite datasets used in unlearning
# groups: unlearn
_register_data(ForgetRetainDataset)

# Register collators
_register_collator(DataCollatorForSupervisedDataset)
