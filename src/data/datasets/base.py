
from __future__ import annotations
from torch.utils.data import Dataset
from typing import List, Sequence, Dict, Any, Optional, Callable, TYPE_CHECKING

from .utils import load_hf_dataset, get_map_kwargs

if TYPE_CHECKING:
    from datasets import Dataset as HFDataset
    from utils.config import TrackingConfig


class BaseDataset(Dataset):
    tok_fn: Callable[..., Dict[str, Any]]
    tok_kwargs: Dict[str, Any]
    _data: Optional[HFDataset] = None

    def __init__(
        self,
        hf_args: TrackingConfig,
        map_args: Optional[TrackingConfig],
        extra_keys: Optional[Sequence[str]] = None
    ):
        super().__init__()
        self.retained_columns = set(extra_keys) if extra_keys else set()
        # raw data
        self.raw_data = load_hf_dataset(**hf_args)
        # map arguments & dataset_mb_str
        self._map_kwargs, raw_data_mb = get_map_kwargs(self.raw_data)
        if map_args is not None:
            self._map_kwargs.update(map_args.to_dict())
        self._raw_data_mb_str = f"{raw_data_mb:.2f} MB" if raw_data_mb is not None else "- MB"

    def map_raw_data(
        self,
        input_columns: List[str],
        name: str = ""
    ) -> HFDataset:
        remove_columns = set(self.raw_data.column_names)

        return self.raw_data.map(
            self.tok_fn,
            input_columns=input_columns,
            with_indices=True,
            with_rank=False,
            batched=False,
            fn_kwargs=self.tok_kwargs,
            remove_columns=list(remove_columns - self.retained_columns),
            desc=f"Pre-tokenizing [{self.__class__.__name__} : {self._raw_data_mb_str}][{name}]",
            **self._map_kwargs
        )

    def prepare_data(self) -> HFDataset:
        raise NotImplementedError(f"Subclasses of BaseDataset must implement the `prepare_data` method.")

    @staticmethod
    def process_sample(sample: Dict[str, Any]):
        input_ids = sample.pop("input_ids")
        labels = sample.pop("labels")
        index = sample.pop("index")

        if len(input_ids) != len(labels):
            raise ValueError(f"Length mismatch: input_ids has length {len(input_ids)}, labels has length {len(labels)}")

        if len(input_ids) == 1:
            return {"input_ids": input_ids[0], "labels": labels[0], "index": index, **sample}
        else:
            return {
                i: {"input_ids": input_ids[i], "labels": labels[i], "index": index, **sample}
                for i in range(len(input_ids))
            }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.process_sample(self.data[idx])

    @property
    def data(self):
        if self._data is None:
            self._data = self.prepare_data()
        return self._data
