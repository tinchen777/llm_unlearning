
from __future__ import annotations
from transformers import DataCollatorForSeq2Seq
from typing import Dict, Sequence, Optional, Any, Union, TYPE_CHECKING

from utils.common import IGNORE_INDEX

if TYPE_CHECKING:
    from transformers import BatchEncoding


class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: Any,
        padding_side: str = "right",
        index: Optional[str] = None,
    ):
        tokenizer.padding_side = padding_side
        self.collator = DataCollatorForSeq2Seq(
            tokenizer,
            padding=True,
            label_pad_token_id=IGNORE_INDEX,
            return_tensors="pt"
        )
        self.index = index

    def __call__(self, samples: Sequence[Dict[Any, Any]]) -> Union[BatchEncoding, Dict[Any, Any]]:
        demo_sample = samples[0]
        if not isinstance(demo_sample, dict):
            raise ValueError(
                f"Expected samples to be a sequence of dicts, but got Sequence({type(demo_sample)})."
            )
        keys = list(demo_sample)
        if "input_ids" not in keys:
            return {k: self([x[k] for x in samples]) for k in keys}
        else:
            idxs = [x.pop(self.index) for x in samples] if self.index in keys else None
            batch = self.collator(samples)
            if idxs is not None:
                batch[self.index] = idxs
            return batch
