
from __future__ import annotations
from transformers import DataCollatorForSeq2Seq
from typing import Dict, Sequence, Optional, Any, Union, TYPE_CHECKING

from utils.common import IGNORE_INDEX

if TYPE_CHECKING:
    from transformers import BatchEncoding


class DataCollatorForNestedData:
    """
    Collate examples for nested data, handling both input sequences and associated metadata.
    """

    def __init__(
        self,
        tokenizer: Any,
        padding_side: str = "right",
        metadata_keys: Optional[Sequence[str]] = None,
        **kwargs
    ):
        self.padding_side = padding_side
        self.collator = DataCollatorForSeq2Seq(
            tokenizer,
            padding=True,
            label_pad_token_id=IGNORE_INDEX,
            return_tensors="pt"
        )
        self.metadata_keys = set(metadata_keys or [])

    def __call__(
        self,
        samples: Sequence[Dict[Any, Any]]
    ) -> Union[BatchEncoding, Dict[Any, Any]]:
        demo_sample = samples[0]
        if not isinstance(demo_sample, dict):
            raise ValueError(
                f"Expected samples to be a sequence of dicts, but got Sequence({type(demo_sample)})."
            )

        keys = list(demo_sample)

        if "input_ids" not in keys:
            return {k: self([x[k] for x in samples]) for k in keys}

        # Set padding side for tokenizer, for work==0 only
        self.collator.tokenizer.padding_side = self.padding_side
        # Extract metadata from samples before collating
        metadata = {
            k: [x.pop(k) for x in samples]
            for k in self.metadata_keys
            if k in keys
        }
        # Collate the remaining samples into a batch
        batch = self.collator(samples)
        # Merge the extracted metadata into the collated batch
        batch.update(metadata)

        return batch
