
from __future__ import annotations
from datasets import Dataset as HFDataset
from typing import Any, Sequence, Optional, TYPE_CHECKING

from .base import BaseDataset, tok_text_sample, collect_text_sample

if TYPE_CHECKING:
    from utils.config import TrackingConfig


class CompletionDataset(BaseDataset):
    def __init__(
        self,
        hf_args: TrackingConfig,
        tokenizer: Any,
        prefix_key: str = "prompt",
        text_key: str = "text",
        max_length: int = 2048,
        predict_with_generate: bool = False,
        insert_space: bool = False,
        map_args: Optional[TrackingConfig] = None,
        **kwargs
    ):
        super().__init__(hf_args, map_args)
        self.prefix_key = prefix_key
        self.text_key = text_key
        # pre-tokenize the dataset for efficiency
        self.tok_fn = tok_text_sample
        self.tok_kwargs = dict(
            tokenizer=tokenizer,
            text_max_length=max_length,
            predict_with_generate=predict_with_generate,
            insert_space=insert_space
        )

    def prepare_data(self):
        # ensure that the prefix column exists in the dataset, if not, add an empty column
        if self.prefix_key not in self.raw_data.column_names:
            self.raw_data = self.raw_data.add_column(
                name=self.prefix_key,
                column=[""] * len(self.raw_data)
            )
        return self.map_raw_data(
            input_columns=[self.prefix_key, self.text_key],
            name="prompt-text data"
        )


class PretrainingDataset(BaseDataset):
    def __init__(
        self,
        hf_args: TrackingConfig,
        tokenizer: Any,
        text_key: str = "text",
        max_length: int = 2048,
        map_args: Optional[TrackingConfig] = None,
        **kwargs
    ):
        super().__init__(hf_args, map_args)
        self.text_key = text_key
        # rebuild raw data
        text_tok_seq = self._chunk_and_tok_text(
            self.raw_data[text_key],
            tokenizer=tokenizer,
            max_length=max_length
        )
        self.raw_data = HFDataset.from_dict({text_key: text_tok_seq})
        # pre-tokenize the dataset for efficiency
        self.tok_fn = collect_text_sample
        self.tok_kwargs = dict(
            max_length=max_length
        )

    def prepare_data(self):
        return self.map_raw_data(
            input_columns=[self.text_key],
            name="text data"
        )

    @staticmethod
    def _chunk_and_tok_text(text_seq: Sequence[str], tokenizer: Any, max_length: int) -> Sequence[Sequence[int]]:
        token_seq = tokenizer(
            "\n\n".join(text_seq),
            add_special_tokens=False
        )["input_ids"]

        num_chunks = len(token_seq) // max_length + 1
        return [token_seq[i * max_length : (i + 1) * max_length] for i in range(num_chunks)]
