
from __future__ import annotations
from datasets import Dataset
from typing import Any, Sequence, TYPE_CHECKING

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
        **kwargs
    ):
        super().__init__(hf_args)
        # pre-tokenize the dataset for efficiency
        self.tok_fn = tok_text_sample
        self.tok_kwargs = dict(
            tokenizer=tokenizer,
            text_max_length=max_length,
            predict_with_generate=predict_with_generate,
            insert_space=insert_space
        )
        self.data = self.prepare_data(
            input_columns=[prefix_key, text_key],
            num_proc=None,
            load_from_cache_file=True,
            desc=f"Pre-tokenizing {self.__class__.__name__}"
        )


class PretrainingDataset(BaseDataset):
    def __init__(
        self,
        hf_args: TrackingConfig,
        tokenizer: Any,
        text_key: str = "text",
        max_length: int = 2048,
        **kwargs
    ):
        super().__init__(hf_args)
        # rebuild raw data
        text_tok_seq = self._chunk_and_tok_text(
            self.raw_data[text_key],
            tokenizer=tokenizer,
            max_length=max_length
        )
        self.raw_data = Dataset.from_dict({text_key: text_tok_seq})
        # pre-tokenize the dataset for efficiency
        self.tok_fn = collect_text_sample
        self.tok_kwargs = dict(
            max_length=max_length
        )
        self.data = self.prepare_data(
            input_columns=[text_key],
            num_proc=None,
            load_from_cache_file=True,
            desc=f"Pre-tokenizing {self.__class__.__name__}"
        )

    @staticmethod
    def _chunk_and_tok_text(text_seq: Sequence[str], tokenizer: Any, max_length: int) -> Sequence[Sequence[int]]:
        token_seq = tokenizer(
            "\n\n".join(text_seq),
            add_special_tokens=False
        )["input_ids"]

        num_chunks = len(token_seq) // max_length + 1
        return [token_seq[i * max_length : (i + 1) * max_length] for i in range(num_chunks)]
