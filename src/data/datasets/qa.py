
from __future__ import annotations
from typing import Any, Optional, TYPE_CHECKING

from .base import (
    BaseDataset,
    prepare_chat_sample_context,
    tok_chat_sample
)
from utils.common import randidx

if TYPE_CHECKING:
    from utils.config import TrackingConfig


class QADataset(BaseDataset):
    def __init__(
        self,
        hf_args: TrackingConfig,
        template_args: TrackingConfig,
        tokenizer: Any,
        question_key: str = "question",
        answer_key: str = "answer",
        few_shot_dataset_hf_args: Optional[TrackingConfig] = None,
        max_length: int = 512,
        predict_with_generate: bool = False,
        **kwargs
    ):
        super().__init__(hf_args)
        self.question_key = question_key
        # prepare context for each sample, e.g., few-shot examples, etc.
        sample_context = prepare_chat_sample_context(
            template_args,
            question_key=question_key,
            answer_key=answer_key, few_shot_dataset_hf_args=few_shot_dataset_hf_args
        )
        # pre-tokenize the dataset for efficiency
        self.tok_fn = tok_chat_sample
        self.tok_kwargs = dict(
            tokenizer=tokenizer,
            template_args=template_args,
            sample_context=sample_context,
            max_length=max_length,
            predict_with_generate=predict_with_generate
        )
        self.data = self.prepare_data(
            input_columns=[question_key, answer_key],
            num_proc=None,
            load_from_cache_file=True,
            desc=f"Pre-tokenizing {self.__class__.__name__}"
        )


class QAwithIdkDataset(QADataset):
    def __init__(self, idk_path: str, return_original: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.return_original = return_original
        self.idk_responses = open(idk_path, "r").readlines()

    def __getitem__(self, idx: int):
        alternate_sample = self.tok_fn(
            q=self.raw_data[idx][self.question_key],
            a=self.idk_responses[randidx(len(self.idk_responses))].strip(),
            **self.tok_kwargs
        )
        idk_item = self.process_sample(alternate_sample)
        if self.return_original:
            return {"original": super().__getitem__(idx), "alternate": idk_item}
        else:
            return idk_item


class QAwithAlternateDataset(QADataset):
    def __init__(self, alternate_key: str, return_original: bool = True, **kwargs):
        self.return_original = return_original
        super().__init__(**kwargs)
        # pre-tokenize the alternate dataset for efficiency
        self.alternate_data = self.prepare_data(
            input_columns=[self.question_key, alternate_key],
            num_proc=None,
            load_from_cache_file=True,
            desc=f"Pre-tokenizing {self.__class__.__name__}"
        )

    def __getitem__(self, idx: int):
        alt_item = self.process_sample(self.alternate_data[idx])
        if self.return_original:
            return {"original": super().__getitem__(idx), "alternate": alt_item}
        else:
            return alt_item
