
from __future__ import annotations
from typing import Any, Optional, TYPE_CHECKING

from .base import (
    BaseDataset,
    prepare_chat_sample_context,
    tok_chat_sample
)
from utils.common import randidx

if TYPE_CHECKING:
    from datasets import Dataset as HFDataset
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
        map_args: Optional[TrackingConfig] = None,
        **kwargs
    ):
        super().__init__(hf_args, map_args)
        self.question_key = question_key
        self.answer_key = answer_key
        # prepare context for each sample, e.g., few-shot examples, etc.
        sample_context = prepare_chat_sample_context(
            template_args,
            question_key=question_key,
            answer_key=answer_key, few_shot_dataset_hf_args=few_shot_dataset_hf_args
        )
        self.tok_fn = tok_chat_sample
        self.tok_kwargs = dict(
            tokenizer=tokenizer,
            template_args=template_args,
            sample_context=sample_context,
            max_length=max_length,
            predict_with_generate=predict_with_generate
        )

    def prepare_data(self):
        return self.map_raw_data(
            input_columns=[self.question_key, self.answer_key],
            name="QA data"
        )


class QAwithIdkDataset(QADataset):
    def __init__(self, idk_path: str, return_original: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.return_original = return_original
        self.idk_responses = open(idk_path, "r").readlines()

    def __getitem__(self, idx: int):
        alternate_sample = self.tok_fn(
            self.raw_data[idx][self.question_key],
            self.idk_responses[randidx(len(self.idk_responses))].strip(),
            idx,
            **self.tok_kwargs
        )
        idk_item = self.process_sample(alternate_sample)
        if self.return_original:
            return {"original": super().__getitem__(idx), "alternate": idk_item}
        else:
            return idk_item


class QAwithAlternateDataset(QADataset):
    _alt_data: Optional[HFDataset] = None

    def __init__(self, alternate_key: str, return_original: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.return_original = return_original
        self.alternate_key = alternate_key

    def prepare_alt_data(self):
        return self.map_raw_data(
            input_columns=[self.question_key, self.alternate_key],
            name=f"Q-{self.alternate_key} data"
        )

    def __getitem__(self, idx: int):
        alt_item = self.process_sample(self.alternate_data[idx])
        if self.return_original:
            return {"original": super().__getitem__(idx), "alternate": alt_item}
        else:
            return alt_item

    @property
    def alternate_data(self):
        if self._alt_data is None:
            self._alt_data = self.prepare_alt_data()
        return self._alt_data
