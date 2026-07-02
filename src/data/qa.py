
from __future__ import annotations
from torch.utils.data import Dataset
from typing import Any, Optional, Dict, Union, TYPE_CHECKING

from .utils import (
    load_hf_dataset,
    prepare_sample_context,
    tok_chat_sample,
    randidx
)

if TYPE_CHECKING:
    import torch
    from utils.config import TrackingConfig


class QADataset(Dataset):
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
        super().__init__()
        # data
        self.data = load_hf_dataset(**hf_args)
        # prepare context for each sample, e.g., few-shot examples, etc.
        sample_context = prepare_sample_context(
            template_args,
            question_key=question_key,
            answer_key=answer_key, few_shot_dataset_hf_args=few_shot_dataset_hf_args
        )
        # pre-tokenize the dataset for efficiency
        self.map_kwargs = dict(
            tokenizer=tokenizer,
            template_args=template_args,
            sample_context=sample_context,
            max_length=max_length,
            predict_with_generate=predict_with_generate
        )
        self.data = self.data.map(
            tok_chat_sample,
            input_columns=[question_key, answer_key],
            with_indices=True,
            fn_kwargs=self.map_kwargs,
            num_proc=4,
            remove_columns=self.data.column_names,
            load_from_cache_file=True,
            desc=f"Pre-tokenizing {self.__class__.__name__}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        data = self.data[idx]
        input_ids, labels, index = data["input_ids"], data["labels"], data["index"]
        if len(input_ids) == 1:
            return {"input_ids": torch.tensor(input_ids[0]), "labels": torch.tensor(labels[0]), "index": index}
        else:
            item: Dict[int, Dict[str, Union[int, torch.Tensor]]] = {}
            for i in range(len(input_ids)):
                item[i] = {"input_ids": torch.tensor(input_ids[i]), "labels": torch.tensor(labels[i]), "index": index}
            return item


class QAwithIdkDataset(QADataset):
    def __init__(self, idk_path: str, return_original: bool = True, **kwargs):
        self.return_original = return_original
        self.idk_responses = open(idk_path, "r").readlines()
        super().__init__(**kwargs)

    def item_with_idk(self, question: str):
        idk_response = self.idk_responses[randidx(len(self.idk_responses))].strip()
        idk_item = self._process_sample(question=question, answer=idk_response)
        return idk_item

    def __getitem__(self, idx: int):
        
        
        
        question = self.data[idx][self.question_key]
        idk_item = self.item_with_idk(question)
        if self.return_original:
            return {"original": super().__getitem__(idx), "alternate": idk_item}
        else:
            return idk_item


class QAwithAlternateDataset(QADataset):
    def __init__(self, alternate_key: str, return_original: bool = True, **kwargs):
        self.alternate_key = alternate_key
        self.return_original = return_original
        super().__init__(**kwargs)

    def __getitem__(self, idx: int):
        alt_item = self._process_sample(
            question=self.data[idx][self.question_key],
            answer=self.data[idx][self.alternate_key]
        )
        if self.return_original:
            return {"original": super().__getitem__(idx), "alternate": alt_item}
        else:
            return alt_item
