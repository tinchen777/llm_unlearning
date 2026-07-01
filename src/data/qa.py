
from __future__ import annotations
import torch
from torch.utils.data import Dataset
from typing import Any, Optional, Dict, List, Union, TYPE_CHECKING

from .utils import load_hf_dataset, preprocess_chat_instance, prepare_sample_context, tok_chat_batch

if TYPE_CHECKING:
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
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_args = template_args
        self.question_key = question_key
        self.answer_key = answer_key
        self.predict_with_generate = predict_with_generate
        # data
        self.data = load_hf_dataset(**hf_args, add_index=True)
        # few-shot data
        self.fs_data: Optional[Dict[str, List[str]]] = None
        if few_shot_dataset_hf_args is not None:
            raw_data = load_hf_dataset(**few_shot_dataset_hf_args)
            self.fs_data = {}
            self.fs_data[question_key] = raw_data[question_key]
            self.fs_data[answer_key] = raw_data[answer_key]
        
        # # prepare context for each sample, e.g., few-shot examples, etc.
        # sample_context = prepare_sample_context(
        #     template_args,
        #     question_key=question_key,
        #     answer_key=answer_key, few_shot_dataset_hf_args=few_shot_dataset_hf_args
        # )
        # # pre-tokenize the dataset for efficiency
        # self.data = self.data.map(
        #     tok_chat_batch,
        #     input_columns=[question_key, answer_key],
        #     with_indices=True,
        #     batched=True,
        #     batch_size=1000,
        #     fn_kwargs=dict(
        #         tokenizer=self.tokenizer,
        #         template_args=self.template_args,
        #         sample_context=sample_context,
        #         max_length=self.max_length,
        #         predict_with_generate=self.predict_with_generate
        #     ),
        #     num_proc=1,
        #     remove_columns=self.data.column_names,
        #     load_from_cache_file=True,
        #     desc=f"Pre-tokenizing {self.__class__.__name__}"
        # )

    def _process_sample(self, question: str, answer: str, index: int = -1) -> Dict[str, Union[int, torch.Tensor]]:
        if self.fs_data is None:
            prompt_msgs, response_msgs = [question], [answer]
        else:
            prompt_msgs = self.fs_data[self.question_key] + [question]
            response_msgs = self.fs_data[self.answer_key] + [answer]
        tokenized_data = preprocess_chat_instance(
            self.tokenizer,
            self.template_args,
            prompt_msgs,
            response_msgs,
            self.max_length,
            self.predict_with_generate,
        )
        tokenized_data["index"] = index
        return tokenized_data

    def __len__(self):
        return len(self.data)
    
    # def __getitem__(self, idx: int):
        
        
        
        
    #     question = self.data[idx][self.question_key]
    #     answer = self.data[idx][self.answer_key]
    #     index = self.data[idx]["index"]
    #     if isinstance(answer, str):
    #         return self._process_sample(question=question, answer=answer, index=index)
    #     elif isinstance(answer, list):
    #         item: Dict[int, Dict[str, Union[int, torch.Tensor]]] = {}
    #         for i, ans in enumerate(answer):
    #             sample_item = self._process_sample(
    #                 question=question, answer=ans, index=index
    #             )
    #             item[i] = sample_item
    #         return item
    #     else:
    #         raise NotImplementedError(f"`answer` format not found, got {type(answer)}.")

    def __getitem__(self, idx: int):
        question = self.data[idx][self.question_key]
        answer = self.data[idx][self.answer_key]
        index = self.data[idx]["index"]
        if isinstance(answer, str):
            return self._process_sample(question=question, answer=answer, index=index)
        elif isinstance(answer, list):
            item: Dict[int, Dict[str, Union[int, torch.Tensor]]] = {}
            for i, ans in enumerate(answer):
                sample_item = self._process_sample(
                    question=question, answer=ans, index=index
                )
                item[i] = sample_item
            return item
        else:
            raise NotImplementedError(f"`answer` format not found, got {type(answer)}.")


class QAwithIdkDataset(QADataset):
    def __init__(self, idk_path: str, return_original: bool = True, **kwargs):
        self.idk_path = idk_path
        self.return_original = return_original
        self.idk_responses = open(self.idk_path, "r").readlines()
        super().__init__(**kwargs)

    def item_with_idk(self, question: str):
        rand_pos = torch.randint(0, len(self.idk_responses), (1,)).item()
        idk_response = self.idk_responses[int(rand_pos)].strip()
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
