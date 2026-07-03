
from __future__ import annotations
import torch
from torch.utils.data import Dataset
import datasets
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable, TYPE_CHECKING

from utils.common import IGNORE_INDEX

if TYPE_CHECKING:
    from utils.config import TrackingConfig


class BaseDataset(Dataset):
    tok_fn: Callable[..., Dict[str, Any]]
    tok_kwargs: Dict[str, Any]
    data: datasets.Dataset

    def __init__(self, hf_args: TrackingConfig):
        super().__init__()
        # raw data
        self.raw_data = load_hf_dataset(**hf_args)

    def prepare_data(
        self,
        input_columns: List[str],
        desc: Optional[str] = None,
        **kwargs
    ) -> datasets.Dataset:
        return self.raw_data.map(
            self.tok_fn,
            input_columns=input_columns,
            with_indices=True,
            fn_kwargs=self.tok_kwargs,
            remove_columns=self.raw_data.column_names,
            desc=desc or f"Pre-tokenizing {self.__class__.__name__}",
            **kwargs
        )

    @staticmethod
    def process_sample(sample: Dict[str, Any]):
        input_ids, labels, index = sample["input_ids"], sample["labels"], sample["index"]
        if len(input_ids) != len(labels):
            raise ValueError(f"Length mismatch: input_ids has length {len(input_ids)}, labels has length {len(labels)}")

        if len(input_ids) == 1:
            return {"input_ids": input_ids[0], "labels": labels[0], "index": index}
        else:
            return {
                i: {"input_ids": input_ids[i], "labels": labels[i], "index": index}
                for i in range(len(input_ids))
            }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.process_sample(self.data[idx])


def load_hf_dataset(path: str, add_index: bool = False, **kwargs) -> datasets.Dataset:
    dataset = datasets.load_dataset(path, **kwargs)
    if add_index:
        dataset = dataset.add_column("index", np.arange(len(dataset)))
    return dataset


def prepare_chat_sample_context(
    template_args: TrackingConfig,
    question_key: str = "question",
    answer_key: str = "answer",
    few_shot_dataset_hf_args: Optional[TrackingConfig] = None
):
    # few-shot data
    fs_question_data, fs_answer_data = [], []
    if few_shot_dataset_hf_args is not None:
        _fs_data = load_hf_dataset(**few_shot_dataset_hf_args)
        fs_question_data = _fs_data[question_key]
        fs_answer_data = _fs_data[answer_key]

    if template_args.get("apply_chat_template", False, allow_none=True):
        # use chat template to format the prompt and response
        chat: List[Dict[str, str]] = []
        # system prompt
        system_prompt = template_args.get("system_prompt", None, allow_none=True)
        if system_prompt:
            chat.append({"role": "system", "content": system_prompt})
        # few-shot examples
        for q, a in zip(fs_question_data, fs_answer_data):
            chat.append({"role": "user", "content": q})
            chat.append({"role": "assistant", "content": a})
        return chat
    else:
        # use user/assistant tags to format the prompt and response
        wrapped_prompt: str = ""
        # system prompt with special tokens
        system_prompt_with_special_tokens = template_args.get(
            "system_prompt_with_special_tokens", None, allow_none=True
        )
        if system_prompt_with_special_tokens:
            wrapped_prompt += str(system_prompt_with_special_tokens)
        # few-shot examples
        for q, a in zip(fs_question_data, fs_answer_data):
            wrapped_prompt += str(
                template_args["user_start_tag"]
                + q
                + template_args["user_end_tag"]
                + template_args["asst_start_tag"]
                + a
                + template_args["asst_end_tag"]
            )
        return wrapped_prompt


def tok_chat_sample(
    q: str,
    a: Union[str, List[str]],
    idx: int,
    /, *,
    tokenizer: Any,
    sample_context: Union[str, List[Dict[str, str]]],
    template_args: TrackingConfig,
    max_length: int,
    predict_with_generate: bool = False,
):
    try:
        # multi-answer support
        if isinstance(a, list):
            if predict_with_generate:
                raise TypeError("`predict_with_generate=True` does not support multiple answers per question, please provide a single answer string.")
            multi_a = a
        elif isinstance(a, str):
            multi_a = [a]
        else:
            raise TypeError(f"Expected `str` or `list of str` for answer, but got {type(a)}.")

        if template_args.get("apply_chat_template", False):
            # use chat template
            assert isinstance(sample_context, list)
            prompt = sample_context + [{"role": "user", "content": q}]
            multi_chat = [prompt + [{"role": "assistant", "content": ans}] for ans in multi_a]

            date_str = template_args.get("date_string", None, allow_none=True)
            date_info = {"date_string": date_str} if date_str is not None else {}
            prompt_ids = tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
                max_length=max_length,
                truncation=True,
                **date_info
            )
            multi_chat_ids = tokenizer.apply_chat_template(
                multi_chat,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=False,
                max_length=max_length,
                truncation=True,
                **date_info
            )
        else:
            # use user/assistant tags
            assert isinstance(sample_context, str)
            prompt = (
                sample_context
                + template_args["user_start_tag"]
                + q
                + template_args["user_end_tag"]
                + template_args["asst_start_tag"]
            )
            multi_chat = [prompt + ans for ans in multi_a]

            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True
            )["input_ids"]
            multi_chat_ids = tokenizer(
                multi_chat,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True
            )["input_ids"]

        multi_input_ids, multi_labels = [], []
        for chat_ids in multi_chat_ids:
            # ensure the chat ends with the EOS token
            if chat_ids[-1] != tokenizer.eos_token_id:
                chat_ids.append(tokenizer.eos_token_id)
            # input idx
            if predict_with_generate:
                # In generation mode, we only want the input_ids to be the prompt.
                multi_input_ids.append(prompt_ids)
            else:
                # In training mode, we want the input_ids to be the full chat (prompt + answer).
                multi_input_ids.append(chat_ids)
            # labels
            labels = [IGNORE_INDEX] * len(prompt_ids) + chat_ids[len(prompt_ids):]
            multi_labels.append(labels)
        return {"input_ids": multi_input_ids, "labels": multi_labels, "index": idx}

    except Exception as e:
        raise RuntimeError(
            f"Error processing sample with index {idx} and sample_context {sample_context}"
        ) from e


def tok_text_sample(
    prompt: str,
    text: str,
    idx: int,
    /, *,
    tokenizer: Any,
    text_max_length: int,
    predict_with_generate: bool = False,
    insert_space: bool = False,
):
    _insert_str = " " if insert_space and prompt and text else ""

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    full_seq_ids = tokenizer(
        prompt + _insert_str + text,
        add_special_tokens=True
    )["input_ids"]

    prompt_len = len(prompt_ids)
    # manual truncation
    full_seq_ids = full_seq_ids[: prompt_len + text_max_length]

    # input idx
    input_ids = prompt_ids if predict_with_generate else full_seq_ids
    # labels
    len_matched = prompt_len
    if len_matched == 0:  # never give loss on index 0, when prefix is empty
        len_matched = 1
    labels = [IGNORE_INDEX] * len_matched + full_seq_ids[len_matched:]

    return {"input_ids": [input_ids], "labels": [labels], "index": idx}


def collect_text_sample(
    text_tok: List[int],
    idx: int,
    /, *,
    max_length: int,
):
    # manual truncation
    text_tok = text_tok[:max_length]
    # labels
    labels = [IGNORE_INDEX] + text_tok[1:]

    return {"input_ids": [text_tok], "labels": [labels], "index": idx}
