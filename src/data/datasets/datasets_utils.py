"""
Utility functions for `data.datasets` module.
"""
from __future__ import annotations
import datasets
import numpy as np
import os
from collections.abc import Iterable
from typing import List, Dict, Any, Optional, Union, TYPE_CHECKING

from ..utils import IGNORE_INDEX

if TYPE_CHECKING:
    from datasets import Dataset as HFDataset
    from utils.config import TrackingConfig


# Splits that are PARTS of an official TOFU split (the HF dataset has no config of that name): {name: (official name, HF split slice)}.
# The official splits are nested, the last authors of the 200 being the smallest set: forget10 = authors 181-200, forget05 = 191-200,
# forget01 = 199-200 (rows 0-19 = author 199, rows 20-39 = author 200), so ONE author, forget005 = author 200 = the LAST 20 rows of forget01
# (and of forget01_perturbed). There is no retain995 model: whatever needs the retain model of the split (forget quality, privleak) is
# not meaningful for it.
VIRTUAL_SPLITS: Dict[str, Any] = {
    "forget005": ("forget01", "train[20:]"),
    "forget005_perturbed": ("forget01_perturbed", "train[20:]"),
}


def resolve_virtual_split(name: Optional[str], split: Optional[str] = None):
    """`(name, split)` as the HF dataset knows them: a virtual split (`VIRTUAL_SPLITS`) becomes its official split sliced to its rows
    (`train[20:]`); anything else is returned as it is. A virtual split cannot be sliced again."""
    if name not in VIRTUAL_SPLITS:
        return name, split
    official, rows = VIRTUAL_SPLITS[name]
    if split not in (None, "train"):
        raise ValueError(f"`{name}` is already {rows} of `{official}`: its HF split must be `train`, got `{split}`.")
    return official, rows


def load_hf_dataset(path: str, add_index: bool = False, **kwargs) -> HFDataset:
    if "name" in kwargs:
        kwargs["name"], resolved_split = resolve_virtual_split(kwargs["name"], kwargs.get("split"))
        if resolved_split is not None:
            kwargs["split"] = resolved_split
    try:
        dataset = datasets.load_dataset(path, **kwargs)
        if add_index:
            dataset = dataset.add_column("index", np.arange(len(dataset)))
        return dataset
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from path: {path} with arguments: {kwargs}") from e


def filter_rows(dataset: HFDataset, row_filter: Optional[Dict[str, Any]]) -> HFDataset:
    """Keep the rows whose columns match `row_filter` ({column: value} or {column: [values]}; all columns must match). `None` / `{}`: all rows.
    Used to split one csv into parts, e.g. the neighbour data: `split: train` (reference of r_UV, retained tokens) / `split: heldout` (evaluation)."""
    if not row_filter:
        return dataset
    missing = [c for c in row_filter if c not in dataset.column_names]
    if missing:
        raise KeyError(f"row_filter {dict(row_filter)}: the dataset has no column(s) {missing}; columns: {dataset.column_names}.")
    # a Hydra list is an omegaconf ListConfig, not a `list`: any non-string iterable is a set of allowed values
    wanted = {c: (set(v) if (not isinstance(v, (str, bytes)) and isinstance(v, Iterable)) else {v}) for c, v in row_filter.items()}
    filtered = dataset.filter(lambda row: all(row[c] in values for c, values in wanted.items()))
    if len(filtered) == 0:
        raise ValueError(f"row_filter {dict(row_filter)} keeps no row of the dataset ({len(dataset)} rows).")
    return filtered


def get_map_kwargs(
    dataset: HFDataset,
    keep_in_memory_threshold_mb: int = 512,
):
    """
    Automatically choose efficient arguments for `datasets.Dataset.map()`.

    Args:
        dataset: Hugging Face Dataset.
        keep_in_memory_threshold_mb:
            Dataset smaller than this threshold will be kept in memory.

    Returns:
        A dict that can be directly expanded into Dataset.map().
    """
    # ---------- keep_in_memory ----------
    try:
        dataset_size = int(dataset.data.nbytes)
        keep_in_memory = dataset_size < keep_in_memory_threshold_mb * 1024**2
        dataset_size = dataset_size / 1024**2  # convert to MB
    except AttributeError:
        dataset_size = None
        keep_in_memory = False

    # ---------- num_proc ----------
    cpu_count = os.cpu_count() or 1
    n = len(dataset)

    if n < 5_000:
        num_proc = None
    elif n < 50_000:
        num_proc = min(cpu_count, 4)
    else:
        num_proc = cpu_count

    return {
        "keep_in_memory": keep_in_memory,
        "num_proc": num_proc,
    }, dataset_size


def prepare_chat_header(
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

    if template_args.get("apply_chat_template", False):
        # use chat template to format the prompt and response
        chat: List[Dict[str, str]] = []
        # system prompt
        system_prompt = template_args.get("system_prompt", None)
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
            "system_prompt_with_special_tokens", None
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
    chat_header: Union[str, List[Dict[str, str]]],
    template_args: TrackingConfig,
    max_length: int,
    generation_mode: bool = False,
    **kwargs
):
    try:
        # multi-answer support
        if isinstance(a, list):
            if generation_mode:
                raise TypeError("`generation_mode=True` does not support multiple answers per question, please provide a single answer string.")
            multi_a = a
        elif isinstance(a, str):
            multi_a = [a]
        else:
            raise TypeError(f"Expected `str` or `list of str` for answer, but got {type(a)}.")

        if template_args.get("apply_chat_template", False):
            # use chat template
            assert isinstance(chat_header, list)
            prompt = chat_header + [{"role": "user", "content": q}]
            multi_chat = [prompt + [{"role": "assistant", "content": ans}] for ans in multi_a]

            date_str = template_args.get("date_string", None)
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
            assert isinstance(chat_header, str)
            prompt = (
                chat_header
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
            if generation_mode:
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
            f"Error processing sample with index {idx} and header {chat_header}"
        ) from e


def tok_text_sample(
    prompt: str,
    text: str,
    idx: int,
    /, *,
    tokenizer: Any,
    text_max_length: int,
    generation_mode: bool = False,
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
    input_ids = prompt_ids if generation_mode else full_seq_ids
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


_EOI_SENTINEL = "Where was the author of this book born?"


def get_eoi_positions(tokenizer: Any, template_args: Any, chat_header: Any = None) -> List[int]:
    """Negative offsets of the end-of-instruction (post-question template) tokens.

    E.g. Llama-3: `<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n` -> [-5..-1].
    Found as the prompt tokens whose decoded text lies entirely after the question, so it
    follows the same templating (system prompt, date string, tags) as the dataset.
    """
    if chat_header is None:
        chat_header = prepare_chat_header(template_args)
    prompt_ids = tok_chat_sample(
        _EOI_SENTINEL, "", 0,
        tokenizer=tokenizer,
        chat_header=chat_header,
        template_args=template_args,
        max_length=4096,
        generation_mode=True,
    )["input_ids"][0]
    # smallest prefix that still contains the whole question
    prefix_len = len(prompt_ids)
    while prefix_len > 0 and _EOI_SENTINEL in tokenizer.decode(prompt_ids[:prefix_len - 1]):
        prefix_len -= 1
    n_eoi = len(prompt_ids) - prefix_len
    if n_eoi == 0:
        raise ValueError("The prompt template has no tokens after the question; use explicit positions.")
    return list(range(-n_eoi, 0))
