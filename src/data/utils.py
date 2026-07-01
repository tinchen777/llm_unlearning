
from __future__ import annotations
import torch
import datasets
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.config import TrackingConfig

IGNORE_INDEX = -100

logger = logging.getLogger("data")


def load_hf_dataset(path: str, add_index: bool = False, **kwargs) -> datasets.Dataset:
    dataset = datasets.load_dataset(path, **kwargs)
    if add_index:
        dataset = dataset.add_column("index", np.arange(len(dataset)))
    return dataset


def prepare_sample_context(
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


def tok_chat_batch(
    question_batch: List[str],
    answer_batch: List[str],
    indices: List[int],
    tokenizer: Any,
    sample_context: Union[str, List[Dict[str, str]]],
    template_args: TrackingConfig,
    max_length: int,
    predict_with_generate: bool = False,
):
    try:
        if template_args.get("apply_chat_template", False):
            # use chat template
            assert isinstance(sample_context, list)
            prompt_batch = [sample_context + [{"role": "user", "content": q}] for q in question_batch]
            chat_batch = [prompt + [{"role": "assistant", "content": a}] for prompt, a in zip(prompt_batch, answer_batch)]

            date_str = template_args.get("date_string", None, allow_none=True)
            date_info = {"date_string": date_str} if date_str is not None else {}
            prompt_ids_batch = tokenizer.apply_chat_template(
                prompt_batch,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
                max_length=max_length,
                truncation=True,
                **date_info
            )
            chat_ids_batch = tokenizer.apply_chat_template(
                chat_batch,
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
            prompt_batch = [(
                sample_context
                + template_args["user_start_tag"]
                + q
                + template_args["user_end_tag"]
                + template_args["asst_start_tag"]
            ) for q in question_batch]
            chat_batch = [prompt + a for prompt, a in zip(prompt_batch, answer_batch)]

            prompt_ids_batch = tokenizer(
                prompt_batch,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True
            )["input_ids"]
            chat_ids_batch = tokenizer(
                chat_batch,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True
            )["input_ids"]

        input_ids_batch, labels_batch = [], []
        for prompt_ids, chat_ids in zip(prompt_ids_batch, chat_ids_batch):
            if chat_ids[-1] != tokenizer.eos_token_id:
                chat_ids = chat_ids + [tokenizer.eos_token_id]     # 先补 eos

            if predict_with_generate:
                input_ids_batch.append(prompt_ids); labels_batch.append(chat_ids)
            else:
                labels = [IGNORE_INDEX]*len(prompt_ids) + chat_ids[len(prompt_ids):]
                input_ids_batch.append(chat_ids);   labels_batch.append(labels)
        return {"input_ids": input_ids_batch, "labels": labels_batch, "index": indices}

    except Exception as e:
        raise RuntimeError(
            f"Error processing batch with indices {indices} and sample_context {sample_context}"
        ) from e








# FIXME change to batch map
def preprocess_chat_instance(
    tokenizer: Any,
    template_args: TrackingConfig,
    prompt_msgs: List[str],
    response_msgs: List[str],
    max_length: int,
    predict_with_generate: bool = False,
):
    """Preprocesses a chat instance for training or generation.
    When in training, both the returned `input_ids` and `labels` cover the entire conversation.
    `input_ids` has no padding, and `labels` assign `IGNORE_INDEX` to tokens where loss is not computed (i.e. all tokens except the final response message).
    When in generation, `input_ids` are returned only up to the last user prompt, excluding the assistant's response. The `labels` returned are the same as during training.
    `attention_mask` is always 1 over the full `input_ids` token sequence.

    `prompt_msgs` and `response_msgs` are lists where, except for the last pair, all
    corresponding pairs are in-context examples. When they are a string and not
    a list, there are no in-context examples.

    Args:
        tokenizer: Tokenizer to apply on text
        template_args (TrackingConfig): Configuration for the chat template (comes from model-specific config).
        prompt_msgs (Union[List[str], str]): List of prompt messages or a single prompt message string.
        response_msgs (Union[List[str], str]): List of response messages or a single response message string.
        max_length (int): Maximum sequence length after tokenization.
        predict_with_generate (bool, optional): Whether to prepare inputs for generation.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing 'input_ids', 'labels', and 'attention_mask' tensors for model input.
    """
    if len(prompt_msgs) != len(response_msgs):
        raise ValueError(
            f"The number of prompt messages ({len(prompt_msgs)}) must match the number of response messages ({len(response_msgs)})."
        )

    if template_args.get("apply_chat_template", False):
        chat = []
        system_prompt = template_args.get("system_prompt", None, allow_none=True)
        if system_prompt:
            chat.append({"role": "system", "content": system_prompt})
        for prompt, response in zip(prompt_msgs, response_msgs):
            chat.append({"role": "user", "content": prompt})
            chat.append({"role": "assistant", "content": response})
        date_str = template_args.get("date_string", None, allow_none=True)
        date_info = {"date_string": date_str} if date_str is not None else {}
        chat_ids = tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
            max_length=max_length,
            truncation=True,
            **date_info
        )
        prompt_ids = tokenizer.apply_chat_template(
            chat[:-1],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            max_length=max_length,
            truncation=True,
            **date_info
        )
    else:
        wrapped_prompt = ""
        system_prompt_with_special_tokens = template_args.get(
            "system_prompt_with_special_tokens", None, allow_none=True
        )
        if system_prompt_with_special_tokens:
            wrapped_prompt += system_prompt_with_special_tokens

        final_response = ""
        for idx, (prompt, response) in enumerate(zip(prompt_msgs, response_msgs)):
            wrapped_prompt += (
                template_args["user_start_tag"]
                + prompt
                + template_args["user_end_tag"]
                + template_args["asst_start_tag"]
            )
            if idx < len(prompt_msgs) - 1:
                # few-shot examples, add the response and end tag
                wrapped_prompt += (response + template_args["asst_end_tag"])
            else:
                # final example, add the response but no end tag, as we want to predict it
                final_response = response

        chat_ids = tokenizer(
            wrapped_prompt + final_response,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True
        )["input_ids"]

        prompt_ids = tokenizer(
            wrapped_prompt,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True
        )["input_ids"]

    if chat_ids[-1] != tokenizer.eos_token_id:
        chat_ids += [tokenizer.eos_token_id]

    chat_ids_tensor = torch.tensor(chat_ids)
    prompt_ids_tensor = torch.tensor(prompt_ids)

    item = {}
    if predict_with_generate:
        item["input_ids"] = prompt_ids_tensor
        item["labels"] = chat_ids_tensor  # contains the entire conversation
    else:
        item["input_ids"] = chat_ids_tensor
        labels_tensor = chat_ids_tensor.clone()
        labels_tensor[: len(prompt_ids_tensor)] = IGNORE_INDEX
        item["labels"] = labels_tensor
        if len(prompt_ids_tensor) == len(chat_ids_tensor):
            # Rarely, tokenization can result in this condition being entered.
            # Say a input prompt is ABC and target output is D, tokenizer(ABCD)
            # can be [AB, CD] and tokenizer(ABC) can be [AB, C]. In this case,
            # we ignore loss on all indices in the labels. So, there is no way
            # to use this for next token prediction. Be careful while
            # interpreting results of such instances.
            logger.warning(
                "Tokenization mismatch: no valid target tokens for loss computation"
            )
    item["attention_mask"] = torch.ones_like(item["input_ids"], dtype=torch.long)
    return item


def preprocess_pretraining_instance(
    tokenizer: Any,
    prefix: str,
    text_content: str,
    max_length: int,
    predict_with_generate: bool = False,
    insert_space: bool = False,
) -> Dict[str, torch.Tensor]:
    """Preprocesses a pretraining instance for training or generation.
    When in training, both the returned `input_ids` and `labels` are over the entire token sequence. `input_ids` has no padding, `labels` assigns `IGNORE_INDEX` to ignore all tokens that we don't compute loss over (i.e. the the 0th index token, all prefix tokens)
    When in generation, `input_ids` are returned only until the prefix portion. The `labels` returned are the same as during training.
    `attention_mask` is always 1 over the full input token sequence.
    Args:
        tokenizer: Tokenizer to apply on text
        prefix (str): The prefix string to prepend to the content.
        text_content (str): The main text content (following the prefix) to be tokenized.
        max_length (int): Maximum text content length after tokenization.
        predict_with_generate (bool, optional): Whether to prepare inputs for generation.
        insert_space (bool, optional): Whether to insert a space between prefix and content.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing 'input_ids', 'labels', and 'attention_mask' tensors for model input.
    """
    full_seq_ids = tokenizer(
        prefix + (" " if insert_space else "") + text_content, add_special_tokens=True
    )["input_ids"]
    prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    prefix_len = len(prefix_ids)
    full_seq_ids = full_seq_ids[: prefix_len + max_length]  # manual truncation

    len_matched = prefix_len
    if len_matched == 0:  # never give loss on index 0, when prefix is empty
        len_matched = 1
    labels = [IGNORE_INDEX] * len_matched + full_seq_ids[len_matched:]
    item = {}
    if predict_with_generate:
        item["input_ids"] = prefix_ids
    else:
        item["input_ids"] = full_seq_ids
    item["labels"] = labels
    item["attention_mask"] = [1] * len(item["input_ids"])
    for attr in item:
        item[attr] = torch.tensor(item[attr])
    return item
