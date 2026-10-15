"""Helpers of the NEAR scripts that need both the data templates and the model."""

from __future__ import annotations
from typing import Any, List

from data.datasets.datasets_utils import prepare_chat_header, tok_chat_sample

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
        predict_with_generate=True,
    )["input_ids"][0]
    # smallest prefix that still contains the whole question
    prefix_len = len(prompt_ids)
    while prefix_len > 0 and _EOI_SENTINEL in tokenizer.decode(prompt_ids[:prefix_len - 1]):
        prefix_len -= 1
    n_eoi = len(prompt_ids) - prefix_len
    if n_eoi == 0:
        raise ValueError("The prompt template has no tokens after the question; use explicit positions.")
    return list(range(-n_eoi, 0))
