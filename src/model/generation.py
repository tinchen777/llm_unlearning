"""Text generation and teacher-forced answer scoring of a model.

- `generate_texts`  : generate for a batch of prompts and return the cleaned texts (stop strings, EOS cut-off); the
                      generation step behind the evals' ROUGE metrics (`evals.metrics.metric_utils.eval_text_similarity`).
- `answer_logprobs` : mean per-token log-prob of the ground-truth answer of every row (teacher forcing), i.e. the log of
                      TOFU's length-normalised answer probability.

Requirements on the batches (same as the generation metrics):
- prompt-only `input_ids` -> dataset built with `predict_with_generate: true`
- LEFT padding -> collator `padding_side: left` (batched generation appends after the last column)
"""

from __future__ import annotations
import torch
from typing import Any, List, Mapping, TYPE_CHECKING

from .ops import mean_answer_logprob
from .utils import append_answers, batch_to_device

if TYPE_CHECKING:
    from utils.config import TrackingConfig


def _cut_off_at_stopwords(s: str, seps: List[str]) -> str:
    """Cut off the string `s` at the earliest occurrence of any of the stopwords in `seps`."""
    end = len(s)
    for sep in seps:
        idx = s.find(sep)
        if idx != -1:
            end = min(end, idx)
    return s[:end].strip()


def generate_texts(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    generation_args: TrackingConfig
) -> List[str]:
    """Generate for every (left padded) prompt of `input_ids`; returns the new text only, cut off at the stopwords."""
    initial_input_length = input_ids.shape[1]

    generation_kwargs = generation_args.to_dict()
    assert not generation_kwargs.get("return_dict_in_generate")
    stopwords = list(generation_kwargs.pop("stopwords", []))
    if stopwords:
        generation_kwargs["stop_strings"] = stopwords
        generation_kwargs["tokenizer"] = tokenizer

    # generate outputs
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        **generation_kwargs,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_texts = tokenizer.batch_decode(
        outputs[:, initial_input_length:],
        skip_special_tokens=True
    )

    # cut off at stopwords and strip
    return [_cut_off_at_stopwords(text, stopwords) for text in gen_texts]


@torch.no_grad()
def answer_logprobs(model: Any, batch: Mapping[str, Any], pad_token_id: int, ignore_index: int = -100) -> List[float]:
    """Mean per-token log-prob of the ground-truth answer of every row, given its prompt (teacher forcing).

    `batch` is a LEFT padded prompt-only batch (`predict_with_generate`); the answer tokens are read from
    `labels` and appended after the prompt columns (`append_answers`), so the prompt keeps the same columns as in
    generation (a `Steerer` hooked on the model therefore redirects exactly the same tokens).
    """
    inputs, answer_ids, answer_mask = append_answers(batch, pad_token_id, ignore_index)
    logits = model(**batch_to_device(inputs, model.device)).logits
    return mean_answer_logprob(logits, answer_ids, answer_mask).tolist()
