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
    `labels` and appended after the prompt columns, so the prompt keeps the same columns as in generation
    (a `Steerer` hooked on the model therefore redirects exactly the same tokens).
    """
    ids, am, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
    if not bool(am[:, -1].all()):
        raise ValueError("answer_logprobs needs LEFT padded prompt-only batches (collator padding_side: left).")
    answers = [label[label != ignore_index] for label in labels]
    bsz, n_prompt, n_ans = ids.shape[0], ids.shape[1], max(len(a) for a in answers)
    ans = torch.full((bsz, n_ans), pad_token_id, dtype=ids.dtype)
    ans_mask = torch.zeros((bsz, n_ans), dtype=am.dtype)
    for i, a in enumerate(answers):
        ans[i, :len(a)] = a
        ans_mask[i, :len(a)] = 1
    full_ids = torch.cat([ids, ans], dim=1).to(model.device)
    full_mask = torch.cat([am, ans_mask], dim=1).to(model.device)
    position_ids = (full_mask.long().cumsum(-1) - 1).clamp(min=0)
    logits = model(input_ids=full_ids, attention_mask=full_mask, position_ids=position_ids).logits
    # the token at column n_prompt + j is predicted by column n_prompt + j - 1
    logp = logits[:, n_prompt - 1:n_prompt + n_ans - 1].float().log_softmax(-1)
    ans, ans_mask = ans.to(model.device), ans_mask.to(model.device).float()
    token_lp = logp.gather(-1, ans.unsqueeze(-1)).squeeze(-1) * ans_mask
    return (token_lp.sum(-1) / ans_mask.sum(-1).clamp(min=1)).tolist()
