"""Generate text responses for a dataloader, independent of any Trainer / evaluator.

Reuses `eval_text_similarity` (the generation step behind every ROUGE metric), so the
decoding behaviour (`generation_args`, stop strings, EOS cut-off) is identical to the evals.

Requirements on the loader (same as the generation metrics):
- prompt-only `input_ids` -> dataset built with `predict_with_generate: true`
- LEFT padding -> collator `padding_side: left` (batched generation appends after the last column)

Besides the text, every record gets:
- `answer_logprob`: mean log-prob per token of the GROUND-TRUTH answer given the prompt (teacher forcing),
                    i.e. log of TOFU's length-normalised answer probability. High = the model knows the answer.
- `refusal`       : the generation matches one of `refusal_patterns` ("I don't know", ...).
- `degenerate`    : empty or highly repetitive generation (a sign of a broken model / too strong an edit).
"""

from __future__ import annotations
import math
import torch
from tqdm import tqdm
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from evals.metrics.metric_utils import eval_text_similarity
from utils.common import IGNORE_INDEX

# keys the model / eval function consume; everything else in a batch is per-sample metadata
MODEL_INPUT_KEYS = ("input_ids", "attention_mask", "labels")

DEFAULT_REFUSAL_PATTERNS = (
    "i don't know", "i do not know", "i don't have", "i do not have", "i'm not aware", "i am not aware",
    "not familiar with", "no information", "i cannot", "i can't", "i'm not able", "i am not able",
    "i'm unable", "i am unable", "unable to provide", "i couldn't find", "i could not find",
    "i'm sorry", "i apologize", "not publicly available", "doesn't exist", "does not exist",
)


def _as_list(values: Any) -> List[Any]:
    return values.tolist() if isinstance(values, torch.Tensor) else list(values)


def is_refusal(text: str, patterns: Sequence[str] = DEFAULT_REFUSAL_PATTERNS) -> bool:
    text = text.lower().replace("’", "'")
    return any(p in text for p in patterns)


def is_degenerate(text: str, n: int = 3, min_distinct: float = 0.5) -> bool:
    """Empty, or (for 2n+ words) fewer than `min_distinct` of its word n-grams are distinct (looping text)."""
    words = text.split()
    if not words:
        return True
    if len(words) < 2 * n:
        return False
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams) < min_distinct


@torch.no_grad()
def answer_logprobs(model: Any, batch: Mapping[str, Any], pad_token_id: int) -> List[float]:
    """Mean per-token log-prob of the ground-truth answer of every row, given its prompt (teacher forcing).

    `batch` is a LEFT padded prompt-only batch (`predict_with_generate`); the answer tokens are read from
    `labels` and appended after the prompt columns, so the prompt keeps the same columns as in generation
    (a `Steerer` hooked on the model therefore redirects exactly the same tokens).
    """
    ids, am, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
    if not bool(am[:, -1].all()):
        raise ValueError("answer_logprobs needs LEFT padded prompt-only batches (collator padding_side: left).")
    answers = [label[label != IGNORE_INDEX] for label in labels]
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


def generate_responses(
    model: Any,
    tokenizer: Any,
    loader: Iterable[Mapping[str, Any]],
    generation_args: Any,
    max_samples: Optional[int] = None,
    desc: str = "",
    with_logprob: bool = True,
    refusal_patterns: Sequence[str] = DEFAULT_REFUSAL_PATTERNS,
    before_batch: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """Generate a response for every sample of `loader`.

    Args:
        generation_args: a `TrackingConfig` (e.g. `cfg["generation_args"]`); see configs/generation/.
        max_samples: stop after this many samples (None -> the whole loader).
        with_logprob: also compute `answer_logprob` of the ground truth (one extra forward pass per batch).
        before_batch: called with every batch (metadata included) before the model sees it, e.g. to set
            the per-row vectors of a `Steerer`.
    Returns:
        One record per sample: the batch metadata (`index`, `author_id`, ...) plus `input` (decoded prompt),
        `ground_truth`, `generation`, the ROUGE scores, `answer_logprob`, `refusal` and `degenerate`.
    """
    model.eval()
    records: List[Dict[str, Any]] = []
    for batch in tqdm(loader, desc=f"Generating [{desc}]", unit="batch(es)", colour="blue"):
        if max_samples is not None and len(records) >= max_samples:
            break
        if before_batch is not None:
            before_batch(batch)
        # metadata (index, author_id, ...) is not a tensor input of the model: set it aside
        meta = {k: _as_list(batch[k]) for k in batch if k not in MODEL_INPUT_KEYS}
        inputs = {k: batch[k] for k in MODEL_INPUT_KEYS}
        logprobs = answer_logprobs(model, inputs, tokenizer.pad_token_id) if with_logprob else None
        outputs = eval_text_similarity(model, inputs, tokenizer=tokenizer, generation_args=generation_args)
        for i, output in enumerate(outputs):
            records.append({
                **{k: v[i] for k, v in meta.items()},
                **output,
                **({"answer_logprob": logprobs[i]} if logprobs is not None else {}),
                "refusal": is_refusal(output["generation"], refusal_patterns),
                "degenerate": is_degenerate(output["generation"]),
            })
    return records[:max_samples] if max_samples is not None else records


def summarize_responses(records: Sequence[Mapping[str, Any]], hallucination_rouge: float = 0.3) -> Dict[str, Any]:
    """Aggregate of `generate_responses` records.

    `hallucination_rate`: a confident (no refusal, not degenerate) answer that does NOT match the ground truth
    (rougeL_recall < `hallucination_rouge`) - the behaviour expected on entities the model has never seen.
    """
    n = len(records)
    if n == 0:
        return {"n": 0}
    mean = lambda key: sum(float(r[key]) for r in records) / n
    out: Dict[str, Any] = {
        "n": n,
        "rougeL_recall": mean("rougeL_recall"),
        "rouge1_recall": mean("rouge1_recall"),
        "refusal_rate": mean("refusal"),
        "degenerate_rate": mean("degenerate"),
        "hallucination_rate": sum(
            (not r["refusal"]) and (not r["degenerate"]) and r["rougeL_recall"] < hallucination_rouge for r in records
        ) / n,
        "mean_words": sum(len(str(r["generation"]).split()) for r in records) / n,
    }
    if all("answer_logprob" in r for r in records):
        out["answer_logprob"] = mean("answer_logprob")
        out["answer_prob"] = sum(math.exp(r["answer_logprob"]) for r in records) / n
    return out
