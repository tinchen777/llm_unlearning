"""
Evaluation functions for `evals.metrics` module.
"""
from __future__ import annotations
import torch
import logging
from typing import List, Any, Dict, Mapping, Tuple, TYPE_CHECKING

from data.utils import IGNORE_INDEX
from model.inference.utils import forward_batch, batch_to_model_device, generate_batch
from model.inference.ops import per_token_CE, build_prompt_answer_batch
from ..response.text_metric_utils import TextMetric
from .utils import to_np

if TYPE_CHECKING:
    from utils.config import TrackingConfig

logger = logging.getLogger("eval.metric")


@batch_to_model_device
def evaluate_probability(model: Any, batch: Mapping[str, torch.Tensor]) -> List[Dict[str, float]]:
    """Evaluate model probabilities and average token-level loss for a given batch."""
    # forward
    logits, _ = forward_batch(model, batch, ignore_labels=True, grad=False)

    losses, shift_labels_mask = per_token_CE(logits, batch["labels"], ignore_index=IGNORE_INDEX)

    avg_losses = losses.sum(dim=-1) / shift_labels_mask.sum(dim=-1)
    normalized_probs = (-avg_losses).exp()  # Convert average loss to probability

    return [
        {"prob": prob, "avg_loss": avg_loss}
        for prob, avg_loss in zip(
            to_np(normalized_probs).tolist(), to_np(avg_losses).tolist()
        )
    ]


@batch_to_model_device
def tokenwise_logprobs(
    model: Any,
    batch: Mapping[str, torch.Tensor],
    grad: bool = False,
    drop_last: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Token-wise next-token logprobs of the labeled tokens of every sample.

    Position t of the logits predicts token t+1, so labeled token j is scored by position j-1; token 0 is never
    a target. `drop_last` skips the last labeled token of each row (the upstream OpenUnlearning behaviour).

    Returns (per sample, `n` = number of scored tokens)
    - `vocab_logprobs_batch`: List[Tensor[n, vocab_size]] (float32), or `None` if `return_vocab` is False
    - `target_logprobs_batch`: List[Tensor[n]] (float32)
    - `labels_batch`: List[Tensor[n]]
    """
    logits, _ = forward_batch(model, batch, ignore_labels=True, grad=grad)
    labels = batch["labels"][:, 1:]                          # target of position t is token t+1
    pred_mask = labels != IGNORE_INDEX                       # [bsz, seq_len-1]
    if drop_last:
        # sync with OpenUnlearning: drop the last labeled token of each row
        cols = torch.arange(pred_mask.shape[1], device=pred_mask.device)
        last = torch.where(pred_mask, cols, -1).max(dim=1).values
        pred_mask = pred_mask & (cols != last.unsqueeze(1))

    rows, cols = pred_mask.nonzero(as_tuple=True)            # only the scored positions
    vocab_logprobs = logits[:, :-1][rows, cols].float().log_softmax(dim=-1)   # [N, vocab_size]
    targets = labels[rows, cols]                             # [N]
    target_logprobs = vocab_logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [N]

    counts = pred_mask.sum(dim=1).tolist()                   # one GPU->CPU sync for the whole batch
    return (
        list(vocab_logprobs.split(counts)),
        list(target_logprobs.split(counts)),
        list(targets.split(counts)),
    )


@batch_to_model_device
def answer_logprobs(model: Any, batch: Mapping[str, torch.Tensor], pad_token_id: int) -> List[float]:
    """Mean per-token log-prob of the ground-truth answer of every row, given its prompt (teacher forcing)."""
    full_batch = build_prompt_answer_batch(batch, pad_token_id, ignore_index=IGNORE_INDEX)
    _, target_logprobs, _ = tokenwise_logprobs(model, full_batch, grad=False, drop_last=False)
    sums = torch.stack([lp.sum() for lp in target_logprobs])
    counts = torch.tensor([len(lp) for lp in target_logprobs], device=sums.device)
    return (sums / counts.clamp(min=1)).tolist()


def eval_text_similarity(
    model: Any,
    batch: Mapping[str, torch.Tensor],
    tokenizer: Any,
    generation_args: TrackingConfig
):
    batch_records = generate_batch(
        model, batch, tokenizer, generation_args,
        ignore_index=IGNORE_INDEX,
        with_meta=False
    )
    # compute rouge scores for each generated record
    text_metrics = TextMetric(rouge_scores={})
    for record in batch_records:
        record.update(text_metrics.compute_metrics(record))
    return batch_records
