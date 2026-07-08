
from __future__ import annotations
from torch import nn
import torch
from rouge_score import rouge_scorer
import logging
from typing import List, Any, Dict, Mapping, TYPE_CHECKING

from utils.common import IGNORE_INDEX
from .utils import batch_to_model_device, forward_batch, to_np

if TYPE_CHECKING:
    from utils.config import TrackingConfig

logger = logging.getLogger("eval.metric")


@batch_to_model_device
def evaluate_probability(model: Any, batch: Mapping[str, torch.Tensor]) -> List[Dict[str, float]]:
    """Evaluate model probabilities and average token-level loss for a given batch."""
    # forward
    logits = forward_batch(model, batch)

    labels = batch["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    shifted_logits = logits[..., :-1, :].contiguous()

    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
    # agg loss across tokens
    losses = loss_fn(shifted_logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    num_token_gt = (shifted_labels != IGNORE_INDEX).sum(-1)
    avg_losses = losses / num_token_gt
    normalized_probs = torch.exp(-avg_losses)

    return [
        {"prob": prob, "avg_loss": avg_loss}
        for prob, avg_loss in zip(
            to_np(normalized_probs).tolist(), to_np(avg_losses).tolist()
        )
    ]


@batch_to_model_device
def tokenwise_logprobs(model: Any, batch: Mapping[str, torch.Tensor], grad: bool = False):
    """Compute token-wise next token prediction logprobs for all labeled tokens for each sample in a batch.
    `grad` decides whether gradients are turned on
    Returns
    - `vocab_logprobs_batch` (List[Tensor]): Tensors of shape [seq_len, vocab_size]
    - `target_logprobs_batch` (List[Tensor]): Tensors of shape [seq_len]
    - `labels_batch` (List[Tensor]): Tensors of shape [seq_len]
    """
    # forward
    logits = forward_batch(model, batch, grad=grad)

    vocab_logprobs = logits.log_softmax(dim=-1)[:, :-1, :]  # shape as [bsz, seq_len-1, vocab_size]
    vocab_size = vocab_logprobs.shape[-1]

    next_tokens = batch["input_ids"][:, 1:].unsqueeze(-1)  # shape as [bsz, seq_len-1, 1]
    target_logprobs = torch.gather(vocab_logprobs, dim=2, index=next_tokens).squeeze(-1)  # shape as [bsz, seq_len-1]

    vocab_logprobs_batch: List[torch.Tensor] = []
    target_logprobs_batch: List[torch.Tensor] = []
    labels_batch: List[torch.Tensor] = []
    for i, labels in enumerate(batch["labels"]):
        # only focus on tokens which have loss on them (i.e. used in labels)

        # Ignore the last labeled token because there is no prediction after it.
        actual_indices = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0][:-1]
        if len(actual_indices) == 0:
            vocab_logprobs_batch.append(vocab_logprobs.new_empty((0, vocab_size)))
            target_logprobs_batch.append(target_logprobs.new_empty(0))
            labels_batch.append(labels.new_empty(0))
            continue
        start_idx, end_idx = actual_indices[0].item(), actual_indices[-1].item()
        if start_idx == 0:
            logger.warning(
                f"labels[0] is not {IGNORE_INDEX}. "
                "The first token should not contribute to the next-token prediction loss."
            )
        # Return full distribution for each position: shape [seq_len, vocab_size]
        vocab_logprobs_batch.append(vocab_logprobs[i, start_idx - 1: end_idx])
        target_logprobs_batch.append(target_logprobs[i, start_idx - 1: end_idx])
        labels_batch.append(labels[actual_indices])

    return vocab_logprobs_batch, target_logprobs_batch, labels_batch


_ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


@batch_to_model_device
def eval_text_similarity(model: Any, batch: Mapping[str, torch.Tensor], tokenizer: Any, generation_args: TrackingConfig):
    """Evaluate text similarity between model-generated outputs and ground truth using ROUGE scores."""

    def _cut_off_at_stopwords(s: str, seps: List[str]) -> str:
        """Cut off the string `s` at the earliest occurrence of any of the stopwords in `seps`."""
        end = len(s)
        for sep in seps:
            idx = s.find(sep)
            if idx != -1:
                end = min(end, idx)
        return s[:end].strip()


    def _eval_rouge_recall(gen_text: str, ground_truth: str):
        rouge_scores = _ROUGE_SCORER.score(ground_truth, gen_text)
        return {
            "rouge1_recall": rouge_scores["rouge1"].recall,
            "rougeL_f1": rouge_scores["rougeL"].fmeasure,
            "rougeL_recall": rouge_scores["rougeL"].recall
        }


    input_ids = batch["input_ids"]
    labels = batch["labels"]
    # batch_size = input_ids.shape[0]
    initial_input_length = input_ids.shape[1]
    input_texts = tokenizer.batch_decode(
        input_ids,
        skip_special_tokens=True
    )
    ground_truths = tokenizer.batch_decode(
        [label[label != IGNORE_INDEX] for label in labels],
        skip_special_tokens=True
    )
    attention_mask = batch["attention_mask"]

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
    gen_texts = [_cut_off_at_stopwords(text, stopwords) for text in gen_texts]

    return [
        {
            "input": input_text,
            "ground_truth": ground_truth,
            "generation": gen_text,
            **_eval_rouge_recall(gen_text, ground_truth)
        }
        for input_text, ground_truth, gen_text in zip(
            input_texts, ground_truths, gen_texts
        )
    ]


def get_decoded_target_texts(tokenizer: Any, batch: Mapping[str, torch.Tensor]) -> List[str]:
    """Extract and detokenize text from activated positions in the batch."""
    labels = batch["labels"]
    return [
        tokenizer.decode(elem[elem != IGNORE_INDEX].tolist(), skip_special_tokens=True)
        for elem in labels
    ]


# def get_forget_quality(model_tr, reference_tr):
#     test_res = sc.stats.ks_2samp(1 / (model_tr + 1e-10), 1 / (reference_tr + 1e-10))
#     return {"agg_value": test_res.pvalue}
