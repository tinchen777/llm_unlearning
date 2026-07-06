
from __future__ import annotations
from torch import nn
import torch
from rouge_score import rouge_scorer
from transformers import StoppingCriteria, StoppingCriteriaList, PreTrainedTokenizer
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


class MultiTokenEOSCriteria(StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence. Stopping Criteria forked
    and modified from [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/27924d77953491f66a038a09892807065e469358/lm_eval/models/utils.py#L208)"""

    def __init__(
        self,
        sequence: str,
        tokenizer: PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ) -> None:
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        # we look back for 2 more tokens than it takes to encode our stop sequence
        # because tokenizers suck, and a model might generate `['\n', '\n']` but our `sequence` is `['\n\n']`
        # and we don't want to mistakenly not stop a generation because our
        # (string) stop sequence was output in a different tokenization

        # NOTE: there is a minor danger that this will end up looking back 2 tokens into the past, into the inputs to the model,
        # and stopping generation immediately as a result. With only 2 extra tokens of lookback, this risk is minimized
        # Additionally, in lookback_ids_batch we should prevent ever looking back into the inputs as described.
        self.sequence_id_len = len(self.sequence_ids) + 2
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :]

        lookback_ids_batch = lookback_ids_batch[:, -self.sequence_id_len :]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> StoppingCriteriaList:
    return StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )


@batch_to_model_device
def eval_text_similarity(model: Any, batch: Mapping[str, torch.Tensor], tokenizer: Any, generation_args: TrackingConfig):
    """Evaluate text similarity between model-generated outputs and ground truth using ROUGE scores."""

    def _eval_rouge_recall_batch(gen_outputs, ground_truths):
        evals = []
        for gen, gt in zip(gen_outputs, ground_truths):
            rouge_scores = _ROUGE_SCORER.score(gt, gen)
            evals.append({
                "rouge1_recall": rouge_scores["rouge1"].recall,
                "rougeL_f1": rouge_scores["rougeL"].fmeasure,
                "rougeL_recall": rouge_scores["rougeL"].recall,
            })
        return evals

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    input_texts = tokenizer.batch_decode(
        input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    tokens = [label[label != IGNORE_INDEX] for label in labels]
    full_texts = tokenizer.batch_decode(
        tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    ground_truths = [
        full_text.replace(input_text, "").strip()
        for input_text, full_text in zip(input_texts, full_texts)
    ]

    attention_mask = batch["attention_mask"]

    # convert to a simple dict from DictConfig
    generation_kwargs = generation_args.to_dict()
    stopwords = generation_kwargs.pop("stopwords", None)
    if stopwords is not None:
        assert isinstance(stopwords, list)
        sc = stop_sequences_criteria(
            tokenizer, stopwords, input_ids.shape[1], input_ids.shape[0]
        )
        generation_kwargs["stopping_criteria"] = sc
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        **generation_kwargs,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_texts = tokenizer.batch_decode(
        output[:, input_ids.shape[-1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # cut off at stopwords
    if stopwords is None:
        stopwords = []
    stopwords = [tokenizer.decode([tokenizer.eos_token_id])] + stopwords
    for i in range(len(gen_texts)):
        raw_text = gen_texts[i]
        for word in stopwords:
            if word and word in raw_text:
                raw_text = raw_text.split(word)[0]
        raw_text = raw_text.strip()
        gen_texts[i] = raw_text

    scores = _eval_rouge_recall_batch(gen_texts, ground_truths)
    scores = [
        {
            **rouge_evals,
            "input": input_text,
            "ground_truth": ground_truth,
            "generation": gen_text,
        }
        for rouge_evals, input_text, ground_truth, gen_text in zip(
            scores, input_texts, ground_truths, gen_texts
        )
    ]
    return scores


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
