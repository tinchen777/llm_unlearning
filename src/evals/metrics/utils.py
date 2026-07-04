
from __future__ import annotations
import numpy as np
from torch import nn
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from rouge_score import rouge_scorer
from transformers import BatchEncoding
from transformers import StoppingCriteria, StoppingCriteriaList, PreTrainedTokenizer
import warnings
from typing import List, Any, Dict, Callable, TYPE_CHECKING

from utils.common import IGNORE_INDEX

if TYPE_CHECKING:
    from utils.config import TrackingConfig

DATA_SPLIT_SUFFIX = "_dl"


def to_device(batch: Any, device: torch.device):
    """Move a batch of data to the specified device."""
    if isinstance(batch, dict):
        return {k: v.to(device) for k, v in batch.items()}
    elif isinstance(batch, BatchEncoding):
        return batch.to(device)
    else:
        raise ValueError(
            f"Expected batch to be a `dict` or `BatchEncoding`, but got {type(batch)}."
        )


def dict_transpose(evals):
    """Transpose a nested dictionary structure to group statistics by item indices."""
    # evals looks like {iidx0: {idx453: {prob: 0.1, loss: 1}},
    #                   iidx1: {idx453: {prob: 0.2, loss: 2}}}
    # multiple answers indexed by intra_item_idx, then item_idx
    # invert the dict, put outermost iidx deepest inside
    # after dict transpose looks like {idx453: {prob: [0.1, 0.2], loss: [1, 2]}
    all_iidxs = list(evals.keys())
    all_idxs = list(evals[all_iidxs[0]].keys())
    all_stat_names = list(evals[all_iidxs[0]][all_idxs[0]].keys())
    evals = {
        idx: {
            stat: [evals[iidx][idx][stat] for iidx in all_iidxs]
            for stat in all_stat_names
        }
        for idx in all_idxs
    }
    return evals


def aggregate_to_1D(x):
    return np.mean(x, axis=tuple(range(1, x.ndim)))


# def get_forget_quality(model_tr, reference_tr):
#     test_res = sc.stats.ks_2samp(1 / (model_tr + 1e-10), 1 / (reference_tr + 1e-10))
#     return {"agg_value": test_res.pvalue}


def run_batchwise_evals(
    model: Any,
    dataloader: Any,
    batch_eval_fn: Callable[..., List[Dict[str, Any]]],
    batch_eval_fn_args: Dict[str, Any],
    eval_msg: str
):
    """Run batch-wise evaluations on a dataset using a specified evaluation function. Handles
    multi-answer datasets by organizing evaluations by answer indices and aggregating results."""
    evals: Dict[int, Dict[int, Dict[str, Any]]] = {}
    # evals looks like {iidx0: {idx453: {prob: 0.1, loss: 1}},
    #                   iidx1: {idx453: {prob: 0.2, loss: 2}}}
    try:
        for batch in tqdm(dataloader, desc=eval_msg, total=len(dataloader)):
            if "input_ids" in batch:
                batch = {0: batch}
            # Assume batch like {0: {"input_ids": [[]]..., index: [453, 454..]},
            #                    1: {"input_ids": [[]]..., index: [453, 454..]}..}
            for intra_item_idx, mini_batch in batch.items():
                if "input_ids" not in mini_batch:
                    raise ValueError(
                        f"Expected mini_batch to contain 'input_ids', but got {list(mini_batch)}."
                    )
                data_indices: List[int] = mini_batch.pop("index")
                # mini_batch = to_device(mini_batch, model.device)
                batch_evals = batch_eval_fn(
                    model=model,
                    batch=to_device(mini_batch, model.device),
                    **batch_eval_fn_args
                )
                indexwise_batch_evals = dict(zip(data_indices, batch_evals))

                evals.setdefault(intra_item_idx, {}).update(indexwise_batch_evals)

        if len(evals) == 1:  # normal single answer dataset, no need for list
            return next(iter(evals.values()))
        else:
            # for each index return a dict with all intra_item_idx values in list
            # after dict transpose looks like {idx453: {prob: [0.1, 0.2], loss: [1, 2]}}
            return dict_transpose(evals)

    except Exception as e:
        raise RuntimeError(f"Error during batch-wise evaluation with {eval_msg}") from e
    finally:
        print("Evaluated", len(evals), "examples")


def evaluate_probability(model, batch):
    """Evaluate model probabilities and average token-level loss for a given batch."""
    # batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)

    logits = output.logits
    labels = batch["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
    # agg loss across tokens
    losses = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    num_token_gt = (batch["labels"] != IGNORE_INDEX).sum(-1)
    avg_losses = losses / num_token_gt
    normalized_probs = torch.exp(-avg_losses)

    # .float() is required: numpy has no bfloat16, so converting a bf16 tensor
    # (when the model runs in bfloat16) directly via .numpy() raises
    # "TypeError: Got unsupported ScalarType BFloat16".
    avg_losses = avg_losses.float().cpu().numpy().tolist()
    normalized_probs = normalized_probs.float().cpu().numpy().tolist()
    return [
        {"prob": prob, "avg_loss": avg_loss}
        for prob, avg_loss in zip(normalized_probs, avg_losses)
    ]


def tokenwise_logprobs(model, batch, grad=False, return_labels=False):
    """
    Compute token-wise next token prediction logprobs for all labeled tokens for each sample in a batch.
    `grad` decides whether gradients are turned on
    Returns
    log_probs_batch (List[Tensor]): Tensors of size seq_len where seq_len is length of labeled tokens
    labels_batch (List[Tensor]): List of tensors of length N. Returned only if return_labels is True
    """
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.set_grad_enabled(grad):
        output = model(**batch)

    logits = output.logits
    bsz, seq_len, V = logits.shape
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[:, :-1, :]
    # ^ we don't predict next token for last token, bsz x seq_len-1 x V
    next_tokens = batch["input_ids"][:, 1:].unsqueeze(-1)  # bsz x seq_len-1 x 1
    target_log_probs = torch.gather(log_probs, dim=2, index=next_tokens).squeeze(-1)
    log_probs_batch = []
    labels_batch = []
    for i in range(bsz):
        labels = batch["labels"][i]
        # only focus on tokens which have loss on them (i.e. used in labels)
        actual_indices = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0][
            :-1
        ]  # -1 to ignore eos prediction
        num_actual_tokens = actual_indices.numel()
        if num_actual_tokens == 0:
            labels_batch.append(torch.tensor([], device=labels.device))
            log_probs_batch.append(torch.tensor([], device=labels.device))
            continue
        start_idx, end_idx = actual_indices[0].item(), actual_indices[-1].item()
        if start_idx == 0:
            warnings.warn(
                "Index 0 in a datapoint's input_ids must not have loss (unignored labels) on it",
                UserWarning,
            )
        log_probs_batch.append(target_log_probs[i, start_idx - 1 : end_idx])
        labels_batch.append(labels[actual_indices])

    return (log_probs_batch, labels_batch) if return_labels else log_probs_batch


def tokenwise_vocab_logprobs(model, batch, grad=False, return_labels=False):
    """Get vocabulary-wise log probabilities for each token in the sequence.

    Returns:
        log_probs_batch (List[Tensor]): List of tensors of shape (N, V) containing log probabilities
        for each sequence, where N is the length of labeled tokens and V is vocab size.
        labels_batch (List[Tensor]): List of tensors of length N. Returned only if return_labels is True
    """
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.set_grad_enabled(grad):
        output = model(**batch)

    logits = output.logits
    bsz, seq_len, V = logits.shape
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[
        :, :-1, :
    ]  # Don't predict for last token

    # Process each sequence in batch separately
    log_probs_batch = []
    labels_batch = []
    for i in range(bsz):
        labels = batch["labels"][i]
        # Only include positions that have labels
        actual_indices = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0][
            :-1
        ]  # -1 to ignore eos prediction
        if len(actual_indices) == 0:
            labels_batch.append(torch.tensor([], device=labels.device))
            log_probs_batch.append(torch.zeros(0, V, device=labels.device))
            continue
        start_idx, end_idx = actual_indices[0].item(), actual_indices[-1].item()
        if start_idx == 0:
            warnings.warn(
                "Index 0 in a datapoint's input_ids must not have loss (unignored labels) on it",
                UserWarning,
            )
        # Return full distribution for each position: shape (N, V)
        log_probs_batch.append(log_probs[i, start_idx - 1 : end_idx])
        labels_batch.append(labels[actual_indices])

    return (log_probs_batch, labels_batch) if return_labels else log_probs_batch


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


def eval_text_similarity(model, tokenizer, batch, generation_args: TrackingConfig):
    """Evaluate text similarity between model-generated outputs and ground truth using ROUGE scores."""

    def eval_rouge_recall_batch(gen_outputs, ground_truths):
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        evals = []
        for gen, gt in zip(gen_outputs, ground_truths):
            rouge_scores = scorer.score(gt, gen)
            evals.append(
                {
                    "rouge1_recall": rouge_scores["rouge1"].recall,
                    "rougeL_f1": rouge_scores["rougeL"].fmeasure,
                    "rougeL_recall": rouge_scores["rougeL"].recall,
                }
            )
        return evals

    batch = {k: v.to(model.device) for k, v in batch.items()}
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
    # generation_args = generation_args.cfg
    generation_args = OmegaConf.to_container(generation_args.cfg, resolve=True)
    stopwords = generation_args.pop("stopwords", None)
    if stopwords is not None:
        assert isinstance(stopwords, list)
        sc = stop_sequences_criteria(
            tokenizer, stopwords, input_ids.shape[1], input_ids.shape[0]
        )
        generation_args["stopping_criteria"] = sc
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        **generation_args,
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

    scores = eval_rouge_recall_batch(gen_texts, ground_truths)
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


def extract_target_texts_from_processed_data(tokenizer, batch):
    """Extract and detokenize text from activated positions in the batch."""
    labels = batch["labels"]
    labels = [elem[elem != -100] for elem in labels]
    texts = [
        tokenizer.decode(elem.tolist(), skip_special_tokens=True) for elem in labels
    ]
    return texts
