
from __future__ import annotations
import logging
import torch
import numpy as np
from functools import partial
from typing import Any, Dict, TYPE_CHECKING

from .utils import (
    aggregate_to_1D,
    evaluate_probability,
    eval_text_similarity,
    run_batchwise_evals,
    tokenwise_vocab_logprobs,
)
from .base import MetricFunc

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from utils.config import TrackingConfig

# Supress the info messages logged while calculating rouge using rouge_scorer
logging.getLogger("absl").setLevel(logging.WARNING)
logger = logging.getLogger("eval.metric")


@MetricFunc
def probability(model: Any, dataloader: DataLoader, **kwargs):
    """Compute the probabilities by data points and report aggregated average"""
    
    
    
    
    fun_args = {}
    scores_by_index = run_batchwise_evals(
        model, dataloader, evaluate_probability, fun_args, "Calculating loss"
    )
    prob_values = np.array(
        [
            evals["prob"]
            for evals in scores_by_index.values()
            if evals["prob"] is not None
        ]
    )
    prob_values = aggregate_to_1D(prob_values)
    return {"agg_value": np.mean(prob_values), "value_by_index": scores_by_index}


@MetricFunc
def probability_w_options(pre_compute: Dict[str, Any], **kwargs):
    """Normalize probabilities of correct answers against false answers for
    open-ended datasets, returning the aggregated value and per-index probabilities."""
    correct_answer_results = pre_compute["correct"]["value_by_index"]
    wrong_answer_results = pre_compute["wrong"]["value_by_index"]

    correct_indices = list(correct_answer_results.keys())
    wrong_indices = list(wrong_answer_results.keys())
    assert correct_indices == wrong_indices

    # Filter out None values from both correct and wrong answers
    filtered_indices = [
        idx
        for idx in correct_indices
        if correct_answer_results[idx] is not None
        and wrong_answer_results[idx] is not None
    ]
    correct = np.array(
        [correct_answer_results[idx]["prob"] for idx in filtered_indices]
    )
    all_wrong = np.array(
        [wrong_answer_results[idx]["prob"] for idx in filtered_indices]
    )
    wrong = np.sum(all_wrong, axis=tuple(range(1, all_wrong.ndim)))
    probs = correct / (correct + wrong + 1e-10)

    value_by_index = dict(zip(correct_indices, [{"prob": val} for val in probs]))
    return {"agg_value": np.mean(probs), "value_by_index": value_by_index}


@MetricFunc
def rouge(
    model: Any,
    tokenizer: Any,
    dataloader: DataLoader,
    generation_args: TrackingConfig,
    rouge_type: str,
    **kwargs
):
    """Calculate ROUGE metrics and return the aggregated value along with per-index scores."""
    scores_by_index = run_batchwise_evals(
        model,
        dataloader,
        eval_text_similarity,
        {"tokenizer": tokenizer, "generation_args": generation_args},
        "Calculating text similarity",
    )
    rouge_values = np.array(
        [
            evals[rouge_type]
            for evals in scores_by_index.values()
            if evals[rouge_type] is not None
        ]
    )
    rouge_values = aggregate_to_1D(rouge_values)
    return {
        "agg_value": np.mean(rouge_values),
        "value_by_index": scores_by_index,
    }


@MetricFunc
def truth_ratio(aggregator: str, pre_compute: Dict[str, Any], **kwargs):
    """Compute the truth ratio, aggregating false/true scores, and
    return the aggregated value."""
    aggregators = {
        # Forget data: It is better if false and true are equally likely,
        # i.e., tr=false/true is closest to 1.
        "closer_to_1_better": lambda arr: np.mean(np.minimum(arr, 1 / (arr + 1e-10))),

        # Non-forget data: It is better if tr=false/true is lower, i.e.,
        # 1-tr is higher.
        "true_better": lambda arr: np.mean(np.maximum(0, 1 - arr)),

        # Extent of knowledge (as used in OpenUnlearning paper's meta-evaluation) uses tr=true/(true+false)
        "prob_mean": lambda arr: np.mean(arr),
    }
    try:
        aggregator_fn = aggregators[aggregator]
    except KeyError:
        raise ValueError(f"Invalid truth ratio aggregator: {aggregator}")

    correct_answer_results = pre_compute["correct"]["value_by_index"]
    wrong_answer_results = pre_compute["wrong"]["value_by_index"]

    correct_indices = list(correct_answer_results.keys())
    wrong_indices = list(wrong_answer_results.keys())
    assert correct_indices == wrong_indices

    # Filter out None values from both correct and wrong answers
    filtered_indices = [
        idx
        for idx in correct_indices
        if correct_answer_results[idx] is not None
        and wrong_answer_results[idx] is not None
    ]
    correct_avg_losses = [
        correct_answer_results[idx]["avg_loss"] for idx in filtered_indices
    ]
    wrong_avg_losses = [
        wrong_answer_results[idx]["avg_loss"] for idx in filtered_indices
    ]

    correct_avg_losses = aggregate_to_1D(np.array(correct_avg_losses))
    wrong_avg_losses = aggregate_to_1D(np.array(wrong_avg_losses))

    correct_prob = np.exp(-correct_avg_losses)
    wrong_prob = np.exp(-wrong_avg_losses)

    if aggregator != "prob_mean":
        # Original definition from TOFU: wrong / correct
        truth_ratios = wrong_prob / (correct_prob + 1e-10)
    else:
        # New definition from OpenUnlearning: correct / (correct + wrong)
        truth_ratios = correct_prob / (correct_prob + wrong_prob + 1e-10)

    value_by_index = dict(
        zip(correct_indices, [{"score": val} for val in truth_ratios])
    )
    truth_ratio_stats = np.array([evals["score"] for evals in value_by_index.values()])
    forget_tr_avg = aggregator_fn(truth_ratio_stats)
    return {"agg_value": forget_tr_avg, "value_by_index": value_by_index}


@MetricFunc
def exact_memorization(model: Any, dataloader: DataLoader, **kwargs):

    def _exact_memorization(model, batch):
        log_probs_batch, labels_batch = tokenwise_vocab_logprobs(
            model, batch, grad=False, return_labels=True
        )
        em_batch = []
        for log_probs, labels in zip(log_probs_batch, labels_batch):
            valid_len = len(labels)
            if valid_len == 0:
                # Rarely, tokenization can result in a mismatch with no valid target
                # tokens for loss computation (see preprocess_chat_instance() for
                # reference). Since this condition makes no sense in terms of
                # computing EM, we just choose to set EM=None
                logger.warning(
                    "EM score for an instance is marked None, due to "
                    "tokenization issues that resulted in no valid target tokens."
                )
                em_batch.append({"score": None})
            else:
                preds = torch.argmax(log_probs, dim=-1)
                em_score = (preds == labels).sum() / valid_len
                em_batch.append({"score": em_score.item()})
        return em_batch

    fun_args = {}
    scores_by_index = run_batchwise_evals(
        model, dataloader, _exact_memorization, fun_args, "Calculating EM"
    )
    em_values = np.array(
        [
            evals["score"]
            for evals in scores_by_index.values()
            if evals["score"] is not None
        ]
    )
    em_values = aggregate_to_1D(em_values)
    return {"agg_value": np.mean(em_values), "value_by_index": scores_by_index}


@MetricFunc
def extraction_strength(model: Any, dataloader: DataLoader, **kwargs):

    def _extraction_strength(model, batch):
        log_probs_batch, labels_batch = tokenwise_vocab_logprobs(
            model, batch, grad=False, return_labels=True
        )
        es_batch = []
        for log_probs, labels in zip(log_probs_batch, labels_batch):
            valid_len = len(labels)
            preds = torch.argmax(log_probs, dim=-1)
            k = 0
            for k in range(valid_len):
                suff_preds = preds[k:]
                suff_labels = labels[k:]
                if torch.equal(suff_preds, suff_labels):
                    break
            if valid_len == 0:
                # Rarely, tokenization can result in a mismatch with no valid target
                # tokens for loss computation (see preprocess_chat_instance() for
                # reference). Since this condition makes no sense in terms of
                # computing ES, we just choose to set ES=None
                logger.warning(
                    "ES score for an instance is marked None, due to "
                    "tokenization issues that resulted in no valid target tokens."
                )
                es_batch.append({"score": 0})
            else:
                es_score = 1 - (k / valid_len)
                es_batch.append({"score": es_score})
        return es_batch

    fun_args = {}
    scores_by_index = run_batchwise_evals(
        model, dataloader, _extraction_strength, fun_args, "Calculating ES"
    )
    es_values = np.array(
        [
            evals["score"]
            for evals in scores_by_index.values()
            if evals["score"] is not None
        ]
    )
    es_values = aggregate_to_1D(es_values)
    return {"agg_value": np.mean(es_values), "value_by_index": scores_by_index}
