
from __future__ import annotations
import math
import logging
from rouge_score import rouge_scorer
from typing import Any, List, Tuple, Mapping, Sequence, Optional, Dict, Callable

from utils.common import get_func_params

logger = logging.getLogger("eval.text_metric")

DEFAULT_REFUSAL_PATTERNS = (
    "i don't know", "i do not know", "i don't have", "i do not have", "i'm not aware", "i am not aware",
    "not familiar with", "no information", "i cannot", "i can't", "i'm not able", "i am not able",
    "i'm unable", "i am unable", "unable to provide", "i couldn't find", "i could not find",
    "i'm sorry", "i apologize", "not publicly available", "doesn't exist", "does not exist",
)
_ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


class TextMetric:
    @classmethod
    def use_all(cls, **all_args: Any):
        return cls(*list(TEXT_METRIC_FUNCS_REGISTRY), **all_args)

    def __init__(self, *fn_names: str, **all_args: Any):
        self.text_metric_funcs: List[Callable[..., Any]] = []
        self.text_metric_summaries: Dict[str, Callable[..., Any]] = {}
        for fn_name in fn_names:
            if fn_name not in TEXT_METRIC_FUNCS_REGISTRY:
                logger.warning(f"Text metric `{fn_name}` is not registered, skipping.")
                continue
            fn, params = TEXT_METRIC_FUNCS_REGISTRY[fn_name]
            func_dict = fn(**{param: all_args[param] for param in params if param in all_args})
            if "func" in func_dict:
                self.text_metric_funcs.append(func_dict["func"])
            if "summary" in func_dict:
                self.text_metric_summaries.update(func_dict["summary"])

    def compute_metrics(self, record: Mapping[str, Any]):
        metrics_record: Dict[str, Any] = {}
        for fn in self.text_metric_funcs:
            metrics_record.update(fn(record))
        return metrics_record

    def compute_summaries(self, records: Sequence[Mapping[str, Any]]):
        return {
            key: fn(records)
            for key, fn in self.text_metric_summaries.items()
        }


def _record_mean(record_key: str, ops: Optional[str] = None):
    def _mean(records: Sequence[Mapping[str, Any]]):
        values: List[Any] = []
        for record in records:
            if ops == "float":
                val = float(record.get(record_key, 0.0))
            elif ops == "exp":
                val = math.exp(float(record.get(record_key, 0.0)))
            else:
                val = record.get(record_key, False)
            values.append(val)
        return sum(values) / len(values) if values else 0.0
    return _mean


# TEXT METRICS FUNC

TEXT_METRIC_FUNCS_REGISTRY: Dict[str, Tuple[Callable[..., Dict[str, Any]], List[str]]] = {}


def _register_text_metric_fn(metric_fn: Callable[..., Dict[str, Any]]):
    TEXT_METRIC_FUNCS_REGISTRY[metric_fn.__name__.removesuffix("_func")] = (metric_fn, get_func_params(metric_fn))


def rouge_scores_func():
    def rouge_scores(record: Mapping[str, Any]):
        """
        Compute ROUGE scores for the generated text against the ground truth.
        `record` must be a dictionary containing `"ground_truth"`, and `"generation"`.
        """
        _rouge_scores = _ROUGE_SCORER.score(record["ground_truth"], record["generation"])
        return {
            "rouge1_recall": _rouge_scores["rouge1"].recall,
            "rougeL_f1": _rouge_scores["rougeL"].fmeasure,
            "rougeL_recall": _rouge_scores["rougeL"].recall
        }
    return {
        "func": rouge_scores,
        "summary": {
            "rouge1_recall": _record_mean("rouge1_recall", ops="float"),
            "rougeL_f1": _record_mean("rougeL_f1", ops="float"),
            "rougeL_recall": _record_mean("rougeL_recall", ops="float"),
        }
    }


def is_refusal_func(patterns: Sequence[str] = DEFAULT_REFUSAL_PATTERNS):
    def is_refusal(record: Mapping[str, Any]):
        """
        Check if the generated text is a refusal based on predefined patterns.
        `record` must be a dictionary containing `"generation"`.
        """
        text = record["generation"].lower().replace("’", "'")
        return {"refusal": any(p in text for p in patterns)}
    return {
        "func": is_refusal,
        "summary": {"refusal_rate": _record_mean("refusal", ops="float")}
    }


def is_degenerate_func(n: int = 3, min_distinct: float = 0.5):
    def is_degenerate(record: Mapping[str, Any]):
        """
        Check if the generated text is degenerate based on n-gram repetition and minimum distinctness.
        `record` must be a dictionary containing `"generation"`.
        """
        words = record["generation"].split()
        if not words:
            degenerate = True
        elif len(words) < 2 * n:
            degenerate = False
        else:
            grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
            degenerate = len(set(grams)) / len(grams) < min_distinct
        return {"degenerate": degenerate}
    return {
        "func": is_degenerate,
        "summary": {"degenerate_rate": _record_mean("degenerate", ops="float")}
    }


def is_hallucination_func(hallucination_rouge: float = 0.3):
    def is_hallucination(record: Mapping[str, Any]):
        """
        Check if the generated text is a hallucination based on ROUGE-L recall.
        `record` must be a dictionary containing `"rougeL_recall"`, `"refusal"`, and `"degenerate"`.
        """
        rougeL_recall = record.get("rougeL_recall", 0.0)
        return {"hallucination": (not record.get("refusal", False)) and (not record.get("degenerate", False)) and rougeL_recall < hallucination_rouge}
    return {
        "func": is_hallucination,
        "summary": {"hallucination_rate": _record_mean("hallucination")}
    }


def word_count_func():
    def word_count(record: Mapping[str, Any]):
        """
        Compute the word count of the generated text.
        `record` must be a dictionary containing `"generation"`.
        """
        text = record["generation"]
        return {"word_count": len(text.split())}
    return {
        "func": word_count,
        "summary": {"mean_words": _record_mean("word_count")}
    }


def answer_logprob_func():
    return {
        "summary": {
            "answer_logprob": _record_mean("answer_logprob", ops="float"),
            "answer_prob": _record_mean("answer_logprob", ops="exp")
        }
    }


_register_text_metric_fn(rouge_scores_func)
_register_text_metric_fn(is_refusal_func)
_register_text_metric_fn(is_degenerate_func)
_register_text_metric_fn(is_hallucination_func)
_register_text_metric_fn(word_count_func)
_register_text_metric_fn(answer_logprob_func)
