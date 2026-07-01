
from __future__ import annotations
import numpy as np
from scipy.stats import ks_2samp
from typing import Any, Dict

from .base import MetricFunc, logger


@MetricFunc
def ks_test(pre_compute: Dict[str, Any], reference_logs: Dict[str, Any], **kwargs):
    """Compare two forget and retain model distributions with a 2-sample KS-test and report the p-value.
    Used in the TOFU benchmark as forget_quality when computed over the truth_ratio statistic."""
    if reference_logs:
        forget_tr_stats = np.array([
            evals["score"] for evals in pre_compute["forget"]["value_by_index"].values()
        ])
        retain_tr_stats = np.array([
            evals["score"]
            for evals in reference_logs["retain_model_logs"]["retain"]["value_by_index"].values()
        ])
        fq = ks_2samp(forget_tr_stats, retain_tr_stats)
        pvalue = fq.pvalue
    else:
        logger.warning(
            "`retain_model_logs` evals not provided for `ks_test`, setting forget_quality to `None`."
        )
        pvalue = None
    return {"agg_value": pvalue}


@MetricFunc
def privleak(pre_compute: Dict[str, Any], reference_logs: Dict[str, Any], ref_value: float, **kwargs):
    """Compare two forget and retain model scores using a relative comparison of a single statistic.
    To be used for MIA AUC scores in ensuring consistency and reproducibility of the MUSE benchmark.
    This function is similar to the rel_diff function below, but due to the MUSE benchmark reporting AUC
    scores as (1-x) when the more conventional way is x, we do adjustments here to our MIA AUC scores.
    calculations in the reverse way,"""
    score = pre_compute["forget"]["agg_value"]
    try:
        ref = reference_logs["retain_model_logs"]["retain"]["agg_value"]
    except Exception as _:
        logger.warning(
            f"`retain_model_logs` evals not provided for `privleak`, using default retain auc of `{ref_value}`."
        )
        ref = ref_value
    score = 1 - score
    ref = 1 - ref
    return {"agg_value": (score - ref) / (ref + 1e-10) * 100}


@MetricFunc
def rel_diff(pre_compute: Dict[str, Any], reference_logs: Dict[str, Any], ref_value: float, **kwargs):
    """Compare two forget and retain model scores using a relative comparison of a single statistic."""
    score = pre_compute["forget"]["agg_value"]
    try:
        ref = reference_logs["retain_model_logs"]["retain"]["agg_value"]
    except Exception as _:
        logger.warning(
            f"`retain_model_logs` evals not provided for `rel_diff`, using default retain auc of `{ref_value}`."
        )
        ref = ref_value
    return {"agg_value": (score - ref) / (ref + 1e-10) * 100}
