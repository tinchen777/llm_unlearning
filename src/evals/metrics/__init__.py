
from __future__ import annotations
from typing import Dict, TYPE_CHECKING

from .memorization import (
    probability,
    probability_w_options,
    rouge,
    truth_ratio,
    extraction_strength,
    exact_memorization,
)
from .privacy import ks_test, privleak, rel_diff
from .mia import (
    mia_loss,
    mia_min_k,
    mia_min_k_plus_plus,
    mia_gradnorm,
    mia_zlib,
    mia_reference,
)
from .utility import (
    hm_aggregate,
    classifier_prob,
)

if TYPE_CHECKING:
    from .base import UnlearningMetric
    from utils.config import TrackingConfig

METRICS_REGISTRY: Dict[str, UnlearningMetric] = {}


def _register_metric(metric):
    METRICS_REGISTRY[metric.name] = metric


def get_metrics(metric_cfgs: TrackingConfig):
    return {metric_name: _get_single_metric(metric_cfg)
            for metric_name, metric_cfg in metric_cfgs.items()}


def _get_single_metric(metric_cfg: TrackingConfig):
    metric_fn = METRICS_REGISTRY[metric_cfg["handler"]]
    pre_compute_cfgs = metric_cfg.get("pre_compute", None)
    if pre_compute_cfgs is not None:
        metric_fn.set_pre_compute_metrics(get_metrics(pre_compute_cfgs))
    return metric_fn


# Register metrics here
_register_metric(probability)
_register_metric(probability_w_options)
_register_metric(rouge)
_register_metric(truth_ratio)
_register_metric(ks_test)
_register_metric(hm_aggregate)
_register_metric(privleak)
_register_metric(rel_diff)
_register_metric(exact_memorization)
_register_metric(extraction_strength)

# Register MIA metrics
_register_metric(mia_loss)
_register_metric(mia_min_k)
_register_metric(mia_min_k_plus_plus)
_register_metric(mia_gradnorm)
_register_metric(mia_zlib)
_register_metric(mia_reference)

# Register Utility metrics
_register_metric(classifier_prob)
