
from __future__ import annotations
from typing import Dict, TYPE_CHECKING

from .base import UnlearningMetric
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
    from .base import MetricFunc
    from utils.config import TrackingConfig

METRIC_FUNCS_REGISTRY: Dict[str, MetricFunc] = {}


def _register_metric_fn(metric_fn: MetricFunc):
    METRIC_FUNCS_REGISTRY[metric_fn.name] = metric_fn


def get_metrics(metric_cfgs: TrackingConfig, **kwargs):
    all_metrics: Dict[str, UnlearningMetric] = {}
    return {
        metric_name: _get_metric(metric_name, all_metrics, metric_cfg, **kwargs)
        for metric_name, metric_cfg in metric_cfgs.items()
    }


def _get_metric(
    metric_name: str,
    all_metrics: Dict[str, UnlearningMetric],
    metric_cfg: TrackingConfig,
    post_compute_name: str = "",
    **kwargs
):
    if metric_name in all_metrics:
        return all_metrics[metric_name]

    metric_cfg_dict = dict(metric_cfg)
    try:
        # get metric function
        handler = metric_cfg_dict.pop("handler", None)
        if handler is None:
            raise KeyError(f"Can not found `'handler'` in `@{metric_cfg.loc_choices}`.")
        try:
            metric_fn = METRIC_FUNCS_REGISTRY[handler]
        except KeyError:
            raise KeyError(f"`'handler'` `{handler}` is not registered.")
        # prepare pre-compute metrics
        pre_compute_cfgs = metric_cfg_dict.pop("pre_compute", {})
        pre_compute_metrics = {
            pre_compute_name: _get_metric(
                pre_compute_name,
                all_metrics,
                pre_compute_cfg,
                post_compute_name=f" of `{metric_name}`{post_compute_name}",
                **kwargs
            )
            for pre_compute_name, pre_compute_cfg in pre_compute_cfgs.items()
        }
        # create UnlearningMetric instance
        metric = UnlearningMetric(
            metric_name, metric_cfg_dict, metric_fn, post_compute_name, pre_compute_metrics, **kwargs
        )
        # update all_metrics
        all_metrics[metric_name] = metric
        return metric

    except Exception as e:
        raise RuntimeError(
            f"Error loading metric `{metric_name}`{post_compute_name} in `@{metric_cfg.loc_choices}`."
        ) from e


# Register metrics here
_register_metric_fn(probability)
_register_metric_fn(probability_w_options)
_register_metric_fn(rouge)
_register_metric_fn(truth_ratio)
_register_metric_fn(ks_test)
_register_metric_fn(hm_aggregate)
_register_metric_fn(privleak)
_register_metric_fn(rel_diff)
_register_metric_fn(exact_memorization)
_register_metric_fn(extraction_strength)

# Register MIA metrics
_register_metric_fn(mia_loss)
_register_metric_fn(mia_min_k)
_register_metric_fn(mia_min_k_plus_plus)
_register_metric_fn(mia_gradnorm)
_register_metric_fn(mia_zlib)
_register_metric_fn(mia_reference)

# Register Utility metrics
_register_metric_fn(classifier_prob)
