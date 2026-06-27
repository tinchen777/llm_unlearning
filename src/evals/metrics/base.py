
from __future__ import annotations
import os
import logging
from typing import Callable, Any, Dict, List, Optional, TYPE_CHECKING

from data import get_datasets, get_collators
from utils.common import load_logs_from_file

if TYPE_CHECKING:
    from utils.config import TrackingConfig

logger = logging.getLogger("metrics")


class UnlearningMetric:
    def __init__(
        self,
        name: str,
        metric_fn: Callable[..., Dict[str, Any]],
    ):
        self.name = name
        self.metric_fn = metric_fn
        self.pre_compute_metrics: Dict[str, UnlearningMetric] = {}

    def set_pre_compute_metrics(self, metrics: Dict[str, UnlearningMetric]):
        self.pre_compute_metrics.update(metrics)

    def _prepare_data(
        self,
        metric_cfg: TrackingConfig,
        tokenizer: Optional[Any],
        template_args: Optional[TrackingConfig]
    ):
        dataset_cfgs = metric_cfg.get("datasets", None, allow_none=True)
        if dataset_cfgs is not None:
            return get_datasets(
                dataset_cfgs,
                tokenizer=tokenizer,
                template_args=template_args
            )

    def _prepare_collators(
        self,
        metric_cfg: TrackingConfig,
        tokenizer: Optional[Any]
    ):
        collator_cfgs = metric_cfg.get("collators", None, allow_none=True)
        if collator_cfgs is not None:
            return get_collators(
                collator_cfgs,
                tokenizer=tokenizer
            )

    def _prepare_pre_compute(
        self,
        metric_cfg: TrackingConfig,
        cache: Dict[str, Dict[str, Any]],
        model: Any,
        post_compute_name: str = "",
        **kwargs
    ):
        pre_metric_results = {}
        pre_compute_cfgs = metric_cfg.get("pre_compute", None, allow_none=True)
        if pre_compute_cfgs is not None and len(pre_compute_cfgs) > 0:
            for pre_metric_name, pre_metric_cfg in pre_compute_cfgs.items():
                # pre-compute metric function
                pre_metric_fn = self.pre_compute_metrics.get(pre_metric_name, None)
                if pre_metric_fn is None:
                    raise ValueError(f"Can not find pre-compute metric `{pre_metric_name}` for `{self.name}`")
                # pre-compute metric evaluation
                _results = pre_metric_fn.evaluate(
                    metric_cfg,
                    cache,
                    model,
                    overwrite_cache=False,
                    post_compute_name=f" for `{self.name}`{post_compute_name}",
                    **kwargs
                )
                # update
                access_name = pre_metric_cfg.get("access_key", pre_metric_name)
                pre_metric_results[access_name] = _results
        return pre_metric_results

    def _prepare_reference_logs(self, metric_cfg: TrackingConfig):
        ref_logs = {}
        ref_logs_cfgs = metric_cfg.get("reference_logs", None)
        if ref_logs_cfgs is not None and len(ref_logs_cfgs) > 0:
            for ref_log_name, ref_log_cfg in ref_logs_cfgs.items():
                path = ref_log_cfg.get("path", None)
                if path is None:
                    continue

                # Load the reference logs
                if os.path.exists(path):
                    logger.info(f"Loading reference logs from {path} ...")
                    _logs = load_logs_from_file(path)
                else:
                    raise ValueError(f"Reference logs {path} doesn't exist!")
                # for each include_cfg, load the corresponding logs
                ref_log = {}
                include_cfgs = ref_log_cfg.get("include", None)
                if include_cfgs is not None and len(include_cfgs) > 0:
                    for key, include_cfg in include_cfgs.items():
                        access_name = include_cfg.get("access_key", key)
                        _results = _logs.get(key, None)
                        ref_log[access_name] = _results
                        if _results is None:
                            logger.warning(
                                f"`{key}` evals not present in the {path}, setting it to None, may result in error soon if code attempts to access."
                            )
                ref_logs[ref_log_name] = ref_log
        return ref_logs

    def evaluate(
        self,
        metric_cfg: TrackingConfig,
        cache: Dict[str, Dict[str, Any]],
        model: Any,
        overwrite_cache: bool = False,
        post_compute_name: str = "",
        tokenizer: Optional[Any] = None,
        template_args: Optional[TrackingConfig] = None
    ):
        """Evaluates a metric including its pre_compute metrics"""
        # metric_full_name
        metric_full_name = f"`{self.name}`{post_compute_name}"

        if self.name in cache and not overwrite_cache:
            # skip evaluation and computation
            logger.info(f"Skipping {metric_full_name}, already evaluated.")
            return cache[self.name]
        # start evaluation and computation
        if overwrite_cache:
            cache.pop(self.name, None)
        logger.info(f"Preparing {metric_full_name} ...")
        metric_kwargs = {
            "data": self._prepare_data(metric_cfg, tokenizer, template_args),
            "collators": self._prepare_collators(metric_cfg, tokenizer),
            "pre_compute": self._prepare_pre_compute(
                metric_cfg,
                cache,
                model,
                post_compute_name,
                tokenizer=tokenizer,
                template_args=template_args,
            ),
            "reference_logs": self._prepare_reference_logs(metric_cfg)
        }
        logger.info(f"Evaluating & computing {metric_full_name} ...")
        results = self.metric_fn(model, **metric_kwargs)
        cache[self.name] = results

        return results

    def __call__(self, model: Any, **kwargs):
        return self.evaluate(model, **kwargs)

    def __repr__(self) -> str:
        return f"{type(self).__name__} {self.name}"


# decorator that wraps simple user-defined metric python functions into callable UnlearningMetric objects
class unlearning_metric:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, metric_fn: Callable[..., Dict[str, Any]]) -> UnlearningMetric:
        return UnlearningMetric(
            name=self.name or metric_fn.__name__,
            metric_fn=metric_fn
        )
