
from __future__ import annotations
import os
import logging
import inspect
from functools import wraps
from torch.utils.data import DataLoader
from typing import Callable, Any, Dict, Union, Optional, TYPE_CHECKING

from .utils import DATA_SPLIT_SUFFIX
from data import get_datasets, get_collators
from utils.common import load_logs_from_file
from utils.config import reprlib

if TYPE_CHECKING:
    from utils.config import TrackingConfig

logger = logging.getLogger("eval.metric")


class UnlearningMetric:
    def __init__(
        self,
        name: str,
        cfg_dict: Dict[str, Union[Any, TrackingConfig]],
        func: MetricFunc,
        post_compute_name: str = "",
        pre_compute_metrics: Dict[str, UnlearningMetric] = {},
        **kwargs
    ):
        self.name = name
        self.full_name = f"`{name}`({func.name}){post_compute_name}"
        self.func = func
        self.pre_compute_metrics = pre_compute_metrics
        self.cfg_dict = cfg_dict

        logger.info(f"Preparing {self.full_name} ...")
        # prepare for dataloaders
        self._prepare_dataloaders(**kwargs)
        # prepare for other required parameters
        for required_param in self.func.params:
            try:
                if required_param == "reference_logs":
                    self._prepare_reference_logs()
                elif required_param == "tokenizer":
                    self.cfg_dict["tokenizer"] = kwargs["tokenizer"]
                elif required_param not in ["model", "pre_compute"] and required_param not in self.cfg_dict:
                    raise KeyError(f"Required parameter `{required_param}` not found.")
            except Exception as e:
                raise RuntimeError(f"Error preparing `{required_param}` for {self.full_name} with {reprlib.repr(self.cfg_dict)}.") from e

    def _prepare_dataloaders(self, **kwargs):
        subdata_params = [
            params
            for params in self.func.params
            if params.endswith(DATA_SPLIT_SUFFIX)
        ]
        if subdata_params or "dataloader" in self.func.params:
            try:
                # prepare data
                data = self._prepare_data(**kwargs)
                # prepare collator
                collator = self._prepare_collator(**kwargs)
                if isinstance(collator, dict):
                    logger.warning(f"Got multiple({len(collator)}) collators, using the first one for dataloader.")
                    collator = next(iter(collator.values()))
                # prepare batch_size
                batch_size = self.cfg_dict.pop("batch_size")

                if "dataloader" in self.func.params:
                    # for the whole dataset
                    self.cfg_dict["dataloader"] = DataLoader(
                        data,  # type: ignore
                        batch_size=batch_size,  # type: ignore
                        collate_fn=collator
                    )
                for subdata_param in subdata_params:
                    # for each subdata split
                    split_name = subdata_param.replace(DATA_SPLIT_SUFFIX, "")
                    if split_name not in data:
                        raise KeyError(f"Required data split `{split_name}` not found for `{subdata_param}`.")
                    self.cfg_dict[subdata_param] = DataLoader(
                        data[split_name],
                        batch_size=batch_size,  # type: ignore
                        collate_fn=collator
                    )
            except Exception as e:
                raise RuntimeError(f"Error preparing dataloaders for {self.full_name} with {reprlib.repr(self.cfg_dict)}.") from e

    def _prepare_data(
        self,
        tokenizer: Optional[Any],
        template_args: Optional[TrackingConfig],
        **kwargs
    ):
        return get_datasets(
            self.cfg_dict.pop("datasets"),
            tokenizer=tokenizer,
            template_args=template_args
        )

    def _prepare_collator(self, tokenizer: Optional[Any], **kwargs):
        return get_collators(
            self.cfg_dict.pop("collators"),
            tokenizer=tokenizer
        )

    def _prepare_reference_logs(self):
        ref_logs = {}
        ref_logs_cfgs = self.cfg_dict.pop("reference_logs", {})
        for ref_log_name, ref_log_cfg in ref_logs_cfgs.items():
            path = ref_log_cfg.get("path", None, allow_none=True)
            if path is None or not os.path.exists(path):
                logger.warning(
                    f"Reference logs path for `{ref_log_name}` not found or doesn't exist."
                )
                continue
            # Load the reference logs
            logger.info(f"Loading reference logs from {path} ...")
            _logs = load_logs_from_file(path)
            # for each include_cfg, load the corresponding logs
            ref_log = {}
            include_cfgs = ref_log_cfg.get("include", {}, allow_none=True)
            for key, include_cfg in include_cfgs.items():
                access_name = include_cfg.get("access_key", key, allow_none=True)
                _results = _logs.get(key, None)
                ref_log[access_name] = _results
                if _results is None:
                    logger.warning(
                        f"`{key}` evals not present in the {path}, setting it to None, may result in error soon if code attempts to access."
                    )
            ref_logs[ref_log_name] = ref_log
        self.cfg_dict["reference_logs"] = ref_logs

    def _prepare_pre_compute(self, model: Any, cache: Dict[str, Dict[str, Any]]):
        pre_metric_results = {}
        for pre_metric_name, pre_metric in self.pre_compute_metrics.items():
            access_name = pre_metric.cfg_dict.get("access_key", pre_metric_name)
            pre_metric_results[access_name] = pre_metric.evaluate(
                model=model,
                cache=cache,
                overwrite_cache=False
            )
        self.cfg_dict["pre_compute"] = pre_metric_results

    def evaluate(
        self,
        model: Any,
        cache: Dict[str, Dict[str, Any]],
        overwrite_cache: bool = False
    ):
        """Evaluates a metric including its pre_compute metrics"""
        if self.name in cache and not overwrite_cache:
            # skip evaluation and computation
            logger.info(f"Skipping {self.full_name}, already evaluated.")
            return cache[self.name]

        if overwrite_cache:
            cache.pop(self.name, None)
        # prepare pre-compute metrics
        self._prepare_pre_compute(model, cache)
        # prepare model
        if "model" in self.func.params:
            self.cfg_dict["model"] = model
        # start evaluation and computation
        logger.info(f"Evaluating & computing {self.full_name} ...")
        try:
            results = self.func(**self.cfg_dict)
        except Exception as e:
            raise RuntimeError(f"Error evaluating {self.full_name} with {reprlib.repr(self.cfg_dict)}.") from e
        # update cache
        cache[self.name] = results

        return results

    def __repr__(self) -> str:
        return f"{type(self).__name__} {self.name}"


ARGS_KWARGS = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


class MetricFunc:
    def __init__(self, func: Callable[..., Dict[str, Any]]):
        self.func = func
        self.name = str(func.__name__)
        self.signature = inspect.signature(func)
        self.params = [
            name
            for name, p in self.signature.parameters.items()
            if p.kind not in ARGS_KWARGS
        ]

        # 保留原函数属性
        wraps(func)(self)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
