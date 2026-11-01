
from __future__ import annotations
import logging
from pathlib import Path
from functools import wraps
from typing import Callable, Any, Dict, List, Union, Optional, TYPE_CHECKING

from data import SplitData
from utils.common import load_logs, get_func_params
from utils.config import reprlib

if TYPE_CHECKING:
    from pathlib import Path
    from utils.config import TrackingConfig

logger = logging.getLogger("eval.metric")


class Metric:
    def __init__(
        self,
        name: str,
        cfg_dict: Dict[str, Union[Any, TrackingConfig]],
        func: MetricFunc,
        post_compute_name: str = "",
        pre_compute_metrics: Dict[str, Metric] = {},
        **kwargs
    ):
        self.name = name
        self.full_name = f"`{name}`({func.name}){post_compute_name}"
        self.func = func
        self.pre_compute_metrics = pre_compute_metrics
        self.cfg_dict = cfg_dict
        # scalars of this metric published as metrics of their own (`expose` in the config), see `ExposedMetric`
        self.exposed: List[ExposedMetric] = []

        logger.info(f"Preparing {self.full_name} ...")
        # prepare for data: `data` (a `Data`, the config has exactly ONE split) or `split_data` (a `SplitData`, any number)
        if "data" in self.func.params or "split_data" in self.func.params:
            self._prepare_data(**kwargs)
        # prepare for other required parameters
        for required_param in self.func.params:
            try:
                if required_param == "reference_logs":
                    self._prepare_reference_logs()
                elif required_param == "tokenizer":
                    self.cfg_dict["tokenizer"] = kwargs["tokenizer"]
                elif required_param == "metric_name":
                    self.cfg_dict["metric_name"] = self.name
                elif required_param not in ["model", "pre_compute", "output_dir"] and required_param not in self.cfg_dict:
                    raise KeyError(f"Required parameter `{required_param}` not found.")
            except Exception as e:
                raise RuntimeError(f"Error preparing `{required_param}` for {self.full_name} with {reprlib.repr(self.cfg_dict)}.") from e

    def _prepare_data(self, **kwargs):
        try:
            split_data = SplitData(self.cfg_dict.pop("data"), **kwargs).set_split_loader(**self.cfg_dict.pop("loader_args"))
            if "split_data" in self.func.params:
                self.cfg_dict["split_data"] = split_data
            if "data" in self.func.params:
                if len(split_data) != 1:
                    raise ValueError(f"`data` needs exactly one split in the config, got {split_data.splits}; use `split_data` for several.")
                self.cfg_dict["data"] = next(iter(split_data.values()))
        except Exception as e:
            raise RuntimeError(f"Error preparing data for {self.full_name} with {reprlib.repr(self.cfg_dict)}.") from e

    def _prepare_reference_logs(self):
        ref_logs = {}
        ref_logs_cfgs = self.cfg_dict.pop("reference_logs", {})
        for ref_log_name, ref_log_cfg in ref_logs_cfgs.items():
            path = ref_log_cfg.get("path", None)
            if path is None:
                logger.info(f"Reference logs path for `{ref_log_name}` not found.")
                continue
            else:
                path = Path(path)
                if not path.exists():
                    logger.warning(
                        f"Reference logs path for `{ref_log_name}` doesn't exist: {path.resolve()}."
                    )
                    continue
            # Load the reference logs
            logger.info(f"Loading reference logs from {path.resolve()} ...")
            _logs = load_logs(path)
            # for each include_cfg, load the corresponding logs
            ref_log = {}
            include_cfgs = ref_log_cfg.get("include", {}, check_none=True)
            for key, include_cfg in include_cfgs.items():
                access_name = include_cfg.get("access_key", key, check_none=True)
                _results = _logs.get(key, None)
                ref_log[access_name] = _results
                if _results is None:
                    logger.warning(
                        f"`{key}` evals not present in the {path.resolve()}, setting it to None, may result in error soon if code attempts to access."
                    )
            ref_logs[ref_log_name] = ref_log
        self.cfg_dict["reference_logs"] = ref_logs

    def _prepare_pre_compute(self, model: Any, cache: Dict[str, Dict[str, Any]]):
        self.cfg_dict["pre_compute"] = {
            access_name: pre_metric.evaluate(
                model=model,
                cache=cache,
                overwrite_cache=False
            )
            for access_name, pre_metric in self.pre_compute_metrics.items()
        }

    def _prepare_output_dir(self, output_dir: Optional[Path]):
        if "output_dir" in self.func.params:
            self.cfg_dict["output_dir"] = output_dir

    def evaluate(
        self,
        model: Any,
        cache: Dict[str, Dict[str, Any]],
        overwrite_cache: bool = False,
        output_dir: Optional[Path] = None
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
        self._prepare_output_dir(output_dir)
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


class ExposedMetric:
    """A scalar taken from the `summary` of another metric (`expose: [key, ...]` in its config).

    The parent is computed ONCE (its result keeps the per-sample `value_by_index` and the whole `summary`); every exposed
    key is a metric of its own, at the same level as the others: `{"agg_value": summary[key]}` in the logs and in the
    summary file, named `<expose_prefix or parent name>_<key>`.
    """

    def __init__(self, name: str, parent: Metric, key: str):
        self.name = name
        self.parent = parent
        self.key = key
        self.full_name = f"`{name}`(`{key}` of `{parent.name}`)"

    def evaluate(
        self,
        model: Any,
        cache: Dict[str, Dict[str, Any]],
        overwrite_cache: bool = False,
        output_dir: Optional[Path] = None
    ):
        if self.name in cache and not overwrite_cache:
            logger.info(f"Skipping {self.full_name}, already evaluated.")
            return cache[self.name]
        # never recompute the parent here: it is evaluated (and overwritten) at its own turn, just before its exposed metrics
        parent_results = self.parent.evaluate(model, cache, overwrite_cache=False, output_dir=output_dir)
        summary = parent_results.get("summary") or {}
        if self.key not in summary:
            raise KeyError(f"`{self.key}` is not in the summary of `{self.parent.name}`, available: {sorted(summary)}.")
        cache[self.name] = {"agg_value": summary[self.key]}
        return cache[self.name]

    def __repr__(self) -> str:
        return f"{type(self).__name__} {self.name}"


class MetricFunc:
    def __init__(self, func: Callable[..., Dict[str, Any]]):
        self.func = func
        self.name = str(func.__name__)
        self.params = get_func_params(func)

        wraps(func)(self)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
