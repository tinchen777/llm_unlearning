
from __future__ import annotations
import logging
from pathlib import Path
from functools import wraps
from typing import Callable, Any, Dict, List, Union, Optional, TYPE_CHECKING

from data import SplitData
from utils.common import load_logs, get_func_params
from utils.config import reprlib

if TYPE_CHECKING:
    from utils.config import TrackingConfig

logger = logging.getLogger("eval.metric")

SPLITDATA_SUFFIX = "_data"


class Metric:
    def __init__(
        self,
        name: str,
        cfg_dict: Dict[str, Union[Any, TrackingConfig]],
        func: MetricFunc,
        post_compute_name: str = "",
        pre_compute_metrics: Dict[str, Metric] = {},
        **metric_kwargs: Any
    ):
        self.name = name
        self.full_name = f"`{name}`({func.name}){post_compute_name}"
        self.func = func
        self.pre_compute_metrics = pre_compute_metrics
        self.cfg_dict = cfg_dict
        # scalars of this metric published as metrics of their own (`expose` in the config), see `ExposedMetric`
        self.exposed: List[ExposedMetric] = []

        logger.info(f"Preparing {self.full_name} ...")
        # prepare data
        self._prepare_data(data_cfg=self.cfg_dict.get("data", None), **metric_kwargs)
        # prepare reference logs
        self._prepare_reference_logs(self.cfg_dict.get("reference_logs", None))

        try:
            # prepare tokenizer and template_args
            if "tokenizer" in self.func.all_params:
                self.cfg_dict["tokenizer"] = metric_kwargs["tokenizer"]
            if "template_args" in self.func.all_params:
                self.cfg_dict["template_args"] = metric_kwargs["template_args"]
            if "metric_name" in self.func.all_params:
                self.cfg_dict["metric_name"] = self.name
            # check required parameters
            other_required_params = [
                required_param
                for required_param in self.func.required_params
                if required_param not in self.cfg_dict and required_param not in ["model", "pre_compute", "output_dir"]
            ]
            if other_required_params:
                raise KeyError(f"Required [{len(other_required_params)}] argument(s) {other_required_params} not found.")
        except Exception as e:
            raise RuntimeError(f"Error preparing arguments for {self.full_name} with {reprlib.repr(self.cfg_dict)}.") from e

    def _prepare_data(self, data_cfg: Optional[TrackingConfig], **metric_kwargs: Any):
        splitdata_params = [
            params
            for params in self.func.all_params
            if params.endswith(SPLITDATA_SUFFIX)
        ]

        if not splitdata_params and "data" not in self.func.all_params:
            return

        try:
            # prepare split data
            if data_cfg is None:
                raise ValueError("Configuration `data` is required.")

            split_data = SplitData(data_cfg, **metric_kwargs)
            split_data.set_split_loader(**self.cfg_dict.pop("loader_args", {}))

            if "data" in self.func.all_params:
                # for single data
                first_split = split_data.splits[0]
                if len(split_data) > 1:
                    logger.warning(f"Multiple splits of data found for metric `{self.name}` to argument `data`, using the first split `{first_split}`")
                self.cfg_dict["data"] = split_data[first_split]

            # for mulit-data, like *_data
            for splitdata_param in splitdata_params:
                split_name = splitdata_param.replace(SPLITDATA_SUFFIX, "")
                if split_name not in split_data.splits:
                    if splitdata_param in self.func.required_params:
                        raise ValueError(f"Required argument `{splitdata_param}`, but split `{split_name}` not found in the data.")
                    else:
                        logger.warning(f"Optional argument `{splitdata_param}`, split `{split_name}` not found in the data for metric `{self.name}`, keeping default.")
                        continue
                self.cfg_dict[splitdata_param] = split_data[split_name]
        except Exception as e:
            raise RuntimeError(f"Error preparing data for {self.full_name} with {reprlib.repr(self.cfg_dict)}.") from e

    def _prepare_reference_logs(self, ref_logs_cfgs: Optional[TrackingConfig] = None):
        if "reference_logs" not in self.func.all_params:
            return
        if "reference_logs" in self.func.optional_params and ref_logs_cfgs is None:
            logger.warning(f"Optional argument `reference_logs`, no configuration `reference_logs` for metric `{self.name}`, keeping default.")
            return

        ref_logs = {}
        try:
            if ref_logs_cfgs is None:
                raise ValueError("Configuration `reference_logs` is required.")

            for ref_log_name, ref_log_cfg in ref_logs_cfgs.items():
                path = ref_log_cfg.get("path", None)
                if path is None:
                    logger.warning(f"Reference logs path for `{ref_log_name}` not found.")
                    continue
                else:
                    path = Path(path)
                    if not path.exists():
                        logger.warning(
                            f"Reference logs path for `{ref_log_name}` doesn't exist: [{path.resolve()}]."
                        )
                        continue
                # Load the reference logs
                _logs = load_logs(path)
                logger.info(f"Loaded reference logs from [{path.resolve()}]")
                # for each include_cfg, load the corresponding logs
                ref_log = {}
                include_cfgs = ref_log_cfg.get("include", {}, check_none=True)
                for key, include_cfg in include_cfgs.items():
                    access_name = include_cfg.get("access_key", key, check_none=True)
                    _results = _logs.get(key, None)
                    ref_log[access_name] = _results
                    if _results is None:
                        logger.warning(
                            f"`{key}` evals not present in [{path.resolve()}], setting it to `None`, may result in error soon if code attempts to access."
                        )
                ref_logs[ref_log_name] = ref_log
            self.cfg_dict["reference_logs"] = ref_logs
        except Exception as e:
            raise RuntimeError(f"Error preparing reference logs for {self.full_name} with {reprlib.repr(self.cfg_dict)}.") from e

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
        if "output_dir" in self.func.all_params:
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
        if "model" in self.func.all_params:
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

# TODO
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
        params_dict = get_func_params(func)
        self.all_params = params_dict["all"]
        self.required_params = params_dict["required"]
        self.optional_params = params_dict["optional"]

        wraps(func)(self)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
