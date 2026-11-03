
from __future__ import annotations
import logging
from pathlib import Path
from functools import wraps
from typing import Callable, Any, Dict, List, Union, Optional, TYPE_CHECKING

from data import Data, SplitData
from utils.common import load_logs, get_func_params
from utils.config import reprlib

if TYPE_CHECKING:
    from utils.config import TrackingConfig

logger = logging.getLogger("eval.metric")

RUNTIME_PARAMS = ("model", "pre_compute", "output_dir")  # given at call time, see `Metric.evaluate`


class Metric:
    """A metric function + the configuration it is called with.

    The arguments of the function are prepared ONCE here, by the NAMES of its parameters:

    - `data` (one `Data`, the config `data` must have exactly one split) / `split_data` (the `SplitData` of all splits, any names),
      `reference_logs`, `tokenizer`, `template_args`, `metric_name`: prepared only when the function has a parameter of that name;
    - at call time: `model`, `pre_compute`, `output_dir`;
    - every other parameter is taken from the config as it is.

    A parameter WITHOUT a default is required: if it cannot be prepared (no config, or the preparation fails) it is an error.
    A parameter WITH a default is optional: the preparation is tried, and if it cannot be done it is a warning and the default of
    the function is used.
    """

    def __init__(
        self,
        name: str,
        cfg_dict: Dict[str, Union[Any, TrackingConfig]],
        func: MetricFunc,
        post_compute_name: str = "",
        pre_compute_metrics: Optional[Dict[str, Metric]] = None,
        **metric_kwargs: Any
    ):
        self.name = name
        self.full_name = f"`{name}`({func.name}){post_compute_name}"
        self.func = func
        self.pre_compute_metrics = pre_compute_metrics or {}
        self.cfg_dict = cfg_dict
        # scalars of this metric published as metrics of their own (`expose` in the config), see `ExposedMetric`
        self.exposed: List[ExposedMetric] = []

        logger.info(f"Preparing {self.full_name} ...")
        # the raw configs below are consumed here: the function only gets what is prepared from them, never the raw config
        data_cfg = self.cfg_dict.pop("data", None)
        loader_args = self.cfg_dict.pop("loader_args", None)
        ref_logs_cfg = self.cfg_dict.pop("reference_logs", None)
        try:
            self._prepare_data(data_cfg, {} if loader_args is None else loader_args, **metric_kwargs)
            self._prepare_param("reference_logs", lambda: self._load_reference_logs(ref_logs_cfg))
            self._prepare_param("tokenizer", lambda: self._from_kwargs("tokenizer", metric_kwargs))
            self._prepare_param("template_args", lambda: self._from_kwargs("template_args", metric_kwargs))
            self._prepare_param("metric_name", lambda: self.name)
            # the other parameters come from the config; at call time: model, pre_compute, output_dir
            missing = [
                param
                for param in self.func.required_params
                if param not in self.cfg_dict and param not in RUNTIME_PARAMS
            ]
            if missing:
                raise KeyError(f"Required [{len(missing)}] argument(s) {missing} not found in the configuration.")
        except Exception as e:
            raise RuntimeError(f"Error preparing arguments for {self.full_name} with {reprlib.repr(self.cfg_dict)}.") from e

    @staticmethod
    def _from_kwargs(param: str, metric_kwargs: Dict[str, Any]) -> Any:
        if param not in metric_kwargs:
            raise KeyError(f"`{param}` is not provided by the evaluator.")
        return metric_kwargs[param]

    def _prepare_param(self, param: str, prepare: Callable[[], Any]):
        """`cfg_dict[param] = prepare()` if the function has the parameter `param`; if `prepare` fails: an error for a required
        parameter, a warning (the default of the function is used) for an optional one."""
        if param not in self.func.all_params:
            return
        try:
            self.cfg_dict[param] = prepare()
        except Exception as e:
            if param in self.func.required_params:
                raise RuntimeError(f"Required argument `{param}` of {self.full_name} cannot be prepared.") from e
            self.cfg_dict.pop(param, None)
            logger.warning(
                f"Optional argument `{param}` of {self.full_name} cannot be prepared, keeping its default: "
                f"{type(e).__name__}: {e}"
            )

    def _prepare_data(self, data_cfg: Optional[TrackingConfig], loader_args: Any, **metric_kwargs: Any):
        """`split_data`: the `SplitData` of all splits of the config `data`; `data`: the `Data` of its only split.
        The datasets are loaded once, even if the function has both."""
        built: List[SplitData] = []

        def get_split_data() -> SplitData:
            if not built:
                if data_cfg is None:
                    raise ValueError("Configuration `data` is not found.")
                split_data = SplitData(data_cfg, **metric_kwargs)
                split_data.set_split_loader(**loader_args)
                built.append(split_data)
            return built[0]

        def get_data() -> Data:
            split_data = get_split_data()
            if len(split_data) != 1:
                raise ValueError(
                    f"Argument `data` needs exactly one split in the configuration `data`, got {split_data.splits}; "
                    "use the argument `split_data` for several."
                )
            return split_data[split_data.splits[0]]

        self._prepare_param("split_data", get_split_data)
        self._prepare_param("data", get_data)

    @staticmethod
    def _load_reference_logs(ref_logs_cfgs: Optional[TrackingConfig]) -> Dict[str, Any]:
        if ref_logs_cfgs is None:
            raise ValueError("Configuration `reference_logs` is not found.")
        ref_logs = {}
        for ref_log_name, ref_log_cfg in ref_logs_cfgs.items():
            path = ref_log_cfg.get("path", None)
            if path is None:
                logger.warning(f"Reference logs path for `{ref_log_name}` not found.")
                continue
            path = Path(path)
            if not path.exists():
                logger.warning(f"Reference logs path for `{ref_log_name}` doesn't exist: [{path.resolve()}].")
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
        return ref_logs

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
