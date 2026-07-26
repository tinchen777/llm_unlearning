
from __future__ import annotations
from pathlib import Path
import logging
from cobra_color import cstr
from typing import Any, Optional, Dict, TYPE_CHECKING

from .metrics import get_metrics
from utils.common import load_logs, save_logs

if TYPE_CHECKING:
    from utils.config import TrackingConfig

logger = logging.getLogger("eval")


class Evaluator:
    def __init__(
        self,
        name: str,
        eval_cfg: TrackingConfig,
        **kwargs
    ):
        self.name = name
        self.overwrite = bool(eval_cfg.get("overwrite", True))
        self.output_dir = str(eval_cfg.get("output_dir", ""))
        if self.output_dir:
            logger.info(
                f"Evaluations of `<{self.name}>` stored in: {self.output_dir}"
            )
        self.init_base(eval_cfg, **kwargs)

    def init_base(self, eval_cfg: TrackingConfig, **kwargs):
        self.metrics_dict = get_metrics(
            eval_cfg.get("metrics", {}, allow_none=True),
            **kwargs
        )

    def get_logs_file_path(self, output_dir: str, suffix: str):
        """Returns the path to json file to store results"""
        if not output_dir:
            return None
        return Path(output_dir) / f"{self.name}_{suffix}.json"

    def summarize(self, logs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize the metrics results"""
        metric_summary = {}
        for metric_name, metric_results in logs.items():
            if metric_name not in self.metrics_dict:
                continue
            agg_value = metric_results.get("agg_value", None)
            if agg_value is not None:
                metric_summary[metric_name] = agg_value
        return metric_summary

    def evaluate(
        self,
        model: Any,
        output_dir: Optional[str] = None,
        overwrite: Optional[bool] = None
    ):
        # set flag to overwrite metrics
        _overwrite = self.overwrite if overwrite is None else overwrite

        # Prepare model for evaluation
        model.eval()

        # Set output_dir and file to store results
        _output_dir = self.output_dir if output_dir is None else output_dir
        eval_detail_path = self.get_logs_file_path(_output_dir, suffix="EVAL")
        eval_summary_path = self.get_logs_file_path(_output_dir, suffix="SUMMARY")

        # Load existing results from file if any.
        if eval_detail_path and eval_detail_path.exists() and not _overwrite:
            logs = load_logs(eval_detail_path)
            logger.info(f"Loading existing evaluations from: {str(eval_detail_path)}")
        else:
            logs = {}

        logger.info(f"=== Running `<{self.name}>` evaluation suite ===")
        if eval_detail_path:
            logger.info(f"Fine-grained evaluations will be saved to: {str(eval_detail_path)}")
        if eval_summary_path:
            logger.info(f"Aggregated evaluations will be summarised in: {str(eval_summary_path)}")
        print("-" * 80)

        for idx, (metric_name, metric_fn) in enumerate(self.metrics_dict.items(), start=1):
            idx_str = f"[{idx}/{len(self.metrics_dict)}]"
            logger.info(f"{cstr(idx_str, fg='y')} Evaluating metric `{metric_name}` ...")
            _results = metric_fn.evaluate(model, logs, overwrite_cache=_overwrite)
            # Update logs
            if eval_detail_path:
                save_logs(logs, eval_detail_path)
            if eval_summary_path:
                save_logs(self.summarize(logs), eval_summary_path)
            logger.info(
                f"{cstr(idx_str, fg='g')} Finished evaluating metric `{metric_name}`, "
                f"agg_value: {_results.get('agg_value', 'N/A')}."
            )
            print("-" * 80)

        return self.summarize(logs)
