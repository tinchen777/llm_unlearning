"""Loader for experiment outputs saved under a run directory.

A run directory (`saves/{mode}/{task_name}`, i.e. hydra/trainer `output_dir`)
may contain:

- `trainer_state.json` -- written by `trainer.save_state()`; its `log_history`
  holds training loss / lr / custom per-step losses logged via `trainer.log()`
  and the evaluator summary metrics logged after each eval.
- `checkpoint-{step}/evals/{BENCH}_EVAL.json` / `{BENCH}_SUMMARY.json` --
  written by `FinetuneTrainer.evaluate()` -> `Evaluator.evaluate()` during
  training (one folder per eval trigger, step 0 included when eval_on_start).
- `{BENCH}_EVAL.json` / `{BENCH}_SUMMARY.json` at the run root -- written by
  the standalone `src/eval.py` entry point (`saves/eval/{task_name}`).
"""

from __future__ import annotations
import os
import pandas as pd
from pathlib import Path
import logging
from typing import Any, Dict, List, Optional, Set

from utils.common import load_logs

logger = logging.getLogger("vis")

# Keys in log_history that are bookkeeping, not plottable series
_NON_METRIC_KEYS = {"step", "epoch", "loss"}


def _check_path(path: Path):
    return path if path.exists() else None


class ExperimentLoader:
    _trainer_state: Dict[str, Any]
    _log_history: List[Dict[str, Any]]
    _log_history_df: pd.DataFrame
    _eval_summaries: Dict[int, Dict[str, Any]]
    _eval_summaries_df: pd.DataFrame
    _eval_details: Dict[int, Dict[str, Any]]
    _metric_keys: Set[str]
    _all_metric_keys: Set[str]

    def __init__(self, run_dir: str):
        # run directory path
        _run_dir = Path(run_dir)
        if not _run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found or not a directory: {run_dir}")
        self._run_dir = _run_dir
        # label
        self.label = _run_dir.stem
        # trainer state path
        self.trainer_state_path = _check_path(_run_dir / "trainer_state.json")
        # checkpoint path
        self.ckp_paths = {int(ckp_path.name.split("-")[1]): ckp_path for ckp_path in self._run_dir.glob("checkpoint-*")}
        # cache

    def _load_trainer_state(self):
        """Load `trainer_state.json` from the run directory."""
        if self.trainer_state_path is not None:
            self._trainer_state = load_logs(self.trainer_state_path)
            self._log_history = self._trainer_state.get("log_history", [])
            self._log_history_df = pd.DataFrame([
                entry for entry in self._log_history
                if _NON_METRIC_KEYS.issubset(entry.keys())
            ]).set_index(["step", "epoch"], drop=True).sort_index()
        else:
            logger.warning(f"No `trainer_state.json` found under {str(self._run_dir)}")
            self._trainer_state = {}
            self._log_history = []
            self._log_history_df = pd.DataFrame()

    def _load_summaries(self):
        self._eval_summaries = {}
        root_summary_paths = list(self._run_dir.glob("*_SUMMARY.json"))
        if len(root_summary_paths) >= 1:
            # run-root SUMMARY
            summary = load_logs(root_summary_paths[0])
            self._eval_summaries[-1] = summary
            self._metric_keys = set(summary)
        else:
            # checkpoint SUMMARY
            self._metric_keys = set()
            for step, ckp_path in self.ckp_paths.items():
                ckp_summary_paths = list(ckp_path.glob(os.path.join("evals", "*_SUMMARY.json")))
                if len(ckp_summary_paths) == 0:
                    logger.warning(f"No `SUMMARY.json` found for checkpoint {str(ckp_path)}")
                elif len(ckp_summary_paths) >= 1:
                    summary = load_logs(ckp_summary_paths[0])
                    self._metric_keys.update(summary)
                    self._eval_summaries[step] = summary
        # eval_summaries_df
        self._eval_summaries_df = pd.DataFrame.from_dict(
            self._eval_summaries, orient="index"
        ).rename_axis("step").sort_index()

    def _load_details(self):
        self._eval_details = {}
        root_detail_paths = list(self._run_dir.glob("*_EVAL.json"))
        if len(root_detail_paths) >= 1:
            # run-root EVAL
            detail = load_logs(root_detail_paths[0])
            self._eval_details[-1] = detail
            self._all_metric_keys = set(detail)
        else:
            # checkpoint EVAL
            self._all_metric_keys = set()
            for step, ckp_path in self.ckp_paths.items():
                ckp_detail_paths = list(ckp_path.glob(os.path.join("evals", "*_EVAL.json")))
                if len(ckp_detail_paths) == 0:
                    logger.warning(f"No `EVAL.json` found for checkpoint {str(ckp_path)}")
                elif len(ckp_detail_paths) >= 1:
                    detail = load_logs(ckp_detail_paths[0])
                    self._all_metric_keys.update(detail)
                    self._eval_details[step] = detail

    @property
    def trainer_state(self):
        if not hasattr(self, "_trainer_state"):
            self._load_trainer_state()
        return self._trainer_state

    @property
    def log_history(self):
        if not hasattr(self, "_log_history"):
            self._load_trainer_state()
        return self._log_history

    @property
    def log_history_df(self):
        if not hasattr(self, "_log_history_df"):
            self._load_trainer_state()
        return self._log_history_df

    @property
    def train_keys(self):
        return set(self.log_history_df.columns)

    @property
    def eval_summaries(self):
        if not hasattr(self, "_eval_summaries"):
            self._load_summaries()
        return self._eval_summaries

    @property
    def eval_summaries_df(self):
        if not hasattr(self, "_eval_summaries_df"):
            self._load_summaries()
        return self._eval_summaries_df

    @property
    def eval_final_summaries(self):
        if len(self.eval_summaries) == 0:
            return {}
        return self.eval_summaries[max(self.eval_summaries)]

    @property
    def metric_keys(self):
        if not hasattr(self, "_metric_keys"):
            self._load_summaries()
        return self._metric_keys

    @property
    def eval_detail(self):
        if not hasattr(self, "_eval_details"):
            self._load_details()
        return self._eval_details

    @property
    def all_metric_keys(self):
        if not hasattr(self, "_all_metric_keys"):
            self._load_details()
        return self._all_metric_keys








# def collect_series(
#     log_history: List[Dict[str, Any]],
#     key: str
# ):
#     """Collect (`steps`, `values`) for a numeric key across log_history entries."""
#     steps = []
#     values = []
#     for entry in log_history:
#         val = entry.get(key)
#         if isinstance(val, (int, float)) and not isinstance(val, bool):
#             steps.append(entry.get("step", len(steps)))
#             values.append(float(val))
#     return steps, values


# def list_numeric_keys(log_history: List[Dict[str, Any]]) -> List[str]:
#     """All numeric keys that appear in log_history (bookkeeping excluded),
#     in first-appearance order."""
#     keys: List[str] = []
#     for entry in log_history:
#         for k, v in entry.items():
#             if k in _NON_METRIC_KEYS or k in keys:
#                 continue
#             if isinstance(v, (int, float)) and not isinstance(v, bool):
#                 keys.append(k)
#     return keys





# def load_eval_details(
#     run_dir: str, step: Optional[int] = None
# ) -> Dict[str, Any]:
#     """Fine-grained eval logs ({metric: {agg_value, value_by_index, ...}}).

#     `run_dir` may also directly be a path to an `*_EVAL.json` file.
#     `step=None` selects the last checkpoint; the run root is the fallback.
#     """
#     if os.path.isfile(run_dir):
#         return load_logs(run_dir)

#     by_step: Dict[int, List[str]] = {}
#     for path in glob.glob(os.path.join(run_dir, "checkpoint-*", "evals", "*_EVAL.json")):
#         m = _CKP_RE.search(os.path.dirname(os.path.dirname(path)))
#         if m:
#             by_step.setdefault(int(m.group(1)), []).append(path)
#     root_paths = glob.glob(os.path.join(run_dir, "*_EVAL.json"))
#     if root_paths:
#         by_step.setdefault(-1, []).extend(root_paths)

#     if not by_step:
#         logger.warning(f"No *_EVAL.json found under {run_dir}")
#         return {}
#     if step is None:
#         step = max(by_step)
#     elif step not in by_step:
#         raise FileNotFoundError(
#             f"No evals for step {step} under {run_dir}; available: {sorted(by_step)}"
#         )
#     details: Dict[str, Any] = {}
#     for path in by_step[step]:
#         details.update(load_logs(path))
#     return details


def extract_stat_values(
    eval_details: Dict[str, Any], metric: str, stat: Optional[str] = None
) -> List[float]:
    """Flatten per-sample values of `stat` from a metric's `value_by_index`.

    `stat=None` picks the first numeric field found (e.g. `prob` / `score`).
    Multi-answer entries (lists) are flattened; `None` scores are skipped.
    """
    try:
        value_by_index = eval_details[metric]["value_by_index"]
    except KeyError:
        raise KeyError(
            f"Metric `{metric}` (with value_by_index) not found; "
            f"available: {list(eval_details)}"
        )
    values: List[float] = []
    for sample_evals in value_by_index.values():
        if sample_evals is None:
            continue
        if stat is None:
            stat = next(
                (k for k, v in sample_evals.items()
                 if isinstance(v, (int, float)) or isinstance(v, list)),
                None,
            )
            if stat is None:
                continue
        val = sample_evals.get(stat)
        if isinstance(val, list):
            values.extend(float(v) for v in val if v is not None)
        elif val is not None:
            values.append(float(val))
    return values
