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
import re
import pandas as pd
from pathlib import Path
import logging
from typing import Any, Dict, List, Set, Tuple, Union, TYPE_CHECKING

from utils.common import load_logs

if TYPE_CHECKING:
    from os import PathLike

logger = logging.getLogger("vis.loader")

# Keys in log_history that are bookkeeping, not plottable series
_NON_METRIC_KEYS = {"step", "epoch", "loss"}


def _check_path(path: Path):
    return path if path.exists() else None


def glob_with_keys(root: Path, name_pattern: str):
    paths = list(root.glob(name_pattern))
    # regex for matching
    name_regex = re.escape(name_pattern).replace(r"\*", "(.*)", 1)
    name_regex = re.compile(f"^{name_regex}$")
    keys_path_list: List[tuple[str, Path]] = []
    for path in paths:
        m = name_regex.match(path.name)
        if m is None:
            continue
        keys_path_list.append((str(m.group(1)), path))
    return keys_path_list


class ExperimentLoader:
    _trainer_state: Dict[str, Any]
    _log_history: List[Dict[str, Any]]
    _log_history_df: pd.DataFrame
    _eval_summaries: Dict[str, Dict[int, Dict[str, Any]]]
    _eval_summaries_df: pd.DataFrame
    _eval_details: Dict[Tuple[int, str], Dict[str, Any]]
    _metric_keys: Set[str]
    _all_metric_keys: Set[str]

    def __init__(self, run_dir: Union[PathLike, str]):
        # run directory path
        _run_dir = Path(run_dir)
        if not _run_dir.is_dir():
            raise NotADirectoryError(f"Run directory not a existing directory: {_run_dir.resolve()}")
        self.run_dir = _run_dir
        # label
        self.label = _run_dir.name
        # trainer state path
        self.trainer_state_path = _check_path(_run_dir / "trainer_state.json")
        # checkpoint path
        self.step_ckp_paths = [(int(step), ckp_path) for step, ckp_path in glob_with_keys(_run_dir, "checkpoint-*")]

    def _load_trainer_state(self):
        """
        Load the trainer state and log history from `trainer_state.json`.
        """
        if self.trainer_state_path is not None:
            self._trainer_state = load_logs(self.trainer_state_path)
            self._log_history = self._trainer_state.get("log_history", [])
            # `log_history_df` only contains entries that have all NON_METRIC_KEYS.
            self._log_history_df = pd.DataFrame([
                entry for entry in self._log_history
                if _NON_METRIC_KEYS.issubset(entry.keys())
            ]).set_index(["step", "epoch"], drop=True).sort_index()
        else:
            logger.warning(f"No `trainer_state.json` found under {self.run_dir.resolve()}")
            self._trainer_state = {}
            self._log_history = []
            self._log_history_df = pd.DataFrame()

    def _load_summaries(self):
        """
        Load the evaluation summaries from `*_SUMMARY.json` files under the run directory.
        """
        self._eval_summaries = {}
        root_summary_paths = glob_with_keys(self.run_dir, "*_SUMMARY.json")
        if len(root_summary_paths) >= 1:
            # run-root SUMMARY
            self._update_step_dict(self._eval_summaries, root_summary_paths)
        else:
            # checkpoint SUMMARY
            for step, ckp_path in self.step_ckp_paths:
                ckp_summary_paths = glob_with_keys(ckp_path / "evals", "*_SUMMARY.json")
                if len(ckp_summary_paths) >= 1:
                    self._update_step_dict(self._eval_summaries, ckp_summary_paths, step=step)
                else:
                    logger.warning(f"No `SUMMARY.json` found for checkpoint {ckp_path.resolve()}")
        # eval_summaries_df
        self._eval_summaries_df = pd.DataFrame.from_dict(
            self._eval_summaries, orient="index"
        ).rename_axis(["step", "eval_name"]).sort_index()

    def _load_details(self):
        self._eval_details = {}
        root_detail_paths = list(self.run_dir.glob("*_EVAL.json"))
        if len(root_detail_paths) >= 1:
            # run-root EVAL
            detail = load_logs(root_detail_paths[0])
            self._eval_details[-1] = detail
            self._all_metric_keys = set(detail)
        else:
            # checkpoint EVAL
            self._all_metric_keys = set()
            for step, ckp_path in self.step_ckp_paths.items():
                ckp_detail_paths = list(ckp_path.glob(os.path.join("evals", "*_EVAL.json")))
                if len(ckp_detail_paths) == 0:
                    logger.warning(f"No `EVAL.json` found for checkpoint {str(ckp_path)}")
                elif len(ckp_detail_paths) >= 1:
                    detail = load_logs(ckp_detail_paths[0])
                    self._all_metric_keys.update(detail)
                    self._eval_details[step] = detail

    @staticmethod
    def _update_step_dict(
        step_dict: Dict[Tuple[int, str], Dict[str, Any]],
        name_paths: List[tuple[str, Path]],
        step: int = -1
    ):
        for name, path in name_paths:
            step_dict[(step, name)] = load_logs(path)

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
        return set(self.eval_summaries_df.columns)

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
