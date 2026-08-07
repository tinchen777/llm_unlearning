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
import re
import pandas as pd
from pathlib import Path
import logging
from typing import Any, Dict, List, Union, TYPE_CHECKING

from utils.common import load_logs

if TYPE_CHECKING:
    from os import PathLike

logger = logging.getLogger("vis.loader")

# Keys in log_history that are bookkeeping, not plottable series
_NON_METRIC_KEYS = {"step", "epoch", "loss"}


def _check_path(path: Path):
    return path if path.exists() else None


def glob_with_keys(root: Path, name_pattern: str):
    """
    Glob for files under `root` matching `name_pattern`, and extract the key from the filename using the `*` in `name_pattern`.
    NOTE: Only the first `*` in `name_pattern` is used for key extraction.
    """
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
    _log_history_df: pd.DataFrame
    _eval_summaries_dfs: Dict[str, pd.DataFrame]
    _eval_details_dfs: Dict[str, pd.DataFrame]

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
        # load trainer state
        if self.trainer_state_path is not None:
            self._trainer_state = load_logs(self.trainer_state_path)
            # load log history
            _log_history: List[Dict[str, Any]] = self._trainer_state.get("log_history", [])
            # create log_history_df, which only contains entries that have all NON_METRIC_KEYS.
            self._log_history_df = pd.DataFrame([
                entry for entry in _log_history
                if _NON_METRIC_KEYS.issubset(entry.keys())
            ]).set_index(["step", "epoch"], drop=True).sort_index()
        else:
            logger.warning(f"No `trainer_state.json` found under {self.run_dir.resolve()}")
            self._trainer_state = {}
            self._log_history_df = pd.DataFrame()

    def _load_summaries(self):
        """
        Load the evaluation summaries from `*_SUMMARY.json` files under the run directory.
        """
        _named_eval_summaries: Dict[str, Dict[int, Dict[str, Any]]] = {}
        # load eval_summaries
        root_summary_paths = glob_with_keys(self.run_dir, "*_SUMMARY.json")
        if len(root_summary_paths) >= 1:
            # run-root SUMMARY
            self._update_named_step_dict(_named_eval_summaries, root_summary_paths)
        else:
            # checkpoint SUMMARY
            for step, ckp_path in self.step_ckp_paths:
                ckp_summary_paths = glob_with_keys(ckp_path / "evals", "*_SUMMARY.json")
                if len(ckp_summary_paths) >= 1:
                    self._update_named_step_dict(_named_eval_summaries, ckp_summary_paths, step=step)
                else:
                    logger.warning(f"No `SUMMARY.json` found for checkpoint {ckp_path.resolve()}")
        # create eval_summaries_dfs
        self._eval_summaries_dfs = {
            name: pd.DataFrame.from_dict(step_dict, orient="index")
            .rename_axis(["step"]).sort_index()
            for name, step_dict in _named_eval_summaries.items()
        }

    def _load_details(self):
        """
        Load the evaluation details from `*_EVAL.json` files under the run directory.
        """
        _named_eval_details: Dict[str, Dict[int, Dict[str, Any]]] = {}
        # load eval_details
        root_detail_paths = glob_with_keys(self.run_dir, "*_EVAL.json")
        if len(root_detail_paths) >= 1:
            # run-root EVAL
            self._update_named_step_dict(_named_eval_details, root_detail_paths)
        else:
            # checkpoint EVAL
            for step, ckp_path in self.step_ckp_paths:
                ckp_detail_paths = glob_with_keys(ckp_path / "evals", "*_EVAL.json")
                if len(ckp_detail_paths) >= 1:
                    self._update_named_step_dict(_named_eval_details, ckp_detail_paths, step=step)
                else:
                    logger.warning(f"No `EVAL.json` found for checkpoint {ckp_path.resolve()}")
        # create eval_details_df
        self._eval_details_dfs = {
            name: pd.DataFrame.from_dict(step_dict, orient="index")
            .rename_axis(["step"]).sort_index()
            for name, step_dict in _named_eval_details.items()
        }

    @staticmethod
    def _update_named_step_dict(
        named_step_dict: Dict[str, Dict[int, Dict[str, Any]]],
        name_paths: List[tuple[str, Path]],
        step: int = -1
    ):
        for name, path in name_paths:
            named_step_dict.setdefault(name, {})[step] = load_logs(path)

    @property
    def trainer_state(self):
        if not hasattr(self, "_trainer_state"):
            self._load_trainer_state()
        return self._trainer_state

    @property
    def log_history_df(self):
        if not hasattr(self, "_log_history_df"):
            self._load_trainer_state()
        return self._log_history_df

    @property
    def train_keys(self):
        return set(self.log_history_df.columns)

    @property
    def eval_summaries_dfs(self):
        if not hasattr(self, "_eval_summaries_dfs"):
            self._load_summaries()
        return self._eval_summaries_dfs

    @property
    def eval_final_summaries(self):
        return {name: df.iloc[-1].to_dict() for name, df in self.eval_summaries_dfs.items()}

    @property
    def named_metric_keys(self):
        return {name: set(df.columns) for name, df in self.eval_summaries_dfs.items()}

    @property
    def eval_details_dfs(self):
        if not hasattr(self, "_eval_details_dfs"):
            self._load_details()
        return self._eval_details_dfs

    @property
    def named_all_metric_keys(self):
        return {name: set(df.columns) for name, df in self.eval_details_dfs.items()}
