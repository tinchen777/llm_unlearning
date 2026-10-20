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
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

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
        logger.info(f"Loading run directory: {_run_dir.resolve()}")
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
        """Update the named step dictionary with data loaded from the given paths."""
        for name, path in name_paths:
            named_step_dict.setdefault(name, {})[step] = load_logs(path)

    def __repr__(self) -> str:
        return f"<Exp {self.run_dir}>"

    @property
    def trainer_state(self):
        """Return the trainer state loaded from `trainer_state.json`."""
        if not hasattr(self, "_trainer_state"):
            self._load_trainer_state()
        return self._trainer_state

    @property
    def log_history_df(self):
        """Return the training log history from `trainer_state.json` as a DataFrame."""
        if not hasattr(self, "_log_history_df"):
            self._load_trainer_state()
        return self._log_history_df

    @property
    def train_keys(self):
        """Return the set of training keys available in the log history DataFrame."""
        return set(self.log_history_df.columns)

    @property
    def eval_summaries_dfs(self):
        """Return a dictionary of evaluation summary DataFrames, keyed by evaluation name."""
        if not hasattr(self, "_eval_summaries_dfs"):
            self._load_summaries()
        return self._eval_summaries_dfs

    @property
    def eval_final_summaries(self):
        """Return a dictionary of the final evaluation summaries, keyed by evaluation name."""
        return {name: df.iloc[-1].to_dict() for name, df in self.eval_summaries_dfs.items()}

    @property
    def named_metric_keys(self):
        """Return a dictionary of metric keys for each evaluation summary DataFrame, keyed by evaluation name."""
        return {name: set(df.columns) for name, df in self.eval_summaries_dfs.items()}

    @property
    def eval_details_dfs(self):
        """Return a dictionary of evaluation details DataFrames, keyed by evaluation name."""
        if not hasattr(self, "_eval_details_dfs"):
            self._load_details()
        return self._eval_details_dfs

    @property
    def named_all_metric_keys(self):
        """Return a dictionary of all metric keys for each evaluation details DataFrame, keyed by evaluation name."""
        return {name: set(df.columns) for name, df in self.eval_details_dfs.items()}


# ---------------------------------------------------------------------------------------------
# NEAR / LUNAR runs
# ---------------------------------------------------------------------------------------------

_NUM = r"[0-9.eE+-]+?"
# setting folder of a task dir: NEAR `k5_c1_eoi`, LUNAR `c1_all`  (config_dir() in src/METHOD/*/_utils/)
_NEAR_SETTING = re.compile(rf"^k(?P<k>\d+)_c(?P<coeff>{_NUM})_(?P<mode>eoi|all)$")
_LUNAR_SETTING = re.compile(rf"^c(?P<coeff>{_NUM})_(?P<mode>eoi|all)$")
# W_down run folder inside a setting: `l16`, `l10-11_cf`  (run_dir())
_RUN = re.compile(r"^l(?P<layers>\d+(?:-\d+)*)(?P<cf>_cf)?$")


def match_dirs(root: Union[PathLike, str], pattern: str) -> List[tuple[Dict[str, str], Path]]:
    """Sub-directories of `root` whose NAME matches the regex `pattern` fully; with the named groups of the match."""
    root = Path(root)
    if not root.is_dir():
        return []
    regex = re.compile(f"^(?:{pattern})$")
    out = []
    for path in sorted(root.iterdir()):
        m = regex.match(path.name)
        if path.is_dir() and m is not None:
            out.append((m.groupdict(), path))
    return out


def _read_json(path: Path) -> Any:
    return load_logs(path) if path.exists() else None


class MethodRunLoader:
    """A NEAR / LUNAR task directory (`saves/NEAR/<task_name>/`, `saves/LUNAR/<task_name>/`).

    Layout (see the file headers of src/METHOD/*/):
        r_uv.pt, r_uv_summary.json
        <setting>/layer_selection.json, responses_steer_l<l>.json          setting = k<K>_c<coeff>_<mode> | c<coeff>_<mode>
        <setting>/l<layers>[_cf]/w_down_summary.json, model/, ...

    Every table is a tidy DataFrame (one row per observation) so it can be written to csv and re-plotted.
    """

    def __init__(self, task_dir: Union[PathLike, str]):
        self.task_dir = Path(task_dir)
        if not self.task_dir.is_dir():
            raise NotADirectoryError(f"Task directory not a existing directory: {self.task_dir.resolve()}")
        self.label = self.task_dir.name
        # `saves/NEAR/<task>` -> NEAR
        self.method = self.task_dir.parent.name
        self._settings: Optional[Dict[str, Dict[str, Any]]] = None

    @property
    def settings(self) -> Dict[str, Dict[str, Any]]:
        """`{setting folder name: {"dir", "k" (None for LUNAR), "coeff", "mode"}}`."""
        if self._settings is None:
            self._settings = {}
            for path in sorted(self.task_dir.iterdir()):
                if not path.is_dir():
                    continue
                for regex in (_NEAR_SETTING, _LUNAR_SETTING):
                    m = regex.match(path.name)
                    if m is not None:
                        d = m.groupdict()
                        self._settings[path.name] = {
                            "dir": path, "k": int(d["k"]) if "k" in d else None,
                            "coeff": float(d["coeff"]), "mode": d["mode"],
                        }
                        break
        return self._settings

    def _setting_columns(self, name: str) -> Dict[str, Any]:
        s = self.settings[name]
        return {"setting": name, "k": s["k"], "coeff": s["coeff"], "mode": s["mode"]}

    def selection(self, setting: str) -> Optional[Dict[str, Any]]:
        return _read_json(self.settings[setting]["dir"] / "layer_selection.json")

    @property
    def layer_selection_df(self) -> pd.DataFrame:
        """One row per (setting, candidate layer): the per-layer metrics of select_layer.py (NEAR: `unseen_gap`, rougeL_recall,
        answer_logprob, refusal / degenerate rate; LUNAR: `s1`, `s2`, `score`) and `selected` (the chosen layer)."""
        rows = []
        for name in self.settings:
            sel = self.selection(name)
            if not sel:
                continue
            for layer, metrics in sel["per_layer"].items():
                rows.append({**self._setting_columns(name), "layer": int(layer), **metrics,
                             "selected": int(layer) == int(sel["selected_layer"])})
        return pd.DataFrame(rows)

    @property
    def runs(self) -> List[Dict[str, Any]]:
        """The W_down runs: `{setting, run, dir, layers, solver}` of every `<setting>/l<layers>[_cf]/w_down_summary.json`."""
        out = []
        for name, s in self.settings.items():
            for d, path in match_dirs(s["dir"], _RUN.pattern.strip("^$")):
                if (path / "w_down_summary.json").exists():
                    out.append({"setting": name, "run": path.name, "dir": path,
                                "layers": [int(x) for x in d["layers"].split("-")],
                                "solver": "closed_form" if d["cf"] else "adam"})
        return out

    @property
    def behavior_df(self) -> pd.DataFrame:
        """One row per (W_down run, phase, split): the response statistics (refusal_rate, hallucination_rate, rougeL_recall,
        answer_prob, ...) of the ORIGINAL model (`phase="before"`) and of the edited one (`"after"`) on a fixed subset of every split."""
        rows = []
        for run in self.runs:
            summary = load_logs(run["dir"] / "w_down_summary.json")
            for phase, per_split in summary.get("responses", {}).items():
                for split, stats in per_split.items():
                    rows.append({**self._setting_columns(run["setting"]), "run": run["run"], "layers": "-".join(map(str, run["layers"])),
                                 "solver": run["solver"], "phase": phase, "split": split, **stats})
        return pd.DataFrame(rows)

    @property
    def edit_df(self) -> pd.DataFrame:
        """One row per (W_down run, layer): how well the edit was fitted (`forget_mse_*`, `retain_mse_*`) and how far the forget
        activations really moved vs the intended shift (`cos`, `norm_ratio`)."""
        rows = []
        for run in self.runs:
            summary = load_logs(run["dir"] / "w_down_summary.json")
            for layer, stats in summary.get("per_layer", {}).items():
                achieved = summary.get("achieved_vs_intended", {}).get(layer, {})
                rows.append({**self._setting_columns(run["setting"]), "run": run["run"], "solver": run["solver"], "layer": int(layer),
                             **{k: v for k, v in stats.items() if k != "history"}, **achieved})
        return pd.DataFrame(rows)

    @property
    def r_uv_df(self) -> pd.DataFrame:
        """`r_uv_summary.json` as rows: NEAR (`k`, `layer`, `norm`, `rel_norm`, `cos_to_kmax`), LUNAR (`layer`, `norm`, `rel_norm`)."""
        data = _read_json(self.task_dir / "r_uv_summary.json")
        if not data:
            return pd.DataFrame()
        rows = []
        for key, value in data.items():
            if all(isinstance(v, dict) for v in value.values()):  # NEAR: {K: {layer: {...}}}
                rows += [{"k": int(key), "layer": int(layer), **stats} for layer, stats in value.items()]
            else:  # LUNAR: {layer: {...}}
                rows.append({"layer": int(key), **value})
        return pd.DataFrame(rows)


class ProbeRunLoader:
    """A probe output directory of src/METHOD/NEAR/probe.py (`activations_summary.json`, `responses/responses_summary.json`)."""

    def __init__(self, run_dir: Union[PathLike, str]):
        self.run_dir = Path(run_dir)
        if not self.run_dir.is_dir():
            raise NotADirectoryError(f"Probe directory not a existing directory: {self.run_dir.resolve()}")
        self.label = self.run_dir.name

    @property
    def layers_df(self) -> pd.DataFrame:
        """One row per layer; columns: `r_uv_norm`, `r_uv_rel_norm`, `norm[<split>]`, `cos[<a>,<b>]`, `auc[<A> | <B>]`,
        `auc_group[<A> | <B>]` (None when a set has fewer than 4 authors), as in the file."""
        data = _read_json(self.run_dir / "activations_summary.json")
        if not data:
            return pd.DataFrame()
        return pd.DataFrame([{"layer": int(layer), **stats} for layer, stats in data["layers"].items()]).sort_values("layer")

    def auc(self, pair: str, by_author: bool = False) -> pd.Series:
        """AUC of separating the two question sets `pair` (e.g. `"forget | neighbor"`) per layer; `by_author` = folds by author."""
        df = self.layers_df
        key = f"{'auc_group' if by_author else 'auc'}[{pair}]"
        if df.empty or key not in df:
            return pd.Series(dtype=float)
        return df.set_index("layer")[key].astype(float)

    @property
    def responses_df(self) -> pd.DataFrame:
        """One row per split: the response statistics (refusal_rate, hallucination_rate, ...)."""
        for name in ("responses/responses_summary.json", "responses_summary.json"):
            data = _read_json(self.run_dir / name)
            if data:
                return pd.DataFrame([{"split": split, **stats} for split, stats in data["splits"].items()])
        return pd.DataFrame()
