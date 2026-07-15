"""Loaders for experiment outputs saved under a run directory.

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
import glob
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("vis")

_CKP_RE = re.compile(r"checkpoint-(\d+)$")

# Keys in log_history that are bookkeeping, not plottable series
_NON_METRIC_KEYS = ("step", "epoch")


def run_label(run_dir: str) -> str:
    """Short label for a run: the task_name directory."""
    return os.path.basename(os.path.normpath(run_dir))


def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def load_trainer_state(run_dir: str) -> Optional[Dict[str, Any]]:
    """Load `trainer_state.json` from the run root (or its last checkpoint)."""
    candidates = [os.path.join(run_dir, "trainer_state.json")]
    ckps = sorted(
        glob.glob(os.path.join(run_dir, "checkpoint-*")),
        key=lambda p: int(_CKP_RE.search(p).group(1)) if _CKP_RE.search(p) else -1,
    )
    candidates += [os.path.join(c, "trainer_state.json") for c in reversed(ckps)]
    for path in candidates:
        if os.path.exists(path):
            return _load_json(path)
    logger.warning(f"No trainer_state.json found under {run_dir}")
    return None


def collect_series(
    log_history: List[Dict[str, Any]], key: str
) -> Tuple[List[float], List[float]]:
    """Collect (steps, values) for a numeric key across log_history entries."""
    steps, values = [], []
    for entry in log_history:
        val = entry.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            steps.append(entry.get("step", len(steps)))
            values.append(float(val))
    return steps, values


def list_numeric_keys(log_history: List[Dict[str, Any]]) -> List[str]:
    """All numeric keys that appear in log_history (bookkeeping excluded),
    in first-appearance order."""
    keys: List[str] = []
    for entry in log_history:
        for k, v in entry.items():
            if k in _NON_METRIC_KEYS or k in keys:
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                keys.append(k)
    return keys


def load_checkpoint_summaries(run_dir: str) -> Dict[int, Dict[str, Any]]:
    """Aggregated eval metrics per checkpoint: {step: {metric: agg_value}}.

    Merges every `checkpoint-*/evals/*_SUMMARY.json`. A run-root
    `*_SUMMARY.json` (standalone eval) is returned under step -1.
    """
    summaries: Dict[int, Dict[str, Any]] = {}
    for path in glob.glob(os.path.join(run_dir, "checkpoint-*", "evals", "*_SUMMARY.json")):
        m = _CKP_RE.search(os.path.dirname(os.path.dirname(path)))
        if m is None:
            continue
        step = int(m.group(1))
        summaries.setdefault(step, {}).update(_load_json(path))
    for path in glob.glob(os.path.join(run_dir, "*_SUMMARY.json")):
        summaries.setdefault(-1, {}).update(_load_json(path))
    return dict(sorted(summaries.items()))


def load_final_summary(run_dir: str) -> Dict[str, Any]:
    """Aggregated eval metrics of the run's last evaluation (or the standalone
    eval result if that's all there is)."""
    summaries = load_checkpoint_summaries(run_dir)
    if not summaries:
        logger.warning(f"No *_SUMMARY.json found under {run_dir}")
        return {}
    last_step = max(summaries)
    return summaries[last_step]


def load_eval_details(
    run_dir: str, step: Optional[int] = None
) -> Dict[str, Any]:
    """Fine-grained eval logs ({metric: {agg_value, value_by_index, ...}}).

    `run_dir` may also directly be a path to an `*_EVAL.json` file.
    `step=None` selects the last checkpoint; the run root is the fallback.
    """
    if os.path.isfile(run_dir):
        return _load_json(run_dir)

    by_step: Dict[int, List[str]] = {}
    for path in glob.glob(os.path.join(run_dir, "checkpoint-*", "evals", "*_EVAL.json")):
        m = _CKP_RE.search(os.path.dirname(os.path.dirname(path)))
        if m:
            by_step.setdefault(int(m.group(1)), []).append(path)
    root_paths = glob.glob(os.path.join(run_dir, "*_EVAL.json"))
    if root_paths:
        by_step.setdefault(-1, []).extend(root_paths)

    if not by_step:
        logger.warning(f"No *_EVAL.json found under {run_dir}")
        return {}
    if step is None:
        step = max(by_step)
    elif step not in by_step:
        raise FileNotFoundError(
            f"No evals for step {step} under {run_dir}; available: {sorted(by_step)}"
        )
    details: Dict[str, Any] = {}
    for path in by_step[step]:
        details.update(_load_json(path))
    return details


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
