
from __future__ import annotations
import json
import logging
import os
import re
from pathlib import Path
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from evals.base import Evaluator

logger = logging.getLogger(__name__)

# When using custom evaluators without an eval dataset, pass a dummy value
# to prevent Trainer from raising on eval_dataset=None when eval_strategy is set
_DATASETS_PLACEHOLDER = "_DATASETS_PLACEHOLDER"


class FinetuneTrainer(Trainer):
    def __init__(self, evaluators: Optional[Dict[str, Evaluator]] = None, *args, **kwargs):
        self.evaluators = evaluators
        if kwargs.get("eval_dataset") is None and evaluators:
            kwargs["eval_dataset"] = _DATASETS_PLACEHOLDER
        super().__init__(*args, **kwargs)

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        trial: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        # Run a custom evaluator and save results
        if self.evaluators and self.accelerator.is_local_main_process:
            if self.accelerator.num_processes != 1:
                logger.error(
                    "Custom evaluator can be run with this Trainer only when a single accelerator process is running. "
                    "Evaluate will be skipped!"
                )
                return {}
            # output_dir
            _run_dir = Path(self._get_output_dir(trial=trial))
            _ckp_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            output_dir = _run_dir / _ckp_folder / "evals"
            output_dir.mkdir(parents=True, exist_ok=True)

            eval_metrics = {}
            for evaluator in self.evaluators.values():
                eval_metrics.update(
                    evaluator.evaluate(
                        self.model,
                        metric_key_prefix=metric_key_prefix,
                        output_dir=output_dir,
                    )
                )
            # NO eval metrics logging in trainer_state.json
            # self.log(eval_metrics)
            self._maybe_stop_at_metric(eval_metrics, _run_dir)
            return eval_metrics

        if eval_dataset is None or eval_dataset == _DATASETS_PLACEHOLDER:
            return {}
        # Run the default HF Trainer evaluate method when eval dataset is provided
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def _maybe_stop_at_metric(self, eval_metrics: Dict[str, float], run_dir: Path) -> None:
        """Optional stopping rule, OFF unless the environment variable STOP_AT_METRIC is set, e.g. STOP_AT_METRIC="forget_Q_A_Prob<=0.25" (A2 shadow edits of the
        training baselines, scripts/exp/spa_lib.sh): after an evaluation at a step > 0 whose metric satisfies the condition, training stops at the end of that epoch, so the
        final model that train.py saves is the model of that evaluation (`<run_dir>/stop.json` records step, epoch and value). If it is never met, training runs to the end
        and stop.json says so (`met: false`)."""
        rule = os.environ.get("STOP_AT_METRIC")
        if not rule:
            return
        m = re.fullmatch(r"\s*(\w+)\s*(<=|>=)\s*([-+0-9.eE]+)\s*", rule)
        if m is None:
            raise ValueError(f"STOP_AT_METRIC must look like `forget_Q_A_Prob<=0.25`, got `{rule}`")
        key, op, thr = m.group(1), m.group(2), float(m.group(3))
        if key not in eval_metrics:
            raise KeyError(f"STOP_AT_METRIC: metric `{key}` is not among the evaluated ones {sorted(eval_metrics)}")
        value = float(eval_metrics[key])
        met = value <= thr if op == "<=" else value >= thr
        step = int(self.state.global_step)
        last = step >= int(self.state.max_steps or 0) > 0
        if step > 0 and (met or last):
            record = {"rule": rule, "met": bool(met), "step": step, "epoch": float(self.state.epoch or 0), key: value}
            with open(run_dir / "stop.json", "w") as fh:
                json.dump(record, fh, indent=1)
            logger.info(f"STOP_AT_METRIC {rule}: step {step} {key}={value:.4f} -> {'stop' if met else 'end of training, never met'}")
            if met:
                self.control.should_training_stop = True
