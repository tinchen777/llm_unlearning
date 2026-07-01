
from __future__ import annotations
import logging
import os
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from evals.base import Evaluator

logger = logging.getLogger(__name__)

# When using custom evaluators without an eval dataset, pass a dummy value
# to prevent Trainer from raising on eval_dataset=None when eval_strategy is set
_EVAL_PLACEHOLDER = "_EVAL_PLACEHOLDER"


class FinetuneTrainer(Trainer):
    def __init__(self, evaluators: Optional[Dict[str, Evaluator]] = None, *args, **kwargs):
        self.evaluators = evaluators
        if kwargs.get("eval_dataset") is None and evaluators:
            kwargs["eval_dataset"] = _EVAL_PLACEHOLDER
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
                logger.warning(
                    "Custom evaluator can be run with this Trainer only when a single accelerator process is running."
                )
                return {}
            # output_dir
            _run_dir = self._get_output_dir(trial=trial)
            _ckp_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            output_dir = os.path.join(_run_dir, _ckp_folder, "evals")
            os.makedirs(output_dir, exist_ok=True)

            eval_metrics = {}
            for _, evaluator in self.evaluators.items():
                eval_metrics.update(evaluator.evaluate(self.model, output_dir=output_dir))
            self.log(eval_metrics)
            return eval_metrics

        if eval_dataset is None or eval_dataset == _EVAL_PLACEHOLDER:
            return {}
        # Run the default HF Trainer evaluate method when eval dataset is provided
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
