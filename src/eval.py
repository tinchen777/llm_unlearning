
from __future__ import annotations
import logging
from typing import TYPE_CHECKING

from model import get_model_and_tokenizer
from evals import get_evaluators
from vis import plot_figures
from utils.common import set_seed, get_cuda_visible_devices
from utils.log import step_logging
from utils.config import hydra_main

if TYPE_CHECKING:
    from utils.config import TrackingConfig

logger = logging.getLogger("main(eval)")


@hydra_main(config_path="../configs", config_name="eval")
def main(cfg: TrackingConfig):
    """Entry point of the code to evaluate models
    Args:
        cfg (TrackingConfig): Config to evaluate
    """
    # cuda device check
    logger.info(f"CUDA_VISIBLE_DEVICES: {get_cuda_visible_devices()}")
    # Set seed for reproducibility
    set_seed(cfg["seed"])

    model_cfg = cfg["model"]
    template_args = model_cfg["template_args"]
    # 1. Load model and tokenizer
    with step_logging(logger, "<1/2>", "model & tokenizer", mode="load", cfg=model_cfg):
        model, tokenizer = get_model_and_tokenizer(model_cfg)
    # 2. Get Evaluators
    with step_logging(logger, "<2/2>", "evaluators", mode="load", cfg=cfg["eval"], ensure_cfg=True) as eval_cfgs:
        evaluators = get_evaluators(
            eval_cfgs,
            tokenizer=tokenizer,
            template_args=template_args
        )

    # ==========
    run_dir = None
    # EVALUATION
    with step_logging(logger, "#1", "evaluation"):
        for evaluator in evaluators.values():
            evaluator.evaluate(model)
            run_dir = evaluator.output_dir

    # plotting
    vis_cfg = cfg.get("vis", None)
    if vis_cfg and run_dir is not None:
        with step_logging(logger, "#2", "plotting"):
            plot_figures(run_dir, vis_cfg=vis_cfg)


if __name__ == "__main__":
    main()
