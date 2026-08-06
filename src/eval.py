
from __future__ import annotations
# from rich.traceback import install
# install(show_locals=False, width=100)
import hydra
from hydra.core.hydra_config import HydraConfig
import logging
from omegaconf import DictConfig

from model import get_model_and_tokenizer
from evals import get_evaluators
from vis import plot_figures
from utils.common import get_cuda_visible_devices
from utils.log import step_logging
from utils.config import TrackingConfig, init_hydra_choices

logger = logging.getLogger("main(eval)")


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(config: DictConfig):
    """Entry point of the code to evaluate models
    Args:
        config (DictConfig): Config to evaluate
    """
    # cuda device check
    logger.info(f"CUDA_VISIBLE_DEVICES: {get_cuda_visible_devices()}")
    # config
    init_hydra_choices(HydraConfig.get().runtime.choices)
    cfg = TrackingConfig(config)
    # Set seed for reproducibility
    # set_seed(cfg["trainer"]["args"]["seed"])
    # mode = cfg.get("mode", "eval")

    model_cfg = cfg["model"]
    template_args = model_cfg["template_args"]
    # 1. Load model and tokenizer
    with step_logging(logger, "[1/2]", "model & tokenizer", model_cfg):
        model, tokenizer = get_model_and_tokenizer(model_cfg)

    # 2. Get Evaluators
    eval_cfgs = cfg["eval"]
    with step_logging(logger, "[2/2]", "evaluators", eval_cfgs):
        evaluators = get_evaluators(
            eval_cfgs,
            tokenizer=tokenizer,
            template_args=template_args
        )

    # START EVALUATION
    vis_cfg = cfg.get("vis", None)
    for evaluator in evaluators.values():
        evaluator.evaluate(model)
        # plotting
        run_dir = evaluator.output_dir
        if vis_cfg and run_dir is not None:
            plot_figures([run_dir], vis_cfg)


if __name__ == "__main__":
    main()
