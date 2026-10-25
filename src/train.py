
from __future__ import annotations
# from rich.traceback import install
# install(show_locals=False, width=100)
import hydra
from hydra.core.hydra_config import HydraConfig
import logging
from typing import TYPE_CHECKING

from data import get_split_data
from model import get_model_and_tokenizer
from trainer import load_trainer
from evals import get_evaluators
from vis import plot_figures
from utils.common import set_seed, get_cuda_visible_devices
from utils.log import step_logging
from utils.config import TrackingConfig, init_hydra_choices

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger("main(finetune)")


@hydra.main(version_base=None, config_path="../configs", config_name="finetune")
def main(config: DictConfig):
    """Entry point of the code to train models
    Args:
        config (DictConfig): Config to finetune
    """
    # cuda device check
    logger.info(f"CUDA_VISIBLE_DEVICES: {get_cuda_visible_devices()}")
    # config
    init_hydra_choices(HydraConfig.get().runtime.choices)
    cfg = TrackingConfig(config)
    # Set seed for reproducibility
    set_seed(cfg["seed"])

    model_cfg = cfg["model"]
    template_args = model_cfg["template_args"]
    # 1. Load model and tokenizer
    with step_logging(logger, "<1/4>", "model & tokenizer", mode="load", cfg=model_cfg):
        model, tokenizer = get_model_and_tokenizer(model_cfg)
    # 2. Load Data
    with step_logging(logger, "<2/4>", "data", mode="load", cfg=cfg["data"], ensure_cfg=True) as data_cfg:
        data = get_split_data(
            data_cfg,
            use_single_dataset=True,
            tokenizer=tokenizer,
            template_args=template_args
        )
        train_dataset, collator = data["train"]
    # 3. Get Evaluators
    with step_logging(logger, "<3/4>", "evaluators", mode="load", cfg=cfg.get("eval", None)) as eval_cfgs:
        if eval_cfgs:
            evaluators = get_evaluators(
                eval_cfgs,
                tokenizer=tokenizer,
                template_args=template_args
            )
        else:
            evaluators = None
    # 4. Get Trainer
    with step_logging(logger, "<4/4>", "trainer", mode="load", cfg=cfg["trainer"], ensure_cfg=True) as trainer_cfg:
        trainer = load_trainer(
            trainer_cfg,
            model=model,
            evaluators=evaluators,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            data_collator=collator
        )

    # ==========
    # TRAINING
    if trainer.args.do_train:
        with step_logging(logger, "#1", "training"):
            trainer.train()
            trainer.save_state()
            trainer.save_model()

    # EVALUATION
    if trainer.args.do_eval and trainer.args.eval_strategy != "epoch":
        with step_logging(logger, "#2", "evaluation"):
            trainer.evaluate(metric_key_prefix="eval")

    # plotting
    vis_cfg = cfg.get("vis", None)
    run_dir = trainer.args.output_dir
    if vis_cfg and run_dir is not None:
        with step_logging(logger, "#3", "plotting"):
            plot_figures(run_dir, vis_cfg=vis_cfg)


if __name__ == "__main__":
    main()
