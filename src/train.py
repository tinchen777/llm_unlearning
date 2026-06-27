
from __future__ import annotations
from rich.traceback import install
install(show_locals=False, width=100)
import hydra
from hydra.core.hydra_config import HydraConfig
import logging
from omegaconf import DictConfig

from data import get_data, get_collators
from model import get_model_and_tokenizer
from trainer import load_trainer
from evals import get_evaluators
from utils.common import set_seed, get_cuda_visible_devices
from utils.log import step_logging
from utils.config import TrackingConfig, init_hydra_choices

logging.getLogger("datasets").setLevel(logging.ERROR)
logger = logging.getLogger("main(train)")


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(config: DictConfig):
    """Entry point of the code to train models
    Args:
        config (DictConfig): Config to train
    """
    # cuda device check
    logger.info(f"CUDA_VISIBLE_DEVICES: {get_cuda_visible_devices()}")
    # config
    init_hydra_choices(HydraConfig.get().runtime.choices)
    cfg = TrackingConfig(config)
    # Set seed for reproducibility
    set_seed(cfg["trainer"]["args"]["seed"])
    mode = cfg.get("mode", "train")

    model_cfg = cfg["model"]
    template_args = model_cfg["template_args"]
    # 1. Load model and tokenizer
    with step_logging(logger, "[1/5]", "model & tokenizer", model_cfg):
        model, tokenizer = get_model_and_tokenizer(model_cfg)

    # 2. Load Dataset
    data_cfg = cfg["data"]
    with step_logging(logger, "[2/5]", "data", data_cfg):
        data = get_data(
            data_cfg,
            mode=mode,
            tokenizer=tokenizer,
            template_args=template_args
        )

    # 3. Load collator
    collator_cfg = cfg["collator"]
    with step_logging(logger, "[3/5]", "collator", collator_cfg):
        collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # 4. Get Evaluators
    eval_cfgs = cfg.get("eval", None, allow_none=True)
    if eval_cfgs:
        with step_logging(logger, "[4/5]", "evaluators", eval_cfgs):
            evaluators = get_evaluators(eval_cfgs, template_args=template_args)
    else:
        with step_logging(logger, "[4/5]", "evaluators", is_skip=True):
            evaluators = None

    # 5. Get Trainer
    trainer_cfg = cfg["trainer"]
    with step_logging(logger, "[5/5]", "trainer", trainer_cfg):
        trainer, args = load_trainer(
            trainer_cfg=trainer_cfg,
            model=model,
            train_dataset=data.get("train", None),
            eval_dataset=data.get("eval", None),
            processing_class=tokenizer,
            data_collator=collator,
            evaluators=evaluators,
            template_args=template_args,
        )

    # START TRAINING & EVALUATION
    if args.do_train:
        trainer.train()
        trainer.save_state()
        # trainer.save_model(args.output_dir)
        trainer.save_model()

    if args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
