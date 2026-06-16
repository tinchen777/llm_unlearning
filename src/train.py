
from rich.traceback import install
install(show_locals=True, width=100)

import os
import hydra
import logging
from omegaconf import DictConfig

from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from trainer.utils import seed_everything
from evals import get_evaluators


logger = logging.getLogger("main(train)")

@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")

    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    logger.info(f"Loading model")
    model, tokenizer = get_model(model_cfg)
    logger.info(f"Loaded model with args {model_cfg.model_args}")

    # Load Dataset
    data_cfg = cfg.data
    logger.info(f"Loading data")
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )
    logger.info(f"Loaded data with config: {data_cfg}")

    # Load collator
    collator_cfg = cfg.collator
    logger.info(f"Loading collator")
    collator = get_collators(collator_cfg, tokenizer=tokenizer)
    logger.info(f"Loaded collator with config {collator_cfg}")

    # Get Trainer
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")

    # Get Evaluators
    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs:
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        processing_class=tokenizer,
        data_collator=collator,
        evaluators=evaluators,
        template_args=template_args,
    )

    if trainer_args.do_train:
        trainer.train()
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
