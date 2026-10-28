
from __future__ import annotations
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch import nn
import logging
from typing import TYPE_CHECKING

from data import get_split_data
from model import get_model_and_tokenizer
from trainer import load_trainer
from evals import get_evaluators
from vis import plot_figures
from utils.common import set_seed, get_cuda_visible_devices
from utils.log import step_logging
from utils.config import hydra_main

if TYPE_CHECKING:
    from utils.config import TrackingConfig

logger = logging.getLogger("main(finetune)")

def get_decoder_layers(model) -> nn.ModuleList:
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    return model.get_decoder().layers

def get_decoder_layers(model) -> nn.ModuleList:
    """The decoder-layer ModuleList, found structurally (works through PEFT/DDP wrappers)."""
    n_layers = model.config.num_hidden_layers
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) == n_layers and name.endswith("layers"):
            return module
    raise ValueError(f"Cannot find the decoder layers (ModuleList of length {n_layers}) in {type(model).__name__}.")


@hydra_main(config_path="../configs", config_name="finetune", argv=["task_name=test"])
def main(cfg: TrackingConfig):
    """Entry point of the code to train models
    Args:
        cfg (TrackingConfig): Config to finetune
    """
    # cuda device check
    logger.info(f"CUDA_VISIBLE_DEVICES: {get_cuda_visible_devices()}")
    # Set seed for reproducibility
    set_seed(cfg["seed"])

    model_cfg = cfg["model"]
    template_args = model_cfg["template_args"]
    # 1. Load model and tokenizer
    with step_logging(logger, "<1/4>", "model & tokenizer", mode="load", cfg=model_cfg):
        model, tokenizer = get_model_and_tokenizer(model_cfg)
    
    print(model)
    print("="*21)
    layers = get_decoder_layers(model)
    print(type(layers))
    print(layers)
    
    layer = layers[0]
    print(type(layer))
    print(layer)
    
    
    
    exit()
    # 2. Load Data
    with step_logging(logger, "<2/4>", "data", mode="load", cfg=cfg["data"], ensure_cfg=True) as data_cfg:
        data = get_split_data(
            data_cfg,
            unwrap_single=True,
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
