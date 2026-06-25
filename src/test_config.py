
from __future__ import annotations
from rich.traceback import install
install(show_locals=False, width=100)
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
import logging
from omegaconf import DictConfig

# from data import get_data, get_collators
# from model import get_model_and_tokenizer
# from trainer import load_trainer
# from evals import get_evaluators
from utils.common import set_seed, get_cuda_visible_devices
# from utils.log import step_logging
from utils.config import TrackingConfig, init_hydra_choices, HYDRA_CHOICES


logger = logging.getLogger("main(train)")



@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(config: DictConfig):
    print("device", get_cuda_visible_devices(), type(get_cuda_visible_devices()))
    
    print("num_devices =", torch.cuda.device_count())
    print(torch.cuda.current_device())
    exit()
    init_hydra_choices(HydraConfig.get().runtime.choices)
    
    print(f"hydra: {HYDRA_CHOICES}")
    print(len(HYDRA_CHOICES))
    
    cfg = TrackingConfig(config)
    
    model_cfg = cfg["model"]
    print(model_cfg.loc)
    print(model_cfg.loc_choices)
    
    model_args = model_cfg["template_args"]
    print(model_args.loc)
    print(model_args.loc_choices)
    
    # path = model_args["pretrained_model_name_or_path"]
    # print(path)
    # print(type(path))
    
    for i in model_args:
        print(i)
    
    print("======")
    
    for key, val in model_args.items():
        print(key, val)
    
    print("======")
    
    a = dict(**model_args)
    print(a)
    
    
    
    
    

    # # cuda device check
    # logger.info(f"CUDA_VISIBLE_DEVICES: {get_cuda_visible_devices()}")
    # cfg = TrackingConfig("@", config)
    # # Set seed for reproducibility
    # set_seed(cfg["trainer"]["args"]["seed"])
    # mode = cfg.get("mode", "train")

    # model_cfg = cfg["model"]
    # template_args = model_cfg["template_args"]
    # # Load model and tokenizer
    # with step_logging(logger, "[1/5]", "model & tokenizer", model_cfg):
    #     model, tokenizer = get_model_and_tokenizer(model_cfg)

    # # Load Dataset
    # data_cfg = cfg["data"]
    # with step_logging(logger, "[2/5]", "data", data_cfg):
    #     data = get_data(
    #         data_cfg,
    #         mode=mode,
    #         tokenizer=tokenizer,
    #         template_args=template_args
    #     )

    # # Load collator
    
    
    
    # collator_cfg = get_cfg(cfg, "collator", "configs")
    # with step_logging(logger, "[3/5]", "collator", collator_cfg):
    #     collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # trainer_cfg = get_cfg(cfg, "trainer", "configs")
    # assert trainer_cfg is not None, ValueError("Please set trainer")

    # # Get Evaluators
    # evaluators = None
    # eval_cfgs = cfg.get("eval", None)
    # if eval_cfgs:
    #     with step_logging(logger, "[4/5]", "evaluators", eval_cfgs):
    #         evaluators = get_evaluators(
    #             eval_cfgs=eval_cfgs,
    #             template_args=template_args,
    #             model=model,
    #             tokenizer=tokenizer,
    #         )
    # else:
    #     logger.info(cstr("[4/5]", fg="y", styles="bold") + cstr(" No evaluators found in config, skipping evaluation setup.", styles="bold"))

    # # Get Trainer
    # with step_logging(logger, "[5/5]", "trainer", trainer_cfg):
    #     trainer, trainer_args = load_trainer(
    #         trainer_cfg=trainer_cfg,
    #         model=model,
    #         train_dataset=data.get("train", None),
    #         eval_dataset=data.get("eval", None),
    #         processing_class=tokenizer,
    #         data_collator=collator,
    #         evaluators=evaluators,
    #         template_args=template_args,
    #     )

    # if trainer_args.do_train:
    #     trainer.train()
    #     trainer.save_state()
    #     trainer.save_model(trainer_args.output_dir)

    # if trainer_args.do_eval:
    #     trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
