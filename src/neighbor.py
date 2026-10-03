
from __future__ import annotations
# from rich.traceback import install
# install(show_locals=False, width=100)
import hydra
from hydra.core.hydra_config import HydraConfig
import logging
import os
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from data import get_split_loaders, get_split_data, one_loader, one_dataset
from model import get_model_and_tokenizer
from model.activations import collect_activations, get_eoi_positions, unlearning_vector
from evals.metrics.metric_utils import eval_text_similarity
from utils.common import get_cuda_visible_devices, save_logs, set_seed
from utils.log import step_logging
from utils.config import TrackingConfig, init_hydra_choices

logger = logging.getLogger("main(neighbor)")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(config: DictConfig):
    """Probe model responses and compute the LUNAR unlearning vector r_UV (no Trainer).
    Args:
        config (DictConfig): e.g. experiment=custom/lunar_uv
    """
    # cuda device check
    logger.info(f"CUDA_VISIBLE_DEVICES: {get_cuda_visible_devices()}")
    # config
    init_hydra_choices(HydraConfig.get().runtime.choices)
    cfg = TrackingConfig(config)
    # Set seed for reproducibility
    set_seed(cfg["trainer"]["args"]["seed"])
    mode = cfg.get("mode", "neighbor", check_none=True)

    model_cfg = cfg["model"]
    template_args = model_cfg["template_args"]
    # 1. Load model and tokenizer
    with step_logging(logger, "[1/4]", "model & tokenizer", model_cfg):
        model, tokenizer = get_model_and_tokenizer(model_cfg)
        model.eval()

    # # 2. Load dataloaders
    # with step_logging(logger, "[2/4]", "dataloaders", cfg["data"]):
    #     loaders = get_split_loaders(
    #         cfg["data"],
    #         batch_size=1000,
    #         tokenizer=tokenizer,
    #         template_args=template_args
    #     )
    #     train_loader = one_loader(loaders["train"])
    #     ref_loader = one_loader(loaders["ref"])

    # 2. Load data
    with step_logging(logger, "[2/4]", "data", cfg["data"]):
        data = get_split_data(
            cfg["data"],
            tokenizer=tokenizer,
            template_args=template_args
        )
        train_dataset = one_dataset(data["train"][0])
        ref_dataset = one_dataset(data["ref"][0])
    
    print("train_dataset", type(train_dataset))
    print((list(train_dataset)))
    print("ref_dataset", type(ref_dataset))
    print((list(ref_dataset)))
    
    
    exit()
    
    
    
    
    
    
    for (data, ref_data) in zip(train_loader, ref_loader):
        
        print("data", type(data))
        print((list(data)))
        print(data["index"])
        input_ids = data["input_ids"]
        print(input_ids.shape)
        # print(tokenizer.batch_decode(data["input_ids"], skip_special_tokens=True))
        print("="*50, end="\n\n")
        print("ref_data", type(ref_data))
        print((list(ref_data)))
        print(ref_data["index"])
        input_ids = ref_data["input_ids"]
        print(input_ids.shape)
        # print(tokenizer.batch_decode(ref_data["input_ids"], skip_special_tokens=True))
        print("="*50, end="\n\n")
        
        # print(list(data))
        
        # exit()
        
    
    
    # print(train_loader)
    
    
    
    
    exit()
    
    
    
    
    
    forget_split = gen_cfg["forget_split"]
    reference_split = gen_cfg["reference_split"]

    # 3. Model responses on each split
    with step_logging(logger, "[3/4]", "responses", gen_cfg):
        num_gen_batches = gen_cfg.get("num_gen_batches", None)
        for split, loader in loaders.items():
            records = []
            for batch_idx, batch in enumerate(loader):
                if num_gen_batches is not None and batch_idx >= num_gen_batches:
                    break
                indices = batch.pop("index")
                outputs = eval_text_similarity(
                    model, batch, tokenizer=tokenizer, generation_args=cfg["generation_args"]
                )
                records.extend({"index": idx, **out} for idx, out in zip(indices, outputs))
            save_logs(records, os.path.join(output_dir, f"responses_{split}.json"))
            logger.info(f"Saved {len(records)} `{split}` responses, e.g.: {records[0]['generation']!r}")

    # 4. Activations & unlearning vector
    with step_logging(logger, "[4/4]", "activations & r_UV", gen_cfg):
        positions = gen_cfg.get("positions", None)
        positions = list(positions) if positions is not None else get_eoi_positions(tokenizer, template_args)
        layers = gen_cfg.get("layers", None)
        point = gen_cfg.get("point", "block_out")
        logger.info(f"Hook point: `{point}`, positions: {positions}")

        means = {}
        for split in (forget_split, reference_split):
            means[split], layers = collect_activations(
                model, loaders[split], positions=positions, layers=layers,
                point=point, reduce="mean", desc=split
            )
        # r_UV[p, l] (layer l == LUNAR's `layer_modified`, i.e. its pre-hook direction at l+1)
        r_uv = unlearning_vector(means[reference_split], means[forget_split])  # [n_pos, n_layers, d]

        torch.save({
            "r_uv": r_uv.float(),
            "mean_forget": means[forget_split].float(),
            "mean_reference": means[reference_split].float(),
            "positions": positions,
            "layers": layers,
            "point": point,
            "n_forget": len(loaders[forget_split].dataset),
            "n_reference": len(loaders[reference_split].dataset),
        }, os.path.join(output_dir, "r_uv.pt"))

        # per-layer diagnostics (averaged over positions) to help pick the layer to modify
        cos = F.cosine_similarity(means[forget_split], means[reference_split], dim=-1)  # [n_pos, n_layers]
        summary = {
            str(layer): {
                "r_uv_norm": r_uv[:, i].norm(dim=-1).mean().item(),
                "forget_norm": means[forget_split][:, i].norm(dim=-1).mean().item(),
                "cos_forget_reference": cos[:, i].mean().item(),
            }
            for i, layer in enumerate(layers)
        }
        save_logs(summary, os.path.join(output_dir, "r_uv_summary.json"))
        logger.info(f"Saved r_UV {tuple(r_uv.shape)} (n_pos, n_layers, d) to {output_dir}")


if __name__ == "__main__":
    main()
