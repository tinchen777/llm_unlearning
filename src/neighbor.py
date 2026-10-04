
from __future__ import annotations
# from rich.traceback import install
# install(show_locals=False, width=100)
import hydra
from hydra.core.hydra_config import HydraConfig
import itertools
import logging
import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import DictConfig

from data import get_split_loaders, one_loader
from model import get_model_and_tokenizer
from model.activations import collect_activations, get_eoi_positions, unlearning_vector
from model.generation import generate_responses
from utils.common import get_cuda_visible_devices, save_logs, set_seed
from utils.log import step_logging
from utils.config import TrackingConfig, init_hydra_choices

logger = logging.getLogger("main(generate)")


@hydra.main(version_base=None, config_path="../configs", config_name="generate")
def main(config: DictConfig):
    """Probe a model on the forget / neighbor / retain question sets (no Trainer):
    read its responses, then compute per-layer activations and the unlearning vector r_UV.
    Args:
        config (DictConfig): e.g. `experiment=custom/neighbor_probe`
    """
    # cuda device check
    logger.info(f"CUDA_VISIBLE_DEVICES: {get_cuda_visible_devices()}")
    # config
    init_hydra_choices(HydraConfig.get().runtime.choices)
    cfg = TrackingConfig(config)
    set_seed(cfg["seed"])
    probe_cfg = cfg["probe"]
    output_dir = Path(str(cfg["paths"]["output_dir"]))

    model_cfg = cfg["model"]
    template_args = model_cfg["template_args"]
    # 1. Load model and tokenizer
    with step_logging(logger, "[1/4]", "model & tokenizer", model_cfg):
        model, tokenizer = get_model_and_tokenizer(model_cfg)
        model.eval()

    # 2. Load one dataloader per split: {split: DataLoader}
    with step_logging(logger, "[2/4]", "dataloaders", cfg["data"]):
        split_loaders = get_split_loaders(
            cfg["data"],
            batch_size=probe_cfg["batch_size"],
            shuffle=False,
            tokenizer=tokenizer,
            template_args=template_args
        )
        forget_loader = one_loader(split_loaders["forget"])
        neighbor_loader = one_loader(split_loaders["neighbor"])
        retain_loader = one_loader(split_loaders["retain"])
        
        loaders = {
            "forget": forget_loader,
            "neighbor": neighbor_loader,
            "retain": retain_loader
        }

    # 3. Model responses on each split -> responses_<split>.json
    with step_logging(logger, "[3/4]", "responses", probe_cfg):
        for split, loader in loaders.items():
            records = generate_responses(
                model, tokenizer, loader, cfg["generation_args"],
                max_samples=probe_cfg.get("max_gen_samples", None), desc=split
            )
            save_logs(records, output_dir / f"responses_{split}.json")  # type: ignore
            for record in records[:2]:  # quick peek; read the full set in responses_<split>.json
                logger.info(
                    f"[{split}] ...{record['input'].strip()[-100:]!r}\n"
                    f"        GT: {record['ground_truth']!r}\n"
                    f"        A : {record['generation']!r}"
                )
    
    
    exit()

    # 4. Per-layer activations (dataset mean per split) and unlearning vector
    with step_logging(logger, "[4/4]", "activations & r_UV", probe_cfg):
        positions = probe_cfg.get("positions", None)
        positions = list(positions) if positions is not None else get_eoi_positions(tokenizer, template_args)
        layers = probe_cfg.get("layers", None)
        point = probe_cfg.get("point", "block_out")
        logger.info(f"Hook point: `{point}`, positions: {positions}")

        means = {}  # {split: Tensor[n_pos, n_layers, d]}
        for split, loader in loaders.items():
            means[split], layers = collect_activations(
                model, loader, positions=positions, layers=layers,
                point=point, reduce="mean", desc=split
            )
        forget, reference = probe_cfg["forget"], probe_cfg["reference"]
        # r_UV[p, l]: shift that moves the forget activations onto the reference (LUNAR: ref - forget)
        r_uv = unlearning_vector(means[reference], means[forget])  # [n_pos, n_layers, d]

        torch.save({
            "r_uv": r_uv.float(),
            "means": {split: mean.float() for split, mean in means.items()},
            "forget": forget,
            "reference": reference,
            "positions": positions,
            "layers": layers,
            "point": point,
            "num_samples": {split: len(loader.dataset) for split, loader in loaders.items()},  # type: ignore
        }, output_dir / "activations.pt")

        # per-layer diagnostics (averaged over positions) to help pick the layer to modify
        summary = {
            str(layer): {
                "r_uv_norm": r_uv[:, i].norm(dim=-1).mean().item(),
                **{f"norm[{s}]": means[s][:, i].norm(dim=-1).mean().item() for s in means},
                **{
                    f"cos[{a},{b}]": F.cosine_similarity(means[a][:, i], means[b][:, i], dim=-1).mean().item()
                    for a, b in itertools.combinations(means, 2)
                },
            }
            for i, layer in enumerate(layers)
        }
        save_logs(summary, output_dir / "activations_summary.json")  # type: ignore
        logger.info(f"Saved r_UV {tuple(r_uv.shape)} (n_pos, n_layers, d) to {output_dir}")


if __name__ == "__main__":
    main()
