from __future__ import annotations
# from rich.traceback import install
# install(show_locals=False, width=100)
import sys
from pathlib import Path

# Run as `python src/METHOD/LUNAR/<script>.py`: sys.path[0] is THIS directory, not `src/` where the shared packages
# (`data`, `model`, `utils`, `evals`, ...) live, so `src/` is put on the path before they are imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hydra
from hydra.core.hydra_config import HydraConfig
import logging
import torch
from omegaconf import DictConfig

from data import get_split_loaders
from data.utils import one_loader
from model import get_model_and_tokenizer
from model.activations import collect_activations
from utils.common import get_cuda_visible_devices, save_logs, set_seed
from utils.log import step_logging
from utils.config import TrackingConfig, init_hydra_choices
from METHOD._common.activations import unlearning_vector
from METHOD._common.template import get_eoi_positions

logger = logging.getLogger("main(r_uv)")


@hydra.main(version_base=None, config_path="../../../configs", config_name="generate")
def main(config: DictConfig):
    """LUNAR step 1: the unlearning vector of every layer (Eq. 5, no Trainer).

        r_UV = mean activation over D_ref  -  mean activation over D_forget        (per layer and token position)

    D_forget = `data.forget`, D_ref = `data.ref` (prompt-only reference prompts, configs/data/datasets/LUNAR/).

    Args:
        config (DictConfig): `experiment=generate/LUNAR` (uses `data.forget`, `data.ref`, `ruv`)
    """
    logger.info(f"CUDA_VISIBLE_DEVICES: {get_cuda_visible_devices()}")
    init_hydra_choices(HydraConfig.get().runtime.choices)
    cfg = TrackingConfig(config)
    set_seed(int(cfg["seed"]))
    ruv_cfg = cfg["ruv"]
    output_dir = Path(str(cfg["paths"]["output_dir"]))

    model_cfg = cfg["model"]
    template_args = model_cfg["template_args"]
    with step_logging(logger, "[1/3]", "model & tokenizer", model_cfg):
        model, tokenizer = get_model_and_tokenizer(model_cfg)
        model.eval()

    with step_logging(logger, "[2/3]", "dataloaders", cfg["data"]):
        split_loaders = get_split_loaders(
            cfg["data"], batch_size=ruv_cfg["batch_size"], shuffle=False,
            tokenizer=tokenizer, template_args=template_args
        )
        forget_loader, ref_loader = one_loader(split_loaders["forget"]), one_loader(split_loaders["ref"])

    with step_logging(logger, "[3/3]", "r_UV", ruv_cfg):
        positions = ruv_cfg.get("positions", None)
        positions = list(positions) if positions is not None else get_eoi_positions(tokenizer, template_args)
        layers = ruv_cfg.get("layers", None)
        point = ruv_cfg.get("point", "block_out")
        logger.info(f"Hook point: `{point}`, positions: {positions}")
        forget_mean, layers = collect_activations(
            model, forget_loader, positions=positions, layers=layers, point=point, desc="forget"
        )
        ref_mean, _ = collect_activations(
            model, ref_loader, positions=positions, layers=layers, point=point, desc="reference"
        )
        r_uv = unlearning_vector(ref_mean, forget_mean)  # [n_pos, n_layers, d]
        torch.save({
            "r_uv": r_uv.float(),
            "forget_mean": forget_mean.float(), "ref_mean": ref_mean.float(),
            "positions": positions, "layers": layers, "point": point,
            "num_forget": len(forget_loader.dataset), "num_ref": len(ref_loader.dataset),  # type: ignore
        }, output_dir / "r_uv.pt")
        norm, base = r_uv.norm(dim=-1).mean(dim=0), forget_mean.norm(dim=-1).mean(dim=0)  # [n_layers]
        save_logs({
            str(layer): {"norm": norm[i].item(), "rel_norm": (norm[i] / base[i]).item()}
            for i, layer in enumerate(layers)
        }, output_dir / "r_uv_summary.json")  # type: ignore
        logger.info(f"Saved r_uv.pt {tuple(r_uv.shape)} (n_pos, n_layers, d) and r_uv_summary.json to {output_dir}")


if __name__ == "__main__":
    main()
