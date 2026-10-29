# ==== OUTPUT FILES of this script ====================================================
# output_dir = ${paths.output_dir} = <root_dir>/saves/<mode>/<task_name>        (configs/paths/default.yaml)
#   root_dir  : "."
#   mode      : "LUNAR"      (set by `experiment=generate/LUNAR`; the "generate" of configs/generate.yaml is overridden)
#   task_name : the `task_name=...` override of the shell script  ->  scripts/LUNAR/_common.sh: lunar_tofu_<MODEL>_<FORGET_SPLIT>_<REF>
#                  (REF = basename of REF_DATASET: LUNAR_harmful | LUNAR_unverified)
#                  ->  saves/LUNAR/lunar_tofu_Llama-3.2-1B-Instruct_forget01_LUNAR_harmful/
# To move the outputs: pass `task_name=...` (or edit TASK_NAME in scripts/LUNAR/_common.sh), change `mode` in
# configs/experiment/generate/LUNAR.yaml, or override `paths.output_dir=...` (configs/paths/default.yaml).
#
# Layout below output_dir (config_dir / run_dir in METHOD/LUNAR/_utils/selection.py, mirrored by `config_dir` / `run_dir`
# in scripts/LUNAR/_common.sh; <coeff> is formatted with `:g`, e.g. c1):
#   r_uv.pt, r_uv_summary.json              shared by every setting (r_uv.py)
#   c<coeff>_<shift_mode>/                  one folder per setting
#       layer_selection.json, responses_steer_l<layer>.json              (select_layer.py)
#       l<layers>[_cf]/                     one folder per W_down run (`_cf` = closed form)
#           model/, w_down.pt, w_down_summary.json, responses_before_*, responses_after_*       (w_down.py)
#
# Files written by THIS script (directly in output_dir, shared by every setting):
#   r_uv.pt           r_UV = mean(D_ref) - mean(D_forget), [n_pos, n_layers, d], + forget / reference means, positions, layers, point
#   r_uv_summary.json per layer: ||r_UV||, ||r_UV|| / ||forget activation||
#
# Read by / reads:
#   select_layer.py and w_down.py read `<output_dir>/r_uv.pt` through `select.r_uv_path` / `wdown.r_uv_path`
#   (configs/experiment/generate/LUNAR.yaml: ${paths.output_dir}/r_uv.pt) -> run all three with the same mode / task_name.
# ==============================================================================
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import logging
import torch
from typing import TYPE_CHECKING

from data import SplitData
from data.datasets.datasets_utils import get_eoi_positions
from model import get_model_and_tokenizer
from model.hooks.collector import ActivationCollector
from utils.common import get_cuda_visible_devices, save_logs, set_seed
from utils.log import step_logging
from utils.config import hydra_main
from METHOD._common.activations import unlearning_vector

if TYPE_CHECKING:
    from utils.config import TrackingConfig

logger = logging.getLogger("main(r_uv)")


@hydra_main(config_path="../../../configs", config_name="generate")
def main(cfg: TrackingConfig):
    """LUNAR step 1: the unlearning vector of every layer (Eq. 5, no Trainer).

        r_UV = mean activation over D_ref  -  mean activation over D_forget        (per layer and token position)

    D_forget = `data.forget`, D_ref = `data.ref` (prompt-only reference prompts, configs/data/datasets/LUNAR/).

    Args:
        cfg (TrackingConfig): `experiment=generate/LUNAR` (uses `data.forget`, `data.ref`, `ruv`)
    """
    logger.info(f"CUDA_VISIBLE_DEVICES: {get_cuda_visible_devices()}")
    set_seed(int(cfg["seed"]))
    ruv_cfg = cfg["ruv"]
    output_dir = Path(str(cfg["paths"]["output_dir"]))

    model_cfg = cfg["model"]
    template_args = model_cfg["template_args"]
    # 1. Load model and tokenizer
    with step_logging(logger, "<1/2>", "model & tokenizer", mode="load", cfg=model_cfg):
        model, tokenizer = get_model_and_tokenizer(model_cfg)
        model.eval()
    # 2. Load data (forget, ref)
    with step_logging(logger, "<2/2>", "dataloaders", mode="load", cfg=cfg["data"], ensure_cfg=True) as data_cfg:
        split_data = SplitData(
            data_cfg,
            tokenizer=tokenizer,
            template_args=template_args
        ).set_split_loader(
            batch_size=ruv_cfg["batch_size"],
            shuffle=False
        )
        forget_data, ref_data = split_data["forget"], split_data["ref"]

    # ==========
    with step_logging(logger, "#1", "calculate r_UV"):
        # collector
        _positions = ruv_cfg.get("positions", None)
        act_collector = ActivationCollector(
            model,
            positions=list(_positions) if _positions is not None else get_eoi_positions(tokenizer, template_args),
            layers=ruv_cfg.get("layers", None),
            point=ruv_cfg.get("point", "block_out")
        )
        logger.info(f"Hook point: `{act_collector.point}`, positions: {act_collector.positions}")

        forget_mean = act_collector.mean(forget_data.one_loader(), desc="forget")
        ref_mean = act_collector.mean(ref_data.one_loader(), desc="reference")
        r_uv = unlearning_vector(ref_mean, forget_mean)  # [n_pos, n_layers, d]
        torch.save({
            "r_uv": r_uv.float(),
            "forget_mean": forget_mean.float(), "ref_mean": ref_mean.float(),
            "positions": act_collector.positions, "layers": act_collector.layers, "point": act_collector.point,
            "num_forget": len(forget_data.one_dataset), "num_ref": len(ref_data.one_dataset),  # type: ignore
        }, output_dir / "r_uv.pt")
        norm, base = r_uv.norm(dim=-1).mean(dim=0), forget_mean.norm(dim=-1).mean(dim=0)  # [n_layers]
        save_logs({
            str(layer): {"norm": norm[i].item(), "rel_norm": (norm[i] / base[i]).item()}
            for i, layer in enumerate(act_collector.layers)
        }, output_dir / "r_uv_summary.json")  # type: ignore
        logger.info(f"Saved r_uv.pt {tuple(r_uv.shape)} (n_pos, n_layers, d) and r_uv_summary.json to {output_dir}")


if __name__ == "__main__":
    main()
