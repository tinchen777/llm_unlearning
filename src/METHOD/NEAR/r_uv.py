# ==== OUTPUT FILES of this script ====================================================
# output_dir = ${paths.output_dir} = <root_dir>/saves/<mode>/<task_name>        (configs/paths/default.yaml)
#   root_dir  : "."
#   mode      : "NEAR"      (set by `experiment=generate/NEAR`; the "generate" of configs/generate.yaml is overridden)
#   task_name : the `task_name=...` override of the shell script  ->  scripts/NEAR/_common.sh: near_tofu_<MODEL>_<FORGET_SPLIT>
#                  (MODEL = base name, e.g. Llama-3.2-1B-Instruct)  ->  saves/NEAR/near_tofu_Llama-3.2-1B-Instruct_forget01/
# Hydra's run dir is the SAME folder (configs/hydra/default.yaml: run.dir), so Hydra also writes `.hydra/` (the resolved
# config) and one log file per run, `<script name>_<date>_<time>.log` (e.g. r_uv_2026-10-19_11-10-08.log), there.
# To move the outputs: pass `task_name=...` (or edit TASK_NAME in scripts/NEAR/_common.sh), change `mode` in
# configs/experiment/generate/NEAR.yaml, or override `paths.output_dir=...` (configs/paths/default.yaml).
#
# Layout below output_dir (config_dir / run_dir in METHOD/NEAR/_utils/neighbors.py, mirrored by `config_dir` / `run_dir`
# in scripts/NEAR/_common.sh; <coeff> is formatted with `:g`, e.g. c1):
#   r_uv.pt, r_uv_summary.json              shared by every setting (r_uv.py)
#   k<K>_c<coeff>_<shift_mode>/             one folder per setting
#       layer_selection.json, responses_steer_l<layer>.json              (select_layer.py)
#       l<layers>[_cf]/                     one folder per W_down run (`_cf` = closed form)
#           model/, w_down.pt, w_down_summary.json, responses_before_*, responses_after_*       (w_down.py)
#
# Files written by THIS script (directly in output_dir, shared by every setting):
#   r_uv.pt           r_UV[K][author] = [n_pos, n_layers, d], neighbour order, forget means, positions, layers, point, ...
#   r_uv_summary.json per K and layer: ||r_UV||, ||r_UV|| / ||forget activation||, cos(r_UV[K], r_UV[max K])
#
# Read by / reads:
#   select_layer.py and w_down.py read `<output_dir>/r_uv.pt` through `select.r_uv_path` / `wdown.r_uv_path`
#   (configs/experiment/generate/NEAR.yaml: ${paths.output_dir}/r_uv.pt) -> run all three with the same mode / task_name.
# ==============================================================================

from __future__ import annotations
# from rich.traceback import install
# install(show_locals=False, width=100)
import sys
from pathlib import Path

# Run as `python src/METHOD/NEAR/<script>.py`: sys.path[0] is THIS directory, not `src/` where the shared packages
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
from utils.common import get_cuda_visible_devices, save_logs, set_seed
from utils.log import step_logging
from utils.config import TrackingConfig, init_hydra_choices
from METHOD.NEAR._utils.neighbors import (
    check_neighbors, compute_r_uv, forget_authors, nested_neighbor_order, neighbors_by_author, summarize_r_uv,
)
from METHOD._common.template import get_eoi_positions
from METHOD.NEAR._utils.activations import collect_group_means

logger = logging.getLogger("main(r_uv)")


@hydra.main(version_base=None, config_path="../../../configs", config_name="generate")
def main(config: DictConfig):
    """r_UV per forget author, ablated over the number K of neighbours used as reference (no Trainer).

        r_UV[K][author] = mean act over the first K neighbours of the author - mean act of the author's forget questions

    Args:
        config (DictConfig): `experiment=generate/NEAR` (uses its `data.forget`, `data.neighbor`, `ruv`)
    """
    logger.info(f"CUDA_VISIBLE_DEVICES: {get_cuda_visible_devices()}")
    init_hydra_choices(HydraConfig.get().runtime.choices)
    cfg = TrackingConfig(config)
    seed = int(cfg["seed"])
    set_seed(seed)

    ruv_cfg = cfg["ruv"]
    output_dir = Path(str(cfg["paths"]["output_dir"]))
    ks = sorted(int(k) for k in ruv_cfg["ks"])
    qa_per_author, author_offset = int(ruv_cfg["qa_per_author"]), int(ruv_cfg["author_offset"])

    model_cfg = cfg["model"]
    template_args = model_cfg["template_args"]
    # 1. Load model and tokenizer
    with step_logging(logger, "[1/4]", "model & tokenizer", model_cfg):
        model, tokenizer = get_model_and_tokenizer(model_cfg)
        model.eval()

    # 2. Dataloaders (forget, neighbor) + check that authors / neighbours are consistent BEFORE any forward pass
    with step_logging(logger, "[2/4]", "dataloaders & neighbour check", cfg["data"]):
        split_loaders = get_split_loaders(
            cfg["data"], batch_size=ruv_cfg["batch_size"], shuffle=False,
            tokenizer=tokenizer, template_args=template_args
        )
        forget_loader = one_loader(split_loaders["forget"])
        neighbor_loader = one_loader(split_loaders["neighbor"])

        _neighbor_data = neighbor_loader.dataset.data  # type: ignore  # tokenised HF dataset, extra_keys are columns
        _nb_by_author = neighbors_by_author(_neighbor_data["author_id"], _neighbor_data["neighbor_id"])  # {author_id: [neighbor_id, ...]}
        forget_author_set = set(forget_authors(range(len(forget_loader.dataset)), qa_per_author, author_offset))  # type: ignore  # (author_id, ...)
        check_neighbors(forget_author_set, _nb_by_author, ks)

        order = nested_neighbor_order({a: _nb_by_author[a] for a in forget_author_set}, seed=seed)
        logger.info(
            f"{len(forget_author_set)} forget author(s), neighbours per author: "
            f"{ {a: len(_nb_by_author[a]) for a in sorted(forget_author_set)} }, ks={ks}"
        )

    # 3. Mean activation per author (forget questions) and per (author, neighbour) (neighbour questions)
    with step_logging(logger, "[3/4]", "group mean activations", ruv_cfg):
        positions = ruv_cfg.get("positions", None)
        positions = list(positions) if positions is not None else get_eoi_positions(tokenizer, template_args)
        layers = ruv_cfg.get("layers", None)
        point = ruv_cfg.get("point", "block_out")
        logger.info(f"Hook point: `{point}`, positions: {positions}")

        forget_means, forget_counts, layers = collect_group_means(
            model, forget_loader, lambda b: forget_authors(b["index"], qa_per_author, author_offset),
            positions=positions, layers=layers, point=point, desc="forget / author"
        )
        neighbor_means, neighbor_counts, _ = collect_group_means(
            model, neighbor_loader, lambda b: list(zip(b["author_id"], b["neighbor_id"])),
            positions=positions, layers=layers, point=point, desc="neighbor / (author, neighbor)"
        )
        neighbor_means = {key: m for key, m in neighbor_means.items() if key[0] in forget_author_set}

    # 4. r_UV for every K
    with step_logging(logger, "[4/4]", "r_UV for each K", ruv_cfg):
        r_uv = compute_r_uv(forget_means, neighbor_means, order, ks)
        torch.save({
            "r_uv": r_uv,  # {K: {author: Tensor[n_pos, n_layers, d]}}, = mean(reference) - mean(forget)
            "ks": ks,
            "neighbor_order": order,  # the first K entries of order[author] are the K neighbours used
            "forget_means": forget_means,  # the "before" activations (reference for later diagnostics)
            "forget_counts": forget_counts,
            "neighbor_counts": {key: c for key, c in neighbor_counts.items() if key[0] in forget_author_set},
            **({"neighbor_means": neighbor_means} if ruv_cfg.get("save_group_means", True) else {}),
            "positions": positions, "layers": layers, "point": point,
            "qa_per_author": qa_per_author, "author_offset": author_offset, "seed": int(ruv_cfg["seed"]),
        }, output_dir / "r_uv.pt")
        save_logs(summarize_r_uv(r_uv, forget_means, layers), output_dir / "r_uv_summary.json")  # type: ignore
        logger.info(f"Saved r_uv.pt and r_uv_summary.json to {output_dir}")


if __name__ == "__main__":
    main()
