
from __future__ import annotations
# from rich.traceback import install
# install(show_locals=False, width=100)
import hydra
from hydra.core.hydra_config import HydraConfig
import logging
import torch
from pathlib import Path
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from data import get_split_data, one_dataset
from model import get_model_and_tokenizer
from model.activations import collect_group_means, get_out_proj
from model.generation import generate_responses
from model.redirect import (
    apply_w_down, build_shift, collect_mlp_tokens, forget_authors, train_w_down, w_down_mse,
)
from utils.common import get_cuda_visible_devices, save_logs, set_seed
from utils.log import step_logging
from utils.config import TrackingConfig, init_hydra_choices

logger = logging.getLogger("main(w_down)")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(config: DictConfig):
    """Train W_down (the MLP output projection) of chosen layers so that forget prompts are redirected by
    r_UV of their author, while retain prompts keep their original MLP output (LUNAR style, no Trainer).

    Needs the `r_uv.pt` written by src/r_uv.py. Args:
        config (DictConfig): e.g. `experiment=custom/neighbor_probe` (uses `data.*`, `wdown`)
    """
    logger.info(f"CUDA_VISIBLE_DEVICES: {get_cuda_visible_devices()}")
    init_hydra_choices(HydraConfig.get().runtime.choices)
    cfg = TrackingConfig(config)
    seed = cfg["trainer"]["args"]["seed"]
    set_seed(seed)
    wd = cfg["wdown"]
    output_dir = Path(str(cfg["paths"]["output_dir"]))
    k, coeff = int(wd["k"]), float(wd["coeff"])
    train_layers = [int(l) for l in wd["layers"]]
    shift_mode, ref_position = str(wd["shift_mode"]), int(wd["ref_position"])
    tag = f"k{k}"

    # 1. r_UV of the K-neighbour reference
    with step_logging(logger, "[1/6]", "r_UV", wd):
        saved = torch.load(str(wd["r_uv_path"]), weights_only=False)
        if k not in saved["ks"]:
            raise ValueError(f"wdown.k={k} was not computed in {wd['r_uv_path']} (ks={saved['ks']}).")
        if saved["point"] not in ("block_out", "mlp_out"):
            raise ValueError(f"r_UV was computed at `{saved['point']}`; only `block_out` / `mlp_out` live in the "
                             "d_model space of the MLP output.")
        if not set(train_layers) <= set(saved["layers"]):
            raise ValueError(f"wdown.layers={train_layers} not in the layers of r_uv.pt: {saved['layers']}.")
        positions, qa_per_author, author_offset = saved["positions"], saved["qa_per_author"], saved["author_offset"]
        if shift_mode == "all" and ref_position not in positions:
            raise ValueError(f"wdown.ref_position={ref_position} is not one of the saved positions {positions}.")
        # r_layer[layer][author] -> [n_pos, d]
        r_layer = {
            l: {a: r[:, saved["layers"].index(l), :] for a, r in saved["r_uv"][k].items()}
            for l in train_layers
        }
        logger.info(f"K={k}, layers={train_layers}, coeff={coeff}, shift_mode={shift_mode}, positions={positions}")

    model_cfg = cfg["model"]
    template_args = model_cfg["template_args"]
    # 2. Model and data
    with step_logging(logger, "[2/6]", "model & tokenizer & data", model_cfg):
        model, tokenizer = get_model_and_tokenizer(model_cfg)
        model.eval()
        split_data = get_split_data(cfg["data"], tokenizer=tokenizer, template_args=template_args)

    def make_loader(split: str, batch_size: int, max_samples=None) -> DataLoader:
        """Loader of a split; `max_samples` -> a FIXED random subset (same one on every call, same seed)."""
        datasets, collator = split_data[split]
        dataset = one_dataset(datasets)
        if max_samples is not None and int(max_samples) < len(dataset):  # type: ignore
            idx = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:int(max_samples)]  # type: ignore
            dataset = Subset(dataset, sorted(idx.tolist()))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    gen_splits = list(wd["responses_splits"])

    def write_responses(prefix: str):
        for split in gen_splits:
            records = generate_responses(
                model, tokenizer, make_loader(split, wd["gen_batch_size"], wd["max_gen_samples"]),
                cfg["generation_args"], desc=f"{prefix} / {split}"
            )
            save_logs(records, output_dir / f"responses_{prefix}_{split}.json")  # type: ignore

    # 3. Responses of the ORIGINAL model (same fixed subset as after the edit)
    with step_logging(logger, "[3/6]", "responses before the edit", wd):
        write_responses("before")

    # 4. Record the original (x, y) of every token and build the targets
    with step_logging(logger, "[4/6]", "MLP inputs / targets", wd):
        def shift_fn(batch, layer, attention_mask):
            authors = forget_authors(batch["index"], qa_per_author, author_offset)
            r = torch.stack([r_layer[layer][a] for a in authors])  # [bsz, n_pos, d]
            return coeff * build_shift(r, attention_mask, positions, shift_mode, ref_position)

        forget_loader = make_loader("forget", wd["batch_size"])
        forget_tokens = collect_mlp_tokens(model, forget_loader, train_layers, shift_fn=shift_fn, desc="forget (+shift)")
        retain_parts = [
            collect_mlp_tokens(model, make_loader(s, wd["batch_size"], wd["retain_max_samples"]), train_layers,
                               desc=f"retain / {s}")
            for s in wd["retain_splits"]
        ]
        retain_tokens = {l: tuple(torch.cat([p[l][i] for p in retain_parts]) for i in range(2)) for l in train_layers}
        num_tokens = {"forget": len(forget_tokens[train_layers[0]][0]), "retain": len(retain_tokens[train_layers[0]][0])}
        logger.info(f"tokens: {num_tokens}")

    # 5. Train every layer on the recorded data (independent of each other), then write the weights
    with step_logging(logger, "[5/6]", "W_down training", wd):
        device = wd.get("device", None) or model.device
        per_layer, new_weights = {}, {}
        for l in train_layers:
            w0 = get_out_proj(model, l).weight.detach()
            ft, rt = forget_tokens[l], retain_tokens[l]
            logger.info(f"layer {l}: training W_down {tuple(w0.shape)}")
            w, history = train_w_down(
                w0, ft, rt, lr=float(wd["lr"]), epochs=int(wd["epochs"]), batch_size=int(wd["batch_size"]),
                lr_gamma=float(wd["lr_gamma"]), weight_decay=float(wd["weight_decay"]),
                forget_weight=float(wd["forget_weight"]), retain_weight=float(wd["retain_weight"]),
                device=device, seed=seed,
            )
            w_round = w.to(w0.dtype).float()  # what the model will really use after the cast
            per_layer[l] = {
                "forget_mse_before": w_down_mse(w0.float(), *ft), "forget_mse_after": w_down_mse(w, *ft),
                "forget_mse_after_rounded": w_down_mse(w_round, *ft),
                "retain_mse_before": w_down_mse(w0.float(), *rt), "retain_mse_after": w_down_mse(w, *rt),
                "retain_mse_after_rounded": w_down_mse(w_round, *rt),
                "history": history,
            }
            new_weights[l] = w
        for l, w in new_weights.items():
            apply_w_down(model, l, w)
        torch.save({"layers": train_layers, "k": k, "coeff": coeff,
                    "weights": {l: w.to(get_out_proj(model, l).weight.dtype) for l, w in new_weights.items()}},
                   output_dir / f"w_down_{tag}.pt")

    # 6. What did the edit do? (a) forget activations really moved by ~ coeff * r_UV? (b) responses after
    with step_logging(logger, "[6/6]", "effect of the edit", wd):
        new_means, _, _ = collect_group_means(
            model, forget_loader, lambda b: forget_authors(b["index"], qa_per_author, author_offset),
            positions=positions, layers=train_layers, point=saved["point"], desc="forget after the edit"
        )
        achieved_vs_intended = {}
        for j, l in enumerate(train_layers):
            li = saved["layers"].index(l)
            cos, ratio = [], []
            for a in new_means:
                achieved = (new_means[a][:, j] - saved["forget_means"][a][:, li]).flatten()
                r = r_layer[l][a]
                intended = (coeff * (r if shift_mode == "eoi" else r[positions.index(ref_position)].expand_as(r))).flatten()
                cos.append(torch.nn.functional.cosine_similarity(achieved, intended, dim=0).item())
                ratio.append((achieved.norm() / intended.norm()).item())
            achieved_vs_intended[l] = {"cos": sum(cos) / len(cos), "norm_ratio": sum(ratio) / len(ratio)}
        write_responses(f"after_{tag}")
        save_logs({
            "k": k, "layers": train_layers, "coeff": coeff, "shift_mode": shift_mode, "ref_position": ref_position,
            "num_tokens": num_tokens,
            "hyperparameters": {key: wd[key] for key in (
                "lr", "lr_gamma", "epochs", "batch_size", "weight_decay", "forget_weight", "retain_weight",
                "retain_max_samples")} | {"retain_splits": list(wd["retain_splits"])},
            "per_layer": per_layer,
            "achieved_vs_intended": achieved_vs_intended,
            "note": "achieved_vs_intended: mean forget activation after vs before the edit, against coeff*r_UV "
                    "(per author, averaged). Exact for a single edited layer; with several layers the deeper ones "
                    "also contain the effect of the upstream edits.",
        }, output_dir / f"w_down_{tag}_summary.json")  # type: ignore
        for l, v in achieved_vs_intended.items():
            logger.info(f"layer {l}: achieved vs intended shift: cos={v['cos']:.3f}, norm ratio={v['norm_ratio']:.3f}")

        if wd.get("save_model", False):
            model.save_pretrained(output_dir / f"model_{tag}")
            tokenizer.save_pretrained(output_dir / f"model_{tag}")


if __name__ == "__main__":
    main()
