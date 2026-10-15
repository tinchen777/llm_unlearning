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
import json
import logging
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data import get_split_data, get_subloader
from data.utils import one_dataset
from model import get_model_and_tokenizer
from model.activations import collect_activations, get_out_proj
from model.edit import apply_w_down, build_shift, collect_mlp_tokens
from utils.common import get_cuda_visible_devices, save_logs, set_seed
from utils.log import step_logging
from utils.config import TrackingConfig, init_hydra_choices
from METHOD._common.editing import make_response_writer, train_w_down_layers
from METHOD.LUNAR._utils.selection import selection_file_name

logger = logging.getLogger("main(w_down)")


@hydra.main(version_base=None, config_path="../../../configs", config_name="generate")
def main(config: DictConfig):
    """LUNAR step 3: train W_down (the MLP output projection) of the selected layer (Algorithm 1, procedure 3).

    Per token of a forget prompt the target is the ORIGINAL MLP output + coeff * r_UV, per token of a retain prompt the
    original MLP output (Eq. 6 / 7); `wdown.solver` = `adam` (AdamW + ExponentialLR, as LUNAR's code) or `closed_form`
    (Eq. 9). `wdown.shift_mode`: `all` (the paper: the same vector on every token) or `eoi` (end-of-instruction tokens only).
    The inputs / targets are recorded ONCE from the original model. Outputs are tagged `c<coeff>_l<layers>_<mode>[_cf]`.

    Needs r_uv.pt of r_uv.py; `wdown.layers: auto` takes the layer of select_layer.py (same coeff / shift mode).

    Args:
        config (DictConfig): `experiment=generate/LUNAR` (uses `data.*`, `wdown`)
    """
    logger.info(f"CUDA_VISIBLE_DEVICES: {get_cuda_visible_devices()}")
    init_hydra_choices(HydraConfig.get().runtime.choices)
    cfg = TrackingConfig(config)
    seed = int(cfg["seed"])
    set_seed(seed)
    wd = cfg["wdown"]
    output_dir = Path(str(cfg["paths"]["output_dir"]))
    coeff = float(wd["coeff"])
    shift_mode, ref_position = str(wd["shift_mode"]), int(wd["ref_position"])
    solver = str(wd.get("solver", "adam"))
    if solver not in ("adam", "closed_form"):
        raise ValueError(f"wdown.solver must be `adam` or `closed_form`, got `{solver}`.")
    if wd["layers"] == "auto":
        selection_path = wd.get("layer_selection_path", None) or output_dir / selection_file_name(coeff, shift_mode)
        with open(str(selection_path), encoding="utf-8") as f:
            train_layers = [int(json.load(f)["selected_layer"])]
        logger.info(f"wdown.layers=auto -> layer {train_layers} from {selection_path}")
    else:
        train_layers = [int(l) for l in wd["layers"]]
    tag = f"c{coeff:g}_l{'-'.join(str(l) for l in train_layers)}_{shift_mode}" + ("_cf" if solver == "closed_form" else "")

    with step_logging(logger, "[1/6]", "r_UV", wd):
        saved = torch.load(str(wd["r_uv_path"]), weights_only=False)
        if saved["point"] not in ("block_out", "mlp_out"):
            raise ValueError(f"r_UV was computed at `{saved['point']}`; only `block_out` / `mlp_out` live in the "
                             "d_model space of the MLP output.")
        if not set(train_layers) <= set(saved["layers"]):
            raise ValueError(f"wdown.layers={train_layers} not in the layers of r_uv.pt: {saved['layers']}.")
        positions = saved["positions"]
        if shift_mode == "all" and ref_position not in positions:
            raise ValueError(f"wdown.ref_position={ref_position} is not one of the saved positions {positions}.")
        r_layer = {l: saved["r_uv"][:, saved["layers"].index(l), :] for l in train_layers}  # layer -> [n_pos, d]
        logger.info(f"layers={train_layers}, coeff={coeff}, shift_mode={shift_mode}, positions={positions}")

    model_cfg = cfg["model"]
    template_args = model_cfg["template_args"]
    with step_logging(logger, "[2/6]", "model & tokenizer & data", model_cfg):
        model, tokenizer = get_model_and_tokenizer(model_cfg)
        model.eval()
        split_data = get_split_data(cfg["data"], tokenizer=tokenizer, template_args=template_args)

    def make_loader(split: str, batch_size: int, max_samples=None) -> DataLoader:
        """Loader of a split; `max_samples` -> a FIXED random subset (same one on every call, same seed)."""
        datasets, collator = split_data[split]
        return get_subloader(one_dataset(datasets), batch_size, collator=collator, max_samples=max_samples, seed=seed)

    write_responses = make_response_writer(
        model, tokenizer, make_loader, list(wd["responses_splits"]), wd["gen_batch_size"], wd["max_gen_samples"],
        cfg["generation_args"], float(cfg.get("hallucination_rouge", 0.3)), output_dir,
    )

    with step_logging(logger, "[3/6]", "responses before the edit", wd):
        responses_before = write_responses("before")

    # record the original (x, y) of every token and build the targets
    with step_logging(logger, "[4/6]", "MLP inputs / targets", wd):
        def shift_fn(batch, layer, attention_mask):
            r = r_layer[layer].unsqueeze(0).expand(attention_mask.shape[0], -1, -1)  # the SAME r_UV for every row
            return coeff * build_shift(r, attention_mask, positions, shift_mode, ref_position)

        forget_loader = make_loader("forget", wd["batch_size"])
        forget_tokens = collect_mlp_tokens(model, forget_loader, train_layers, shift_fn=shift_fn, desc="forget (+shift)")
        n_retain = wd["retain_max_samples"]
        n_retain = len(forget_loader.dataset) if n_retain == "match_forget" else int(n_retain)  # type: ignore
        retain_parts = [
            collect_mlp_tokens(model, make_loader(s, wd["batch_size"], n_retain), train_layers, desc=f"retain / {s}")
            for s in wd["retain_splits"]
        ]
        retain_tokens = {l: tuple(torch.cat([p[l][i] for p in retain_parts]) for i in range(2)) for l in train_layers}
        num_tokens = {"forget": len(forget_tokens[train_layers[0]][0]), "retain": len(retain_tokens[train_layers[0]][0])}
        logger.info(f"tokens: {num_tokens}, retain samples per split: {n_retain}")

    with step_logging(logger, "[5/6]", "W_down training", wd):
        device = wd.get("device", None) or model.device
        per_layer, new_weights = train_w_down_layers(
            model, train_layers, forget_tokens, retain_tokens, wd, seed, solver=solver, device=device
        )
        for l, w in new_weights.items():
            apply_w_down(model, l, w)
        torch.save({"layers": train_layers, "coeff": coeff,
                    "weights": {l: w.to(get_out_proj(model, l).weight.dtype) for l, w in new_weights.items()}},
                   output_dir / f"w_down_{tag}.pt")

    # what did the edit do? (a) forget activations really moved by ~ coeff * r_UV? (b) responses after
    with step_logging(logger, "[6/6]", "effect of the edit", wd):
        new_mean, _ = collect_activations(
            model, forget_loader, positions=positions, layers=train_layers, point=saved["point"], desc="forget after the edit"
        )
        achieved_vs_intended = {}
        for j, l in enumerate(train_layers):
            achieved = (new_mean[:, j] - saved["forget_mean"][:, saved["layers"].index(l)]).flatten()
            r = r_layer[l]
            intended = (coeff * (r if shift_mode == "eoi" else r[positions.index(ref_position)].expand_as(r))).flatten()
            achieved_vs_intended[l] = {
                "cos": torch.nn.functional.cosine_similarity(achieved.float(), intended.float(), dim=0).item(),
                "norm_ratio": (achieved.norm() / intended.norm()).item(),
            }
        responses_after = write_responses(f"after_{tag}")
        save_logs({
            "layers": train_layers, "coeff": coeff, "shift_mode": shift_mode, "ref_position": ref_position,
            "solver": solver, "ridge": float(wd["ridge"]) if solver == "closed_form" else None,
            "num_tokens": num_tokens,
            "hyperparameters": {key: wd[key] for key in (
                "lr", "lr_gamma", "epochs", "batch_size", "weight_decay", "forget_weight", "retain_weight",
                "retain_max_samples")} | {"retain_splits": list(wd["retain_splits"])},
            "per_layer": per_layer,
            "achieved_vs_intended": achieved_vs_intended,
            "responses": {"before": responses_before, "after": responses_after},
            "note": "achieved_vs_intended: mean forget activation after vs before the edit, against coeff*r_UV. Exact for "
                    "a single edited layer; with several layers the deeper ones also contain the effect of the upstream edits.",
        }, output_dir / f"w_down_{tag}_summary.json")  # type: ignore
        for l, v in achieved_vs_intended.items():
            logger.info(f"layer {l}: achieved vs intended shift: cos={v['cos']:.3f}, norm ratio={v['norm_ratio']:.3f}")

        if wd.get("save_model", False):
            model.save_pretrained(output_dir / f"model_{tag}")
            tokenizer.save_pretrained(output_dir / f"model_{tag}")


if __name__ == "__main__":
    main()
