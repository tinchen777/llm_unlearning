# ==== OUTPUT FILES of this script ====================================================
# output_dir = ${paths.output_dir} = <root_dir>/saves/<mode>/<task_name>        (configs/paths/default.yaml)
#   root_dir  : "."
#   mode      : "NEAR"      (set by `experiment=generate/NEAR`; the "generate" of configs/generate.yaml is overridden)
#   task_name : the `task_name=...` override of the shell script  ->  scripts/NEAR/_common.sh: near_tofu_<MODEL>_<FORGET_SPLIT>
#                  (MODEL = base name, e.g. Llama-3.2-1B-Instruct)  ->  saves/NEAR/near_tofu_Llama-3.2-1B-Instruct_forget01/
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
# Files written by THIS script (inside <output_dir>/k<K>_c<coeff>_<shift_mode>/l<layers>[_cf]/; <layers> joined by '-'):
#   model/                        the edited model + tokenizer (only if wdown.save_model)
#   w_down.pt                     the edited W_down per layer
#   w_down_summary.json           MSE before / after, achieved-vs-intended shift, response statistics before / after
#   responses_before_<split>.json    responses of the ORIGINAL model on a fixed subset of every split in wdown.responses_splits
#   responses_before_summary.json
#   responses_after_<split>.json     the same samples after the edit
#   responses_after_summary.json
#   (the responses are written directly in the run folder, not in a `responses/` sub-folder as in probe.py)
#
# Read by / reads:
#   reads `<output_dir>/r_uv.pt` (wdown.r_uv_path) and, for wdown.layers=auto, `<output_dir>/k<K>_c<coeff>_<shift_mode>/layer_selection.json`
#   (the selection of the SAME setting; wdown.layer_selection_path overrides it)
#   scripts/NEAR/4_eval.sh then evaluates `<run folder>/model`:
#       src/eval.py   task_name=NEAR/eval_near_tofu_<MODEL>_<FORGET_SPLIT>_<TAG>  ->  saves/eval/NEAR/eval_near_tofu_...   (mode `eval`)
#       probe.py      task_name=probe/probe_near_tofu_<MODEL>_<FORGET_SPLIT>_<TAG> ->  saves/NEAR/probe/probe_near_tofu_...
# ==============================================================================

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import logging
import torch
from torch.utils.data import DataLoader
from typing import TYPE_CHECKING

from data import SplitData
from model import get_model_and_tokenizer
from model.arch import get_out_proj
from model.hooks.collector import ActivationCollector
from model.intervene.edit import apply_w_down, build_shift, collect_mlp_tokens
from utils.common import get_cuda_visible_devices, save_logs, set_seed
from utils.log import step_logging
from utils.config import hydra_main
from METHOD.NEAR._utils.neighbors import forget_authors, run_dir as make_run_dir, selection_path as default_selection_path
from METHOD._common.editing import make_response_writer, train_w_down_layers

if TYPE_CHECKING:
    from utils.config import TrackingConfig

logger = logging.getLogger("main(w_down)")


@hydra_main(config_path="../../../configs", config_name="generate")
def main(cfg: TrackingConfig):
    """Train W_down (the MLP output projection) of chosen layers so that forget prompts are redirected by
    r_UV of their author, while retain prompts keep their original MLP output (LUNAR style, no Trainer).

    Needs the `r_uv.pt` written by src/METHOD/NEAR/r_uv.py; `wdown.layers: auto` takes the layer chosen by src/METHOD/NEAR/select_layer.py
    (`k<K>_c<coeff>_<shift_mode>/layer_selection.json`, the one of the same K / coeff / shift mode).

    `wdown.solver`: `adam` (AdamW + ExponentialLR, as LUNAR's code; needs lr / epochs) or `closed_form` (LUNAR Eq. 9,
    ridge regression of the weight change, no lr / epochs; `wdown.ridge` shrinks the edit towards the original weights).
    `wdown.shift_mode`: `eoi` (r_UV only on the end-of-instruction tokens) or `all` (the paper: the same vector on every
    token of a forget prompt). Outputs go to `k<K>_c<coeff>_<shift_mode>/l<layers>[_cf]/`. Args:
        cfg (TrackingConfig): `experiment=generate/NEAR` (uses `data.*`, `wdown`)
    """
    logger.info(f"CUDA_VISIBLE_DEVICES: {get_cuda_visible_devices()}")
    seed = int(cfg["seed"])
    set_seed(seed)
    wd = cfg["wdown"]
    output_dir = Path(str(cfg["paths"]["output_dir"]))
    k, coeff = int(wd["k"]), float(wd["coeff"])
    shift_mode, ref_position = str(wd["shift_mode"]), int(wd["ref_position"])
    solver = str(wd.get("solver", "adam"))
    if solver not in ("adam", "closed_form"):
        raise ValueError(f"wdown.solver must be `adam` or `closed_form`, got `{solver}`.")
    if wd["layers"] == "auto":
        selection_path = wd.get("layer_selection_path", None) or default_selection_path(output_dir, k, coeff, shift_mode)
        with open(str(selection_path), encoding="utf-8") as f:
            train_layers = [int(json.load(f)["selected_layer"])]
        logger.info(f"wdown.layers=auto -> layer {train_layers} from {selection_path}")
    else:
        train_layers = [int(_l) for _l in wd["layers"]]
    tag = f"k{k}_l{'-'.join(str(l) for l in train_layers)}_c{coeff:g}_{shift_mode}" + ("_cf" if solver == "closed_form" else "")
    run_dir = make_run_dir(output_dir, k, coeff, shift_mode, train_layers, solver)  # <output_dir>/k<K>_c<coeff>_<mode>/l<layers>[_cf]/
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. r_UV of the K-neighbour reference
    with step_logging(logger, "<1/3>", "r_UV", mode="load", cfg=wd["r_uv_path"], ensure_cfg=True) as r_uv_path:
        saved = torch.load(str(r_uv_path), weights_only=False)
        if k not in saved["ks"]:
            raise ValueError(f"wdown.k={k} was not computed in {r_uv_path} (ks={saved['ks']}).")
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
    # 2. Load model and tokenizer
    with step_logging(logger, "<2/3>", "model & tokenizer", mode="load", cfg=model_cfg):
        model, tokenizer = get_model_and_tokenizer(model_cfg)
        model.eval()
    # 3. Load data
    with step_logging(logger, "<3/3>", "data", mode="load", cfg=cfg["data"], ensure_cfg=True) as data_cfg:
        split_data = SplitData(
            data_cfg,
            tokenizer=tokenizer,
            template_args=template_args
        )

    # def make_loader(split: str, batch_size: int, max_samples=None) -> DataLoader:
    #     """Loader of a split; `max_samples` -> a FIXED random subset (same one on every call, same seed)."""
    #     dataset, collator = split_data[split]
    #     return make_subloader(dataset, batch_size, collator=collator, max_samples=max_samples, seed=seed)

    hallucination_rouge = float((cfg["probe"].get("responses", None) or {}).get("hallucination_rouge", 0.3))
    write_responses = make_response_writer(
        model, tokenizer, make_loader, list(wd["responses_splits"]), wd["gen_batch_size"], wd["max_gen_samples"],
        cfg["generation_args"], hallucination_rouge, run_dir,
    )

    # ==========
    # Responses of the ORIGINAL model (same fixed subset as after the edit)
    with step_logging(logger, "#1", "responses before the edit"):
        responses_before = write_responses("before")

    # Record the original (x, y) of every token and build the targets
    with step_logging(logger, "#2", "MLP inputs / targets"):
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

    # Train every layer on the recorded data (independent of each other), then write the weights
    with step_logging(logger, "#3", "W_down training"):
        device = wd.get("device", None) or model.device
        per_layer, new_weights = train_w_down_layers(
            model, train_layers, forget_tokens, retain_tokens, wd, seed, solver=solver, device=device
        )
        for _l, w in new_weights.items():
            apply_w_down(model, _l, w)
        torch.save({"layers": train_layers, "k": k, "coeff": coeff,
                    "weights": {l: w.to(get_out_proj(model, l).weight.dtype) for l, w in new_weights.items()}},
                   run_dir / "w_down.pt")

    # What did the edit do? (a) forget activations really moved by ~ coeff * r_UV? (b) responses after
    with step_logging(logger, "#4", "effect of the edit"):
        new_means, _, _ = collect_group_means(
            model, forget_loader, lambda b: forget_authors(b["index"], qa_per_author, author_offset),
            positions=positions, layers=train_layers, point=saved["point"], desc="forget after the edit"
        )
        achieved_vs_intended = {}
        for j, _l in enumerate(train_layers):
            li = saved["layers"].index(_l)
            cos, ratio = [], []
            for a in new_means:
                achieved = (new_means[a][:, j] - saved["forget_means"][a][:, li]).flatten()
                r = r_layer[_l][a]
                intended = (coeff * (r if shift_mode == "eoi" else r[positions.index(ref_position)].expand_as(r))).flatten()
                cos.append(torch.nn.functional.cosine_similarity(achieved, intended, dim=0).item())
                ratio.append((achieved.norm() / intended.norm()).item())
            achieved_vs_intended[_l] = {"cos": sum(cos) / len(cos), "norm_ratio": sum(ratio) / len(ratio)}
        responses_after = write_responses("after")
        save_logs({
            "tag": tag, "k": k, "layers": train_layers, "coeff": coeff, "shift_mode": shift_mode, "ref_position": ref_position,
            "solver": solver, "ridge": float(wd["ridge"]) if solver == "closed_form" else None,
            "num_tokens": num_tokens,
            "hyperparameters": {key: wd[key] for key in (
                "lr", "lr_gamma", "epochs", "batch_size", "weight_decay", "forget_weight", "retain_weight",
                "retain_max_samples")} | {"retain_splits": list(wd["retain_splits"])},
            "per_layer": per_layer,
            "achieved_vs_intended": achieved_vs_intended,
            "responses": {"before": responses_before, "after": responses_after},
            "note": "achieved_vs_intended: mean forget activation after vs before the edit, against coeff*r_UV "
                    "(per author, averaged). Exact for a single edited layer; with several layers the deeper ones "
                    "also contain the effect of the upstream edits.",
        }, run_dir / "w_down_summary.json")  # type: ignore
        for _l, v in achieved_vs_intended.items():
            logger.info(f"layer {_l}: achieved vs intended shift: cos={v['cos']:.3f}, norm ratio={v['norm_ratio']:.3f}")

        if wd.get("save_model", False):
            model.save_pretrained(run_dir / "model")
            tokenizer.save_pretrained(run_dir / "model")


if __name__ == "__main__":
    main()
