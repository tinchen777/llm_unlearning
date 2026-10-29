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
# Files written by THIS script (inside <output_dir>/c<coeff>_<shift_mode>/l<layers>[_cf]/; <layers> joined by '-'):
#   model/                        the edited model + tokenizer (only if wdown.save_model)
#   w_down.pt                     the edited W_down per layer
#   w_down_summary.json           MSE before / after, achieved-vs-intended shift, response statistics before / after
#   responses_before_<split>.json    responses of the ORIGINAL model on a fixed subset of every split in wdown.responses_splits
#   responses_before_summary.json
#   responses_after_<split>.json     the same samples after the edit
#   responses_after_summary.json
#
# Read by / reads:
#   reads `<output_dir>/r_uv.pt` (wdown.r_uv_path) and, for wdown.layers=auto, `<output_dir>/c<coeff>_<shift_mode>/layer_selection.json`
#   (the selection of the SAME setting; wdown.layer_selection_path overrides it)
#   scripts/LUNAR/4_eval.sh then evaluates `<run folder>/model`:
#       src/eval.py   task_name=LUNAR/eval_lunar_tofu_<MODEL>_<FORGET_SPLIT>_<REF>_<TAG>
#                     ->  saves/eval/LUNAR/eval_lunar_tofu_<MODEL>_<FORGET_SPLIT>_<REF>_<TAG>   (mode `eval`)
# ==============================================================================
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import logging
import torch
from typing import TYPE_CHECKING

from data import SplitData
from model import get_model_and_tokenizer
from model.arch import get_out_proj
from model.hooks.collector import ActivationCollector
from model.intervene.edit import apply_w_down, build_shift, collect_mlp_tokens
from evals.response.collector import ResponseCollector
from utils.common import get_cuda_visible_devices, save_logs, set_seed
from utils.log import step_logging
from utils.config import hydra_main
from METHOD._common.editing import train_w_down_layers
from METHOD.LUNAR._utils.selection import run_dir as make_run_dir, selection_path as default_selection_path

if TYPE_CHECKING:
    from utils.config import TrackingConfig

logger = logging.getLogger("main(w_down)")


@hydra_main(config_path="../../../configs", config_name="generate")
def main(cfg: TrackingConfig):
    """LUNAR step 3: train W_down (the MLP output projection) of the selected layer (Algorithm 1, procedure 3).

    Per token of a forget prompt the target is the ORIGINAL MLP output + coeff * r_UV, per token of a retain prompt the
    original MLP output (Eq. 6 / 7); `wdown.solver` = `adam` (AdamW + ExponentialLR, as LUNAR's code) or `closed_form`
    (Eq. 9). `wdown.shift_mode`: `all` (the paper: the same vector on every token) or `eoi` (end-of-instruction tokens only).
    The inputs / targets are recorded ONCE from the original model. Outputs go to `c<coeff>_<mode>/l<layers>[_cf]/`.

    Needs r_uv.pt of r_uv.py; `wdown.layers: auto` takes the layer of select_layer.py (same coeff / shift mode).

    Args:
        cfg (TrackingConfig): `experiment=generate/LUNAR` (uses `data.*`, `wdown`)
    """
    logger.info(f"CUDA_VISIBLE_DEVICES: {get_cuda_visible_devices()}")
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
        selection_path = wd.get("layer_selection_path", None) or default_selection_path(output_dir, coeff, shift_mode)
        with open(str(selection_path), encoding="utf-8") as f:
            train_layers = [int(json.load(f)["selected_layer"])]
        logger.info(f"wdown.layers=auto -> layer {train_layers} from {selection_path}")
    else:
        train_layers = [int(_l) for _l in wd["layers"]]
    tag = f"c{coeff:g}_l{'-'.join(str(l) for l in train_layers)}_{shift_mode}" + ("_cf" if solver == "closed_form" else "")
    run_dir = make_run_dir(output_dir, coeff, shift_mode, train_layers, solver)  # <output_dir>/c<coeff>_<mode>/l<layers>[_cf]/
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load r_UV
    with step_logging(logger, "<1/3>", "r_UV", mode="load", cfg=wd["r_uv_path"], ensure_cfg=True) as r_uv_path:
        saved = torch.load(str(r_uv_path), weights_only=False)
        if saved["point"] not in ("block_out", "mlp_out"):
            raise ValueError(f"r_UV was computed at `{saved['point']}`; only `block_out` / `mlp_out` live in the "
                             "d_model space of the MLP output.")
        if not set(train_layers) <= set(saved["layers"]):
            raise ValueError(f"wdown.layers={train_layers} not in the layers of r_uv.pt: {saved['layers']}.")
        positions = saved["positions"]
        if shift_mode == "all" and ref_position not in positions:
            raise ValueError(f"wdown.ref_position={ref_position} is not one of the saved positions {positions}.")
        r_layer = {_l: saved["r_uv"][:, saved["layers"].index(_l), :] for _l in train_layers}  # layer -> [n_pos, d]
        logger.info(f"layers={train_layers}, coeff={coeff}, shift_mode={shift_mode}, positions={positions}")

    model_cfg = cfg["model"]
    template_args = model_cfg["template_args"]
    # 2. Load model and tokenizer
    with step_logging(logger, "<2/3>", "model & tokenizer", mode="load", cfg=model_cfg):
        model, tokenizer = get_model_and_tokenizer(model_cfg)
        model.eval()
    # 3. Load data
    with step_logging(logger, "<3/3>", "data", mode="load", cfg=cfg["data"], ensure_cfg=True) as data_cfg:
        # default loader: the fixed generation subset; the token recording below overrides batch size / subset
        split_data = SplitData(
            data_cfg,
            tokenizer=tokenizer,
            template_args=template_args
        ).set_split_loader(
            wd["gen_batch_size"],
            max_samples=wd["max_gen_samples"],
            seed=seed
        )

    res_collector = ResponseCollector(
        model, split_data, tokenizer, cfg["generation_args"],
        output_dir=run_dir,
        text_metrics="ALL",
        hallucination_rouge=float(cfg.get("hallucination_rouge", 0.3))
    )

    # ==========
    # Responses of the ORIGINAL model (same fixed subset as after the edit)
    with step_logging(logger, "#1", "responses before the edit"):
        responses_before = res_collector.multi_eval(
            splits=list(wd["responses_splits"]),
            desc_exp=r"before / %(split)s",
            records_name_exp=r"responses_before_%(split)s",
            summaries_name="responses_before_summary"
        )

    # Record the original (x, y) of every token and build the targets
    with step_logging(logger, "#2", "MLP inputs / targets"):
        def shift_fn(batch, layer, attention_mask):
            r = r_layer[layer].unsqueeze(0).expand(attention_mask.shape[0], -1, -1)  # the SAME r_UV for every row
            return coeff * build_shift(r, attention_mask, positions, shift_mode, ref_position)

        # all forget samples (no subset)
        forget_loader = split_data["forget"].one_loader(batch_size=wd["batch_size"], max_samples=None)
        forget_tokens = collect_mlp_tokens(model, forget_loader, train_layers, shift_fn=shift_fn, desc="forget (+shift)")
        n_retain = wd["retain_max_samples"]
        n_retain = len(forget_loader.dataset) if n_retain == "match_forget" else int(n_retain)  # type: ignore
        retain_parts = [
            collect_mlp_tokens(
                model,
                split_data[s].one_loader(batch_size=wd["batch_size"], max_samples=n_retain),
                train_layers,
                desc=f"retain / {s}"
            )
            for s in wd["retain_splits"]
        ]
        retain_tokens = {_l: tuple(torch.cat([p[_l][i] for p in retain_parts]) for i in range(2)) for _l in train_layers}
        num_tokens = {"forget": len(forget_tokens[train_layers[0]][0]), "retain": len(retain_tokens[train_layers[0]][0])}
        logger.info(f"tokens: {num_tokens}, retain samples per split: {n_retain}")

    # Train every layer on the recorded data (independent of each other), then write the weights
    with step_logging(logger, "#3", "W_down training"):
        device = wd.get("device", None) or model.device
        per_layer, new_weights = train_w_down_layers(
            model, train_layers, forget_tokens, retain_tokens, wd, seed, solver=solver, device=device
        )
        for _l, w in new_weights.items():
            apply_w_down(model, _l, w)
        torch.save(
            {
                "layers": train_layers, "coeff": coeff,
                "weights": {_l: w.to(get_out_proj(model, _l).weight.dtype) for _l, w in new_weights.items()}
            },
            run_dir / "w_down.pt"
        )

    # What did the edit do? (a) forget activations really moved by ~ coeff * r_UV? (b) responses after
    with step_logging(logger, "#4", "effect of the edit"):
        new_mean = ActivationCollector(
            model,
            positions=positions,
            layers=train_layers,
            point=saved["point"]
        ).mean(forget_loader, desc="forget after the edit")
        achieved_vs_intended = {}
        for j, _l in enumerate(train_layers):
            achieved = (new_mean[:, j] - saved["forget_mean"][:, saved["layers"].index(_l)]).flatten()
            r = r_layer[_l]
            intended = (coeff * (r if shift_mode == "eoi" else r[positions.index(ref_position)].expand_as(r))).flatten()
            achieved_vs_intended[_l] = {
                "cos": torch.nn.functional.cosine_similarity(achieved.float(), intended.float(), dim=0).item(),
                "norm_ratio": (achieved.norm() / intended.norm()).item(),
            }

        responses_after = res_collector.multi_eval(
            splits=list(wd["responses_splits"]),
            desc_exp=r"after / %(split)s",
            records_name_exp=r"responses_after_%(split)s",
            summaries_name="responses_after_summary"
        )
        save_logs({
            "tag": tag, "layers": train_layers, "coeff": coeff, "shift_mode": shift_mode, "ref_position": ref_position,
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
        }, run_dir / "w_down_summary.json")  # type: ignore
        for _l, v in achieved_vs_intended.items():
            logger.info(f"layer {_l}: achieved vs intended shift: cos={v['cos']:.3f}, norm ratio={v['norm_ratio']:.3f}")

        if wd.get("save_model", False):
            model.save_pretrained(run_dir / "model")
            tokenizer.save_pretrained(run_dir / "model")


if __name__ == "__main__":
    main()
