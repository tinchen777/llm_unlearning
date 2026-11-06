# ==== OUTPUT FILES of this script ====================================================
# output_dir = ${paths.output_dir} = <root_dir>/saves/<mode>/<task_name>        (configs/paths/default.yaml)
#   root_dir  : "."
#   mode      : "NEAR"      (set by `experiment=unlearn/tofu/NEAR`; the "generate" of configs/generate.yaml is overridden)
#   task_name : the `task_name=...` override of the shell script  ->  scripts/NEAR/_common.sh: near_tofu_<MODEL>_<FORGET_SPLIT>
#                  (MODEL = base name, e.g. Llama-3.2-1B-Instruct)  ->  saves/NEAR/near_tofu_Llama-3.2-1B-Instruct_forget01/
# To move the outputs: pass `task_name=...` (or edit TASK_NAME in scripts/NEAR/_common.sh), change `mode` in
# configs/experiment/unlearn/tofu/NEAR.yaml, or override `paths.output_dir=...` (configs/paths/default.yaml).
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
import copy
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
from METHOD.NEAR._utils.neighbors import forget_authors, run_dir as make_run_dir
from METHOD.NEAR.kl_refine import build_kl_pairs, refine_w_down
from METHOD._common.editing import layer_increments, train_w_down_layers
from METHOD._common.lineage import check_r_uv_identity, model_name, pick, resolve_r_uv_path

if TYPE_CHECKING:
    from utils.config import TrackingConfig

logger = logging.getLogger("main(w_down)")


@hydra_main(config_path="../../../configs", config_name="generate")
def main(cfg: TrackingConfig):
    """Train W_down (the MLP output projection) of chosen layers so that forget prompts are redirected by
    r_UV of their author, while retain prompts keep their original MLP output (LUNAR style, no Trainer).

    Follows ONE `layer_selection.json` of src/METHOD/NEAR/select_layer.py (`wdown.layer_selection_path`): K, coeff, shift mode,
    reference position and the selected layer (`wdown.layers: auto`) are read from it, r_uv.pt through its `r_uv_path`; the
    forget split / model must be the ones r_uv.pt was computed on.

    `wdown.solver`: `adam` (AdamW + ExponentialLR, as LUNAR's code; needs lr / epochs) or `closed_form` (LUNAR Eq. 9,
    ridge regression of the weight change, no lr / epochs; `wdown.ridge` shrinks the edit towards the original weights).
    `wdown.shift_mode`: `eoi` (r_UV only on the end-of-instruction tokens) or `all` (the paper: the same vector on every
    token of a forget prompt). Outputs go to `k<K>_c<coeff>_<shift_mode>/l<layers>[_cf]/`. Args:
        cfg (TrackingConfig): `experiment=unlearn/tofu/NEAR` (uses `data.*`, `wdown`)
    """
    logger.info(f"CUDA_VISIBLE_DEVICES: {get_cuda_visible_devices()}")
    seed = int(cfg["seed"])
    set_seed(seed)
    wd_cfg = cfg["wdown"]
    output_dir = Path(str(cfg["paths"]["output_dir"]))
    solver = str(wd_cfg.get("solver", "adam"))
    if solver not in ("adam", "closed_form"):
        raise ValueError(f"wdown.solver must be `adam` or `closed_form`, got `{solver}`.")

    # 1. Upstream: the layer selection (k, coeff, shift_mode, ref_position, layer, r_uv.pt) and r_UV itself
    sel_path = wd_cfg.get("layer_selection_path", None)
    with step_logging(logger, "<1/3>", "layer selection & r_UV", mode="load"):
        selection = None
        if sel_path is not None:
            with open(str(sel_path), encoding="utf-8") as f:
                selection = json.load(f)
        src = str(sel_path) if sel_path is not None else "the config"
        up = (lambda key: selection.get(key)) if selection is not None else (lambda key: None)
        k = int(pick("k", wd_cfg.get("k", None), up("k"), src))
        coeff = float(pick("coeff", wd_cfg.get("coeff", None), up("coeff"), src))
        shift_mode = str(pick("shift_mode", wd_cfg.get("shift_mode", None), up("shift_mode"), src))
        ref_position = int(pick("ref_position", wd_cfg.get("ref_position", None), up("ref_position"), src))
        include_answer = bool(wd_cfg.get("include_answer", False))
        answer_coeff = wd_cfg.get("answer_coeff", None)  # null: the answer tokens get the same shift as the prompt; else the factor on `coeff` there
        answer_scale = 1.0 if answer_coeff is None else float(answer_coeff)
        if answer_scale != 1.0 and not include_answer:
            raise ValueError("wdown.answer_coeff needs wdown.include_answer=true.")
        if include_answer and shift_mode != "all":
            raise ValueError("wdown.include_answer needs shift_mode=all: the `eoi` positions are counted from the end of the prompt.")

        layers_cfg = wd_cfg.get("layers", "auto")
        if layers_cfg is None or layers_cfg == "auto":
            if selection is None:
                raise ValueError("wdown.layers=auto needs wdown.layer_selection_path (or give explicit wdown.layers).")
            train_layers = [int(selection["selected_layer"])]
            logger.info(f"wdown.layers=auto -> layer {train_layers} from {sel_path}")
        else:
            train_layers = [int(_l) for _l in layers_cfg]
        train_layers = sorted(set(train_layers))
        multi_layer = str(wd_cfg.get("multi_layer", "same"))
        if multi_layer not in ("same", "increment"):
            raise ValueError(f"wdown.multi_layer must be `same` or `increment`, got `{multi_layer}`.")
        sequential = bool(wd_cfg.get("sequential", False))
        kl_cfg = wd_cfg.get("kl", None) or {}
        use_kl = bool(kl_cfg.get("enabled", False))
        if use_kl and not include_answer:
            raise ValueError("wdown.kl.enabled needs wdown.include_answer=true (the refinement trains on prompt + answer tokens).")

        if selection is not None:
            r_uv_path = resolve_r_uv_path(selection.get("r_uv_path"), Path(str(sel_path)))
            cfg_r_uv = wd_cfg.get("r_uv_path", None)
            if cfg_r_uv is not None and Path(str(cfg_r_uv)).resolve() != r_uv_path.resolve():
                raise ValueError(f"wdown.r_uv_path={cfg_r_uv} disagrees with {r_uv_path} recorded in {sel_path}.")
        else:
            r_uv_path = Path(str(pick("r_uv_path", wd_cfg.get("r_uv_path", None), None, src)))
        saved = torch.load(str(r_uv_path), weights_only=False)
        if k not in saved["ks"]:
            raise ValueError(f"k={k} was not computed in {r_uv_path} (ks={saved['ks']}).")
        if saved["point"] not in ("block_out", "mlp_out"):
            raise ValueError(f"r_UV was computed at `{saved['point']}`; only `block_out` / `mlp_out` live in the "
                             "d_model space of the MLP output.")
        if not set(train_layers) <= set(saved["layers"]):
            raise ValueError(f"layers={train_layers} not in the layers of r_uv.pt: {saved['layers']}.")
        positions, qa_per_author, author_offset = saved["positions"], saved["qa_per_author"], saved["author_offset"]
        if shift_mode == "all" and ref_position not in positions:
            raise ValueError(f"ref_position={ref_position} is not one of the saved positions {positions}.")
        # r_full[layer][author] -> [n_pos, d]: the unlearning vector at the output of the layer
        r_full = {
            _l: {a: r[:, saved["layers"].index(_l), :] for a, r in saved["r_uv"][k].items()}
            for _l in train_layers
        }
        # r_layer: what each edited layer adds. `same`: its whole r_UV (one layer: the usual); `increment`: the share of this layer, so
        # that the sum over the edited layers is the r_UV of the last one (see `layer_increments`)
        r_layer = layer_increments(r_full) if multi_layer == "increment" else r_full
        logger.info(
            f"K={k}, layers={train_layers}, coeff={coeff}, shift_mode={shift_mode}, positions={positions}, "
            f"multi_layer={multi_layer}, sequential={sequential}"
        )

    tag = f"k{k}_l{'-'.join(str(l) for l in train_layers)}_c{coeff:g}_{shift_mode}" + ("_cf" if solver == "closed_form" else "")
    if len(train_layers) > 1:
        tag += f"_{multi_layer}" + ("_seq" if sequential else "")
    run_dir = make_run_dir(output_dir, k, coeff, shift_mode, train_layers, solver)  # <output_dir>/k<K>_c<coeff>_<mode>/l<layers>[_cf]/
    run_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = cfg["model"]
    template_args = model_cfg["template_args"]
    # 2. Load model and tokenizer
    with step_logging(logger, "<2/3>", "model & tokenizer", mode="load", cfg=model_cfg):
        model, tokenizer = get_model_and_tokenizer(model_cfg)
        model.eval()
        # the ORIGINAL model (frozen) = the target of the activations and of the KL of the refinement; taken before any edit
        ref_model = copy.deepcopy(model).eval() if use_kl else None
    # 3. Load data
    with step_logging(logger, "<3/3>", "data", mode="load", cfg=cfg["data"], ensure_cfg=True) as data_cfg:
        split_data = SplitData(
            data_cfg,
            tokenizer=tokenizer,
            template_args=template_args
        ).set_split_loader(
            wd_cfg["gen_batch_size"],
            max_samples=wd_cfg["max_gen_samples"],
            sampling_seed=seed
        )
        # the forget rows / model must be the ones r_uv.pt was computed on (author of a row = row index // qa_per_author)
        check_r_uv_identity(
            saved,
            forget_split=cfg.get("forget_split", None),
            num_forget=len(split_data["forget"].dataset),
            model=model_name(model_cfg),
            source=str(r_uv_path),
        )

    res_collector = ResponseCollector(
        model, tokenizer, cfg["generation_args"],
        output_dir=run_dir,
        text_metrics="ALL",
        hallucination_rouge=float(wd_cfg["hallucination_rouge"])
    ).set_split_data(split_data)

    # ==========
    # Responses of the ORIGINAL model (same fixed subset as after the edit)
    with step_logging(logger, "#1", "responses before the edit"):
        responses_before = res_collector.multi_eval(
            splits=list(wd_cfg["responses_splits"]),
            desc_exp=r"before / %(split)s",
            records_name_exp=r"responses_before_%(split)s",
            summaries_name="responses_before_summary"
        )

    # Record the original (x, y) of every token and build the targets
    with step_logging(logger, "#2", "MLP inputs / targets"):
        def shift_fn(batch, layer, attention_mask):
            authors = forget_authors(batch["index"], qa_per_author, author_offset)
            r = torch.stack([r_layer[layer][a] for a in authors])  # [bsz, n_pos, d]
            shift = coeff * build_shift(r, attention_mask, positions, shift_mode, ref_position)
            if answer_scale != 1.0:  # labels are aligned with input_ids and the prompt is masked (-100): label != -100 = answer token
                is_answer = (batch["labels"] != -100).to(shift.device).unsqueeze(-1)
                shift = torch.where(is_answer, answer_scale * shift, shift)
            return shift

        # all forget samples (no subset). `include_answer`: the tokens of prompt AND answer (the evaluation, MIA and the answer
        # probabilities, feeds the model the whole text), for the forget samples (shifted) and the retain ones (kept) alike
        forget_loader = split_data["forget"].loader(batch_size=wd_cfg["batch_size"], max_samples=None)  # prompts: the effect check below
        as_trained = (lambda data: data.with_generation_mode(False)) if include_answer else (lambda data: data)
        # fresh loaders of the SAME tokens in the SAME order (no shuffle, fixed subsets): `sequential` records the inputs again
        make_forget_loader = lambda: as_trained(split_data["forget"]).loader(batch_size=wd_cfg["batch_size"], max_samples=None)
        make_retain_loader = lambda s: as_trained(split_data[s]).loader(batch_size=wd_cfg["batch_size"], max_samples=wd_cfg["retain_max_samples"])
        forget_tokens = collect_mlp_tokens(model, make_forget_loader(), train_layers, shift_fn=shift_fn, desc="forget (+shift)")
        retain_parts = [
            collect_mlp_tokens(model, make_retain_loader(s), train_layers, desc=f"retain / {s}")
            for s in wd_cfg["retain_splits"]
        ]
        retain_tokens = {_l: tuple(torch.cat([p[_l][i] for p in retain_parts]) for i in range(2)) for _l in train_layers}
        num_tokens = {"forget": len(forget_tokens[train_layers[0]][0]), "retain": len(retain_tokens[train_layers[0]][0])}
        logger.info(f"tokens: {num_tokens}")

    # Train every layer on the recorded data, then write the weights. Default: the layers are independent (all tokens were recorded on
    # the ORIGINAL model). `sequential`: from the lowest layer up, after each layer the inputs of the next one are recorded again on the
    # model edited so far, with the SAME targets (original output (+ shift for forget)): a layer also repairs what the edits below did to its input
    with step_logging(logger, "#3", "W_down training"):
        if sequential:
            per_layer, new_weights = {}, {}
            for i, _l in enumerate(train_layers):
                if i > 0:
                    fx = collect_mlp_tokens(model, make_forget_loader(), [_l], desc=f"forget inputs, layer {_l}, edited below")[_l][0]
                    rx = torch.cat([
                        collect_mlp_tokens(model, make_retain_loader(s), [_l], desc=f"retain / {s} inputs, layer {_l}")[_l][0]
                        for s in wd_cfg["retain_splits"]
                    ])
                    if len(fx) != len(forget_tokens[_l][1]) or len(rx) != len(retain_tokens[_l][1]):
                        raise RuntimeError("The recorded tokens changed between the passes (the loaders must be deterministic).")
                    forget_tokens[_l], retain_tokens[_l] = (fx, forget_tokens[_l][1]), (rx, retain_tokens[_l][1])
                layer_info, layer_weight = train_w_down_layers(
                    model, [_l], {_l: forget_tokens[_l]}, {_l: retain_tokens[_l]}, wd_cfg, seed, solver=solver
                )
                per_layer.update(layer_info)
                new_weights.update(layer_weight)
                apply_w_down(model, _l, layer_weight[_l])
        else:
            per_layer, new_weights = train_w_down_layers(
                model, train_layers, forget_tokens, retain_tokens, wd_cfg, seed, solver=solver
            )
            for _l, w in new_weights.items():
                apply_w_down(model, _l, w)
        kl_history = None
        if use_kl:
            with step_logging(logger, "#3b", "end-to-end refinement with the KL term"):
                groups = build_kl_pairs(
                    split_data["forget"].dataset, split_data["neighbor"].with_generation_mode(False).dataset,
                    qa_per_author, author_offset, int(kl_cfg["n_tokens"]),
                )
                logger.info(f"KL pairs: {sum(len(v) for v in groups.values())} (neighbour answers) for {len(groups)} forget questions")
                retain_loaders = lambda: [b for s_ in wd_cfg["retain_splits"] for b in as_trained(split_data[s_]).loader(
                    batch_size=wd_cfg["batch_size"], max_samples=int(kl_cfg["retain_max_samples"]))]
                refined = refine_w_down(
                    model, ref_model, tokenizer, train_layers, new_weights, make_forget_loader, retain_loaders, shift_fn, groups,
                    kl_cfg, float(wd_cfg["forget_weight"]), float(wd_cfg["retain_weight"]), seed,
                )
                new_weights, kl_history = refined["weights"], refined["history"]
                for _l, w in new_weights.items():
                    apply_w_down(model, _l, w)
            del ref_model
            torch.cuda.empty_cache()
        torch.save(
            {
                "layers": train_layers, "k": k, "coeff": coeff,
                "weights": {_l: w.to(get_out_proj(model, _l).weight.dtype) for _l, w in new_weights.items()}
            },
            run_dir / "w_down.pt"
        )

    # What did the edit do? (a) forget activations really moved by ~ coeff * r_UV? (b) responses after
    with step_logging(logger, "#4", "effect of the edit"):
        new_means, _ = ActivationCollector(
            model,
            positions=positions,
            layers=train_layers,
            point=saved["point"]
        ).group_means(
            dataloader=forget_loader,
            group_fn=lambda b: forget_authors(b["index"], qa_per_author, author_offset),
            desc="forget after the edit"
        )
        achieved_vs_intended = {}
        for j, _l in enumerate(train_layers):
            li = saved["layers"].index(_l)
            cos, ratio = [], []
            for a in new_means:
                achieved = (new_means[a][:, j] - saved["forget_means"][a][:, li]).flatten()
                r = r_full[_l][a]  # the shift accumulated up to this layer: c * r_UV (exact for one layer / `increment`)
                intended = (coeff * (r if shift_mode == "eoi" else r[positions.index(ref_position)].expand_as(r))).flatten()
                cos.append(torch.nn.functional.cosine_similarity(achieved, intended, dim=0).item())
                ratio.append((achieved.norm() / intended.norm()).item())
            achieved_vs_intended[_l] = {"cos": sum(cos) / len(cos), "norm_ratio": sum(ratio) / len(ratio)}

        responses_after = res_collector.multi_eval(
                splits=list(wd_cfg["responses_splits"]),
                desc_exp=r"after / %(split)s",
                records_name_exp=r"responses_after_%(split)s",
                summaries_name="responses_after_summary"
            )
        save_logs({
            "layer_selection_path": str(sel_path) if sel_path is not None else None, "r_uv_path": str(r_uv_path),
            "tag": tag, "k": k, "layers": train_layers, "coeff": coeff, "shift_mode": shift_mode, "ref_position": ref_position,
            "solver": solver, "ridge": float(wd_cfg["ridge"]) if solver == "closed_form" else None,
            "include_answer": include_answer, "answer_coeff": answer_scale, "multi_layer": multi_layer, "sequential": sequential, "num_tokens": num_tokens,
            "kl": ({k_: kl_cfg[k_] for k_ in ("steps", "lr", "weight", "n_tokens", "batch_size", "retain_max_samples", "forget_prob_weight", "forget_prob_target") if k_ in kl_cfg} | {"history": kl_history}) if use_kl else None,
            "hyperparameters": {key: wd_cfg[key] for key in (
                "lr", "lr_gamma", "epochs", "batch_size", "weight_decay", "forget_weight", "retain_weight",
                "retain_max_samples")} | {"retain_splits": list(wd_cfg["retain_splits"])},
            "per_layer": per_layer,
            "achieved_vs_intended": achieved_vs_intended,
            "responses": {"before": responses_before, "after": responses_after},
            "note": "achieved_vs_intended: mean forget activation after vs before the edit, against coeff*r_UV "
                    "(per author, averaged). Exact for a single edited layer; with several layers the deeper ones "
                    "also contain the effect of the upstream edits (exact with multi_layer=increment: the shift accumulated up to a layer is coeff*r_UV of it).",
        }, run_dir / "w_down_summary.json")  # type: ignore
        for _l, v in achieved_vs_intended.items():
            logger.info(f"layer {_l}: achieved vs intended shift: cos={v['cos']:.3f}, norm ratio={v['norm_ratio']:.3f}")

        if wd_cfg.get("save_model", False):
            model.save_pretrained(run_dir / "model")
            tokenizer.save_pretrained(run_dir / "model")


if __name__ == "__main__":
    main()
