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
# Files written by THIS script (inside <output_dir>/k<K>_c<coeff>_<shift_mode>/):
#   layer_selection.json         per layer: rougeL_recall, answer_logprob, refusal / degenerate rates, unseen_gap;
#                                `selected_layer`, `ranking`
#   responses_steer_l<layer>.json   steered forget answers of every candidate layer
#
# Read by / reads:
#   reads `<output_dir>/r_uv.pt` (select.r_uv_path)
#   w_down.py (wdown.layers=auto, or wdown.layer_selection_path) and scripts/NEAR/4_eval.sh read the selection file
# ==============================================================================
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import logging
import torch
from typing import TYPE_CHECKING

from data import SplitData
from model import get_model_and_tokenizer
from model.intervene.steer import Steerer
from evals.response.collector import ResponseCollector
from utils.common import get_cuda_visible_devices, save_logs, set_seed
from utils.log import step_logging
from utils.config import hydra_main
from METHOD.NEAR._utils.neighbors import config_dir, forget_authors, selection_path
from METHOD._common.editing import resolve_candidate_layers

if TYPE_CHECKING:
    from utils.config import TrackingConfig

logger = logging.getLogger("main(select_layer)")

# below these gaps the unsteered forget behaviour already equals the unseen one for that key (nothing to close)
_MIN_GAP = {"rougeL_recall": 0.02, "answer_logprob": 0.05}


def unseen_gap(steered: dict, original: dict, unseen: dict, keys=("rougeL_recall", "answer_logprob")) -> float:
    """How far the STEERED forget behaviour still is from the UNSEEN behaviour. Per key, normalised by the gap of
    the unsteered model (1 = nothing changed, 0 = matches the unseen questions, > 1 = moved the wrong way or
    overshot by more than the original gap), averaged over the keys with a real original gap, plus the absolute
    gap of the refusal rate (we want the model to answer like for a stranger, not to refuse)."""
    gaps = [
        abs(steered[k] - unseen[k]) / abs(original[k] - unseen[k])
        for k in keys if abs(original[k] - unseen[k]) >= _MIN_GAP[k]
    ]
    if not gaps:  # e.g. the base model: forget questions are already "unseen"
        logger.warning("The unsteered forget behaviour already matches the unseen one; unseen_gap only uses raw gaps.")
        gaps = [abs(steered[k] - unseen[k]) for k in keys]
    return sum(gaps) / len(gaps) + abs(steered["refusal_rate"] - unseen["refusal_rate"])


@hydra_main(config_path="../../../configs", config_name="generate")
def main(cfg: TrackingConfig):
    """Step 2b: LUNAR-style layer selection, adapted to the neighbour target.

    LUNAR modifies the layer where activation redirection is most effective. Here, for every candidate layer l,
    the forget prompts are redirected at inference time (`Steerer`: block_out of layer l += coeff * r_UV[K][author],
    exactly what a perfect W_down edit of that layer would do) and the steered model is compared with the
    UNSTEERED model on unseen questions (neighbor / holdout):

        rougeL_recall to the forget ground truth, ground-truth answer log-prob, refusal rate, degeneration rate.

    The selected layer minimises `unseen_gap` (forget answers look like answers about strangers) among the layers
    whose degeneration rate stays below `select.max_degenerate`. Writes `k<K>_c<coeff>_<shift_mode>/layer_selection.json` (all metrics
    per layer and the ranking, read it before trusting the automatic choice) and `responses_steer_l<l>.json` (same folder).

    Candidates: `select.layers` (explicit), else `select.layer_fraction` (window of relative depths), else all layers.

    Args:
        cfg (TrackingConfig): `experiment=generate/NEAR` (uses `data.*`, `select`); needs r_uv.pt of src/METHOD/NEAR/r_uv.py.
    """
    logger.info(f"CUDA_VISIBLE_DEVICES: {get_cuda_visible_devices()}")
    seed = int(cfg["seed"])
    set_seed(seed)
    sel = cfg["select"]
    output_dir = Path(str(cfg["paths"]["output_dir"]))
    k, coeff = int(sel["k"]), float(sel["coeff"])
    mode, ref_position = str(sel["shift_mode"]), int(sel["ref_position"])

    # 1. Load r_UV of the K-neighbour reference
    with step_logging(logger, "<1/3>", "r_UV", mode="load", cfg=sel["r_uv_path"], ensure_cfg=True) as r_uv_path:
        saved = torch.load(str(r_uv_path), weights_only=False)
        if k not in saved["ks"]:
            raise ValueError(f"select.k={k} was not computed in {r_uv_path} (ks={saved['ks']}).")
        if saved["point"] not in ("block_out", "mlp_out"):
            raise ValueError(
                f"r_UV was computed at `{saved['point']}`; "
                "steering needs the residual stream (`block_out` / `mlp_out`)."
            )
        positions, qa_per_author, author_offset = saved["positions"], saved["qa_per_author"], saved["author_offset"]
        candidates = resolve_candidate_layers(sel, saved["layers"])
        logger.info(f"K={k}, coeff={coeff}, mode={mode}, positions={positions}, candidate layers={candidates}")
    model_cfg = cfg["model"]
    template_args = model_cfg["template_args"]
    # 2. Load model and tokenizer
    with step_logging(logger, "<2/3>", "model & tokenizer", mode="load", cfg=model_cfg):
        model, tokenizer = get_model_and_tokenizer(model_cfg)
        model.eval()
        n_layers = model.model.config.num_hidden_layers
        if max(candidates) >= n_layers:
            raise ValueError(f"Candidate layers {candidates} vs a model with {n_layers} layers: wrong r_uv.pt?")
    # 3. load Data
    with step_logging(logger, "<3/3>", "data", mode="load", cfg=cfg["data"], ensure_cfg=True) as data_cfg:
        split_data = SplitData(
            data_cfg,
            tokenizer=tokenizer,
            template_args=template_args
        ).set_split_loader(
            sel["batch_size"],
            max_samples=sel.get("max_samples", None),
            seed=seed
        )

    res_collector = ResponseCollector(
        model, split_data, tokenizer, cfg["generation_args"],
        output_dir=config_dir(output_dir, k, coeff, mode),
        text_metrics="ALL",
        hallucination_rouge=float((cfg["probe"].get("responses", None) or {}).get("hallucination_rouge", 0.3))
    )

    # Reference behaviour of the unsteered model: forget (seen) vs the unseen splits
    with step_logging(logger, "#1", "get unsteered reference", cfg=sel["unseen_splits"], ensure_cfg=True) as unseen_splits:
        _, original = res_collector.eval("forget", desc="forget / unsteered")
        unseen_parts = {
            s: res_collector.eval(s, desc=f"{s} / unsteered")[1]
            for s in unseen_splits
        }
        unseen = {
            key: sum(p[key] for p in unseen_parts.values()) / len(unseen_parts)
            for key in ("rougeL_recall", "answer_logprob", "refusal_rate", "degenerate_rate", "hallucination_rate")
        }
        logger.info(f"forget (unsteered): {original}\nunseen (unsteered): {unseen}")

    # Steer every candidate layer
    with step_logging(logger, "#2", "steering per layer"):
        per_layer = {}
        for layer in candidates:
            li = saved["layers"].index(layer)
            r_author = {a: r[:, li, :] for a, r in saved["r_uv"][k].items()}  # author -> [n_pos, d]
            with Steerer(model, layer, positions, mode=mode, ref_position=ref_position, coeff=coeff) as steerer:
                def before_batch(batch):
                    authors = forget_authors(batch["index"], qa_per_author, author_offset)
                    steerer.set(torch.stack([r_author[a] for a in authors]), prompt_len=batch["input_ids"].shape[1])
                _, steered = res_collector.eval(
                    "forget",
                    desc=f"forget / steer layer {layer}",
                    before_batch=before_batch,
                    save_name=f"responses_steer_l{layer}"
                )
            steered["unseen_gap"] = unseen_gap(steered, original, unseen)
            per_layer[layer] = steered
            logger.info(
                f"layer {layer:>2}: unseen_gap={steered['unseen_gap']:.3f}  rougeL_recall={steered['rougeL_recall']:.3f}  "
                f"answer_logprob={steered['answer_logprob']:.3f}  refusal={steered['refusal_rate']:.2f}  "
                f"degenerate={steered['degenerate_rate']:.2f}"
            )

        criterion = str(sel["criterion"])
        valid = [_l for _l in candidates if per_layer[_l]["degenerate_rate"] <= float(sel["max_degenerate"])]
        if not valid:
            logger.warning("Every layer exceeds select.max_degenerate; selecting among all layers.")
            valid = candidates
        if criterion not in ("unseen_gap", "rougeL_recall", "answer_logprob"):
            raise ValueError(f"select.criterion must be unseen_gap | rougeL_recall | answer_logprob, got `{criterion}`.")
        # unseen_gap: look like a stranger; rougeL_recall / answer_logprob (lower is better): largest knowledge removal
        ranking = sorted(valid, key=lambda _l: per_layer[_l][criterion])
        selected = ranking[0]
        save_logs({
            "selected_layer": selected,
            "ranking": ranking,  # best first, among the non-degenerate layers; LUNAR also edits the top-K (K=3) layers
            "criterion": criterion,
            "k": k, "coeff": coeff, "shift_mode": mode, "ref_position": ref_position, "positions": positions,
            "max_degenerate": float(sel["max_degenerate"]),
            "forget_unsteered": original,
            "unseen_unsteered": {"mean": unseen, **unseen_parts},
            "per_layer": {str(_l): v for _l, v in per_layer.items()},
        }, selection_path(output_dir, k, coeff, mode))  # type: ignore
        logger.info(
            f"Selected layer {selected} ({criterion}), ranking {ranking[:5]}; "
            f"details in {selection_path(output_dir, k, coeff, mode)}"
        )


if __name__ == "__main__":
    main()
