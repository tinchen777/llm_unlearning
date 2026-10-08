from __future__ import annotations
# from rich.traceback import install
# install(show_locals=False, width=100)
import hydra
from hydra.core.hydra_config import HydraConfig
import logging
import torch
from pathlib import Path
from omegaconf import DictConfig

from data import get_split_data, split_loader
from model import get_model_and_tokenizer
from model.activations import get_decoder_layers
from model.generation import generate_responses, summarize_responses
from model.redirect import Steerer, forget_authors, selection_file_name
from utils.common import get_cuda_visible_devices, save_logs, set_seed
from utils.log import step_logging
from utils.config import TrackingConfig, init_hydra_choices

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


@hydra.main(version_base=None, config_path="../configs", config_name="generate")
def main(config: DictConfig):
    """Step 2b: LUNAR-style layer selection, adapted to the neighbour target.

    LUNAR modifies the layer where activation redirection is most effective. Here, for every candidate layer l,
    the forget prompts are redirected at inference time (`Steerer`: block_out of layer l += coeff * r_UV[K][author],
    exactly what a perfect W_down edit of that layer would do) and the steered model is compared with the
    UNSTEERED model on unseen questions (neighbor / holdout):

        rougeL_recall to the forget ground truth, ground-truth answer log-prob, refusal rate, degeneration rate.

    The selected layer minimises `unseen_gap` (forget answers look like answers about strangers) among the layers
    whose degeneration rate stays below `select.max_degenerate`. Writes `layer_selection_k<K>_c<coeff>_<shift_mode>.json` (all metrics
    per layer and the ranking, read it before trusting the automatic choice) and responses_steer_*_l<l>.json.

    Candidates: `select.layers` (explicit), else `select.layer_fraction` (window of relative depths), else all layers.

    Args:
        config (DictConfig): `experiment=generate/neighbor_probe` (uses `data.*`, `select`); needs r_uv.pt of src/r_uv.py.
    """
    logger.info(f"CUDA_VISIBLE_DEVICES: {get_cuda_visible_devices()}")
    init_hydra_choices(HydraConfig.get().runtime.choices)
    cfg = TrackingConfig(config)
    seed = int(cfg["seed"])
    set_seed(seed)
    sel = cfg["select"]
    output_dir = Path(str(cfg["paths"]["output_dir"]))
    k, coeff = int(sel["k"]), float(sel["coeff"])
    mode, ref_position = str(sel["shift_mode"]), int(sel["ref_position"])
    refusal_patterns = list(cfg["probe"]["refusal_patterns"])
    hallucination_rouge = float(cfg["probe"]["hallucination_rouge"])

    # 1. r_UV of the K-neighbour reference
    with step_logging(logger, "[1/4]", "r_UV", sel):
        saved = torch.load(str(sel["r_uv_path"]), weights_only=False)
        if k not in saved["ks"]:
            raise ValueError(f"select.k={k} was not computed in {sel['r_uv_path']} (ks={saved['ks']}).")
        if saved["point"] not in ("block_out", "mlp_out"):
            raise ValueError(f"r_UV was computed at `{saved['point']}`; steering needs the residual stream "
                             "(`block_out` / `mlp_out`).")
        positions, qa_per_author, author_offset = saved["positions"], saved["qa_per_author"], saved["author_offset"]
        candidates = sel.get("layers", None)
        if candidates is None and sel.get("layer_fraction", None) is not None:
            # a window of relative depths, e.g. [0.4, 0.8]: LUNAR picks layers at 50-75 % of the depth (appendix E)
            lo, hi = (float(x) for x in sel["layer_fraction"])
            n_total = max(saved["layers"]) + 1
            candidates = [l for l in saved["layers"] if lo * n_total <= l <= hi * n_total]
            if not candidates:
                raise ValueError(f"select.layer_fraction={[lo, hi]} selects no layer of {saved['layers']}.")
        candidates = list(saved["layers"]) if candidates is None else [int(l) for l in candidates]
        if not set(candidates) <= set(saved["layers"]):
            raise ValueError(f"select.layers={candidates} not in the layers of r_uv.pt: {saved['layers']}.")
        logger.info(f"K={k}, coeff={coeff}, mode={mode}, positions={positions}, candidate layers={candidates}")

    model_cfg = cfg["model"]
    template_args = model_cfg["template_args"]
    # 2. Model and data
    with step_logging(logger, "[2/4]", "model & tokenizer & data", model_cfg):
        model, tokenizer = get_model_and_tokenizer(model_cfg)
        model.eval()
        n_layers = len(get_decoder_layers(model))
        if max(candidates) >= n_layers:
            raise ValueError(f"Candidate layers {candidates} vs a model with {n_layers} layers: wrong r_uv.pt?")
        split_data = get_split_data(cfg["data"], tokenizer=tokenizer, template_args=template_args)

    def loader(split):
        return split_loader(split_data, split, sel["batch_size"], sel.get("max_samples", None), seed)

    def run(split, desc, before_batch=None):
        records = generate_responses(model, tokenizer, loader(split), cfg["generation_args"], desc=desc,
                                     refusal_patterns=refusal_patterns, before_batch=before_batch)
        return records, summarize_responses(records, hallucination_rouge)

    # 3. Reference behaviour of the unsteered model: forget (seen) vs the unseen splits
    with step_logging(logger, "[3/4]", "unsteered reference", sel):
        _, original = run("forget", "forget / unsteered")
        unseen_parts = {s: run(s, f"{s} / unsteered")[1] for s in sel["unseen_splits"]}
        unseen = {
            key: sum(p[key] for p in unseen_parts.values()) / len(unseen_parts)
            for key in ("rougeL_recall", "answer_logprob", "refusal_rate", "degenerate_rate", "hallucination_rate")
        }
        logger.info(f"forget (unsteered): {original}\nunseen (unsteered): {unseen}")

    # 4. Steer every candidate layer
    with step_logging(logger, "[4/4]", "steering per layer", sel):
        per_layer = {}
        for layer in candidates:
            li = saved["layers"].index(layer)
            r_author = {a: r[:, li, :] for a, r in saved["r_uv"][k].items()}  # author -> [n_pos, d]
            with Steerer(model, layer, positions, mode=mode, ref_position=ref_position, coeff=coeff) as steerer:
                def before_batch(batch):
                    authors = forget_authors(batch["index"], qa_per_author, author_offset)
                    steerer.set(torch.stack([r_author[a] for a in authors]), prompt_len=batch["input_ids"].shape[1])
                records, steered = run("forget", f"forget / steer layer {layer}", before_batch)
            save_logs(records, output_dir / f"responses_steer_k{k}_c{coeff:g}_{mode}_l{layer}.json")  # type: ignore
            steered["unseen_gap"] = unseen_gap(steered, original, unseen)
            per_layer[layer] = steered
            logger.info(
                f"layer {layer:>2}: unseen_gap={steered['unseen_gap']:.3f}  rougeL_recall={steered['rougeL_recall']:.3f}  "
                f"answer_logprob={steered['answer_logprob']:.3f}  refusal={steered['refusal_rate']:.2f}  "
                f"degenerate={steered['degenerate_rate']:.2f}"
            )

        criterion = str(sel["criterion"])
        valid = [l for l in candidates if per_layer[l]["degenerate_rate"] <= float(sel["max_degenerate"])]
        if not valid:
            logger.warning("Every layer exceeds select.max_degenerate; selecting among all layers.")
            valid = candidates
        if criterion not in ("unseen_gap", "rougeL_recall", "answer_logprob"):
            raise ValueError(f"select.criterion must be unseen_gap | rougeL_recall | answer_logprob, got `{criterion}`.")
        # unseen_gap: look like a stranger; rougeL_recall / answer_logprob (lower is better): largest knowledge removal
        ranking = sorted(valid, key=lambda l: per_layer[l][criterion])
        selected = ranking[0]
        save_logs({
            "selected_layer": selected,
            "ranking": ranking,  # best first, among the non-degenerate layers; LUNAR also edits the top-K (K=3) layers
            "criterion": criterion,
            "k": k, "coeff": coeff, "shift_mode": mode, "ref_position": ref_position, "positions": positions,
            "max_degenerate": float(sel["max_degenerate"]),
            "forget_unsteered": original,
            "unseen_unsteered": {"mean": unseen, **unseen_parts},
            "per_layer": {str(l): v for l, v in per_layer.items()},
        }, output_dir / selection_file_name(k, coeff, mode))  # type: ignore
        logger.info(f"Selected layer {selected} ({criterion}), ranking {ranking[:5]}; "
                    f"details in {output_dir / selection_file_name(k, coeff, mode)}")


if __name__ == "__main__":
    main()
