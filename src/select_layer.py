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
from model.redirect import Steerer, forget_authors
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
    whose degeneration rate stays below `select.max_degenerate`. Writes layer_selection.json (all metrics per layer,
    read it before trusting the automatic choice) and responses_steer_l<l>.json.

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
            save_logs(records, output_dir / f"responses_steer_l{layer}.json")  # type: ignore
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
        if criterion == "unseen_gap":
            selected = min(valid, key=lambda l: per_layer[l]["unseen_gap"])
        elif criterion in ("rougeL_recall", "answer_logprob"):  # LUNAR-like: largest knowledge removal
            selected = min(valid, key=lambda l: per_layer[l][criterion])
        else:
            raise ValueError(f"select.criterion must be unseen_gap | rougeL_recall | answer_logprob, got `{criterion}`.")
        save_logs({
            "selected_layer": selected,
            "criterion": criterion,
            "k": k, "coeff": coeff, "shift_mode": mode, "ref_position": ref_position, "positions": positions,
            "max_degenerate": float(sel["max_degenerate"]),
            "forget_unsteered": original,
            "unseen_unsteered": {"mean": unseen, **unseen_parts},
            "per_layer": {str(l): v for l, v in per_layer.items()},
        }, output_dir / "layer_selection.json")  # type: ignore
        logger.info(f"Selected layer {selected} ({criterion}); details in {output_dir / 'layer_selection.json'}")


if __name__ == "__main__":
    main()
