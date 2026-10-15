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
import logging
import torch
from omegaconf import DictConfig

from data import get_split_data, get_subloader
from data.utils import one_dataset
from model import get_model_and_tokenizer
from model.activations import get_decoder_layers
from model.steer import Steerer
from utils.common import get_cuda_visible_devices, save_logs, set_seed
from utils.log import step_logging
from utils.config import TrackingConfig, init_hydra_choices
from METHOD._common.editing import resolve_candidate_layers
from METHOD._common.responses import generate_responses, summarize_responses
from METHOD.LUNAR._utils.embedding import SentenceEmbedder, layer_score, rank_layers
from METHOD.LUNAR._utils.selection import selection_file_name

logger = logging.getLogger("main(select_layer)")


@hydra.main(version_base=None, config_path="../../../configs", config_name="generate")
def main(config: DictConfig):
    """LUNAR step 2: layer selection (section 3.2, appendix E).

    For every candidate layer l the forget prompts are redirected at inference time (`Steerer`: block_out of layer l +=
    coeff * r_UV, exactly what a perfect W_down edit of that layer would do) and the steered responses are embedded. The
    layer that maximises s1 - s2 is selected: s1 = similarity to the desired responses (the model states that it cannot
    answer), s2 = similarity to responses unrelated to unlearning (harmfulness, danger, unethicality).

    Writes `layer_selection_c<coeff>_<shift_mode>.json` (s1, s2, score and the response statistics of every layer, and the
    ranking; read it before trusting the automatic choice) and `responses_steer_c<coeff>_<mode>_l<l>.json`.

    Args:
        config (DictConfig): `experiment=generate/LUNAR` (uses `data.forget`, `select`); needs r_uv.pt of r_uv.py.
    """
    logger.info(f"CUDA_VISIBLE_DEVICES: {get_cuda_visible_devices()}")
    init_hydra_choices(HydraConfig.get().runtime.choices)
    cfg = TrackingConfig(config)
    seed = int(cfg["seed"])
    set_seed(seed)
    sel = cfg["select"]
    output_dir = Path(str(cfg["paths"]["output_dir"]))
    coeff, mode, ref_position = float(sel["coeff"]), str(sel["shift_mode"]), int(sel["ref_position"])
    hallucination_rouge = float(cfg.get("hallucination_rouge", 0.3))

    with step_logging(logger, "[1/4]", "r_UV", sel):
        saved = torch.load(str(sel["r_uv_path"]), weights_only=False)
        if saved["point"] not in ("block_out", "mlp_out"):
            raise ValueError(f"r_UV was computed at `{saved['point']}`; steering needs the residual stream "
                             "(`block_out` / `mlp_out`).")
        positions = saved["positions"]
        candidates = resolve_candidate_layers(sel, saved["layers"])
        logger.info(f"coeff={coeff}, mode={mode}, positions={positions}, candidate layers={candidates}")

    model_cfg = cfg["model"]
    template_args = model_cfg["template_args"]
    with step_logging(logger, "[2/4]", "model & tokenizer & data", model_cfg):
        model, tokenizer = get_model_and_tokenizer(model_cfg)
        model.eval()
        n_layers = len(get_decoder_layers(model))
        if max(candidates) >= n_layers:
            raise ValueError(f"Candidate layers {candidates} vs a model with {n_layers} layers: wrong r_uv.pt?")
        split_data = get_split_data(cfg["data"], tokenizer=tokenizer, template_args=template_args)
        datasets, collator = split_data["forget"]
        forget_loader = get_subloader(one_dataset(datasets), sel["batch_size"], collator=collator,
                                      max_samples=sel.get("max_samples", None), seed=seed)

    with step_logging(logger, "[3/4]", "sentence embeddings of the target responses", sel):
        embedder = SentenceEmbedder(str(sel["embedder"]), device=model.device)
        desired = embedder([str(x) for x in sel["desired_responses"]])
        unrelated = embedder([str(x) for x in sel["unrelated_responses"]])

    with step_logging(logger, "[4/4]", "steering per layer", sel):
        per_layer = {}
        for layer in candidates:
            r = saved["r_uv"][:, saved["layers"].index(layer), :]  # [n_pos, d]
            with Steerer(model, layer, positions, mode=mode, ref_position=ref_position, coeff=coeff) as steerer:
                def before_batch(batch):
                    steerer.set(r.unsqueeze(0).expand(batch["input_ids"].shape[0], -1, -1),
                                prompt_len=batch["input_ids"].shape[1])
                records = generate_responses(model, tokenizer, forget_loader, cfg["generation_args"],
                                             desc=f"forget / steer layer {layer}", before_batch=before_batch)
            save_logs(records, output_dir / f"responses_steer_c{coeff:g}_{mode}_l{layer}.json")  # type: ignore
            stats = summarize_responses(records, hallucination_rouge)
            scores = layer_score(embedder([r_["generation"] for r_ in records]), desired, unrelated,
                                 str(sel["aggregate"]))
            per_layer[layer] = {**scores, **stats}
            logger.info(
                f"layer {layer:>2}: s1-s2={scores['score']:.3f} (s1={scores['s1']:.3f}, s2={scores['s2']:.3f})  "
                f"rougeL_recall={stats['rougeL_recall']:.3f}  refusal={stats['refusal_rate']:.2f}  "
                f"degenerate={stats['degenerate_rate']:.2f}"
            )

        ranking = rank_layers(per_layer, float(sel["max_degenerate"]))
        selected = ranking[0]
        path = output_dir / selection_file_name(coeff, mode)
        save_logs({
            "selected_layer": selected,
            "ranking": ranking,  # best first; LUNAR also edits the top layers
            "criterion": "s1 - s2",
            "coeff": coeff, "shift_mode": mode, "ref_position": ref_position, "positions": positions,
            "max_degenerate": float(sel["max_degenerate"]), "aggregate": str(sel["aggregate"]),
            "embedder": str(sel["embedder"]),
            "per_layer": {str(l): v for l, v in per_layer.items()},
        }, path)  # type: ignore
        logger.info(f"Selected layer {selected} (s1 - s2), ranking {ranking[:5]}; details in {path}")


if __name__ == "__main__":
    main()
