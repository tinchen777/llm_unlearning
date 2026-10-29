# ==== OUTPUT FILES of this script ====================================================
# output_dir = ${paths.output_dir} = <root_dir>/saves/<mode>/<task_name>        (configs/paths/default.yaml)
#   root_dir  : "."
#   mode      : "LUNAR"      (set by `experiment=generate/LUNAR`; the "generate" of configs/generate.yaml is overridden)
#   task_name : the `task_name=...` override of the shell script  ->  scripts/LUNAR/_common.sh: lunar_tofu_<MODEL>_<FORGET_SPLIT>_<REF>
#                  (REF = basename of REF_DATASET: LUNAR_harmful | LUNAR_unverified)
#                  ->  saves/LUNAR/lunar_tofu_Llama-3.2-1B-Instruct_forget01_LUNAR_harmful/
# Hydra's run dir is the SAME folder (configs/hydra/default.yaml: run.dir), so Hydra also writes `.hydra/` (the resolved
# config) and one log file per run, `<script name>_<date>_<time>.log` (e.g. r_uv_2026-10-19_11-10-08.log), there.
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
# Files written by THIS script (inside <output_dir>/c<coeff>_<shift_mode>/):
#   layer_selection.json         per layer: s1, s2, s1-s2, response statistics; `selected_layer`, `ranking`
#   responses_steer_l<layer>.json   steered forget answers of every candidate layer
#
# Read by / reads:
#   reads `<output_dir>/r_uv.pt` (select.r_uv_path)
#   w_down.py (wdown.layers=auto, or wdown.layer_selection_path) and scripts/LUNAR/4_eval.sh read the selection file
# ==============================================================================
from __future__ import annotations
# from rich.traceback import install
# install(show_locals=False, width=100)
import sys
from pathlib import Path

# Run as `python src/METHOD/LUNAR/<script>.py`: sys.path[0] is THIS directory, not `src/` where the shared packages
# (`data`, `model`, `utils`, `evals`, ...) live, so `src/` is put on the path before they are imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import logging
import torch
from typing import TYPE_CHECKING

from data import SplitData
from model import get_model_and_tokenizer
from model.arch import get_decoder_layers
from model.intervene.steer import Steerer
from evals.response.collector import ResponseCollector
from utils.common import get_cuda_visible_devices, save_logs, set_seed
from utils.log import step_logging
from utils.config import hydra_main
from METHOD._common.editing import resolve_candidate_layers
from METHOD.LUNAR._utils.embedding import SentenceEmbedder, layer_score, rank_layers
from METHOD.LUNAR._utils.selection import config_dir, selection_path

if TYPE_CHECKING:
    from utils.config import TrackingConfig

logger = logging.getLogger("main(select_layer)")


@hydra_main(config_path="../../../configs", config_name="generate")
def main(cfg: TrackingConfig):
    """LUNAR step 2: layer selection (section 3.2, appendix E).

    For every candidate layer l the forget prompts are redirected at inference time (`Steerer`: block_out of layer l +=
    coeff * r_UV, exactly what a perfect W_down edit of that layer would do) and the steered responses are embedded. The
    layer that maximises s1 - s2 is selected: s1 = similarity to the desired responses (the model states that it cannot
    answer), s2 = similarity to responses unrelated to unlearning (harmfulness, danger, unethicality).

    Writes `c<coeff>_<shift_mode>/layer_selection.json` (s1, s2, score and the response statistics of every layer, and the
    ranking; read it before trusting the automatic choice) and `responses_steer_l<l>.json` (same folder).

    Args:
        cfg (TrackingConfig): `experiment=generate/LUNAR` (uses `data.forget`, `select`); needs r_uv.pt of r_uv.py.
    """
    logger.info(f"CUDA_VISIBLE_DEVICES: {get_cuda_visible_devices()}")
    seed = int(cfg["seed"])
    set_seed(seed)
    sel = cfg["select"]
    output_dir = Path(str(cfg["paths"]["output_dir"]))
    coeff, mode, ref_position = float(sel["coeff"]), str(sel["shift_mode"]), int(sel["ref_position"])

    # 1. Load r_UV
    with step_logging(logger, "<1/4>", "r_UV", mode="load", cfg=sel["r_uv_path"], ensure_cfg=True) as r_uv_path:
        saved = torch.load(str(r_uv_path), weights_only=False)
        if saved["point"] not in ("block_out", "mlp_out"):
            raise ValueError(f"r_UV was computed at `{saved['point']}`; steering needs the residual stream "
                             "(`block_out` / `mlp_out`).")
        positions = saved["positions"]
        candidates = resolve_candidate_layers(sel, saved["layers"])
        logger.info(f"coeff={coeff}, mode={mode}, positions={positions}, candidate layers={candidates}")

    model_cfg = cfg["model"]
    template_args = model_cfg["template_args"]
    # 2. Load model and tokenizer
    with step_logging(logger, "<2/4>", "model & tokenizer", mode="load", cfg=model_cfg):
        model, tokenizer = get_model_and_tokenizer(model_cfg)
        model.eval()
        n_layers = len(get_decoder_layers(model))
        if max(candidates) >= n_layers:
            raise ValueError(f"Candidate layers {candidates} vs a model with {n_layers} layers: wrong r_uv.pt?")
    # 3. Load data (left padded, prompt only)
    with step_logging(logger, "<3/4>", "data", mode="load", cfg=cfg["data"], ensure_cfg=True) as data_cfg:
        split_data = SplitData(
            data_cfg,
            tokenizer=tokenizer,
            template_args=template_args
        ).set_split_loader(
            sel["batch_size"],
            max_samples=sel.get("max_samples", None),
            seed=seed
        )
    # 4. Load sentence embeddings of the target responses
    with step_logging(logger, "<4/4>", "sentence embeddings of the target responses", mode="load", cfg=sel):
        embedder = SentenceEmbedder(str(sel["embedder"]), device=model.device)
        desired = embedder([str(x) for x in sel["desired_responses"]])
        unrelated = embedder([str(x) for x in sel["unrelated_responses"]])

    res_collector = ResponseCollector(
        model, split_data, tokenizer, cfg["generation_args"],
        output_dir=config_dir(output_dir, coeff, mode),
        text_metrics="ALL",
        hallucination_rouge=float(cfg.get("hallucination_rouge", 0.3))
    )

    # ==========
    with step_logging(logger, "#1", "steering per layer"):
        per_layer = {}
        for layer in candidates:
            r = saved["r_uv"][:, saved["layers"].index(layer), :]  # [n_pos, d]
            with Steerer(model, layer, positions, mode=mode, ref_position=ref_position, coeff=coeff) as steerer:
                def before_batch(batch):
                    steerer.set(r.unsqueeze(0).expand(batch["input_ids"].shape[0], -1, -1),
                                prompt_len=batch["input_ids"].shape[1])
                records, stats = res_collector.eval(
                    "forget",
                    desc=f"forget / steer layer {layer}",
                    before_batch=before_batch,
                    save_name=f"responses_steer_l{layer}"
                )
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
        path = selection_path(output_dir, coeff, mode)
        save_logs({
            "selected_layer": selected,
            "ranking": ranking,  # best first; LUNAR also edits the top layers
            "criterion": "s1 - s2",
            "coeff": coeff, "shift_mode": mode, "ref_position": ref_position, "positions": positions,
            "max_degenerate": float(sel["max_degenerate"]), "aggregate": str(sel["aggregate"]),
            "embedder": str(sel["embedder"]),
            "per_layer": {str(_l): v for _l, v in per_layer.items()},
        }, path)  # type: ignore
        logger.info(f"Selected layer {selected} (s1 - s2), ranking {ranking[:5]}; details in {path}")


if __name__ == "__main__":
    main()
