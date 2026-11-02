# ==== OUTPUT FILES of this script ====================================================
# output_dir = ${paths.output_dir} = <root_dir>/saves/<mode>/<task_name>        (configs/paths/default.yaml)
#   root_dir  : "."
#   mode      : "NEAR"      (set by `experiment=unlearn/tofu/NEAR`; the "generate" of configs/generate.yaml is overridden)
#   task_name : the `task_name=...` override of the shell script  ->  scripts/NEAR/0_probe.sh : 0_probe/<MODEL>                   (retain model;  MODEL = tofu_<BASE>_<retain split>)
#                  scripts/NEAR/0_probe.sh : 0_probe/<MODEL>/<FORGET_SPLIT>    (full model;   MODEL = tofu_<BASE>_full)
#                  scripts/NEAR/4_eval.sh  : probe/probe_near_tofu_<MODEL>_<FORGET_SPLIT>_<TAG>   (edited model)
#                  ->  saves/NEAR/0_probe/tofu_Llama-3.2-1B-Instruct_retain90/  or  saves/NEAR/probe/probe_near_tofu_...
# To move the outputs: pass `task_name=...` (or edit TASK_NAME in scripts/NEAR/_common.sh), change `mode` in
# configs/experiment/unlearn/tofu/NEAR.yaml, or override `paths.output_dir=...` (configs/paths/default.yaml).
#
# Files written below output_dir:
#   responses/responses_<split>.json   generations + ROUGE + answer_logprob + refusal/degenerate flags, one file per split
#                                       (probe.responses; splits = probe.splits, default: every split defined under `data:`)
#   responses/responses_summary.json   per split: rougeL_recall, answer_prob, refusal / hallucination / degenerate rates
#   activations.pt                     per-split mean activations, r_UV (reference - forget), positions, layers   (probe.activations)
#   activations_summary.json           per layer: norms, r_UV size, cosines, AUCs of pairs of question sets
#
# Read by / reads:
#   neighbor_compare.py takes two of these `activations_summary.json` (full model / retain model) via --full / --retain
#
# NOTE: the responses live in the sub-folder `responses/` (variable `responses_dir`), the activation files directly in output_dir.
# ==============================================================================
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import logging
import torch
from typing import TYPE_CHECKING

from data import SplitData
from data.datasets.datasets_utils import get_eoi_positions
from model import get_model_and_tokenizer
from model.hooks.collector import ActivationCollector
from evals.response.collector import ResponseCollector
from utils.common import get_cuda_visible_devices, save_logs, set_seed
from utils.log import step_logging
from utils.config import hydra_main
from METHOD._common.lineage import check_r_uv_identity
from evals.activation.analysis import analyze_activations, save_activation_outputs, sample_group  # noqa: F401  (sample_group: tests)

if TYPE_CHECKING:
    from utils.config import TrackingConfig

logger = logging.getLogger("main(NEAR-probe)")

@hydra_main(config_path="../../../configs", config_name="generate")
def main(cfg: TrackingConfig):
    """Step 1 (+ step 2 diagnostics) of the neighbour experiment: probe ONE model (no Trainer) on the
    forget / neighbor / retain / holdout question sets.

    1. (`probe.responses`) responses + statistics per split (ROUGE to the ground truth, ground-truth answer
       log-prob, refusal and hallucination rates) -> responses_<split>.json, responses_summary.json.
       Run it on the base / TOFU-full / TOFU-retain model (`model=tofu/tofu_<base>_<full|retainXX>`) to test the
       hypothesis "a model answers questions about entities it has never seen with hallucinations".
    2. (`probe.activations`) per-layer activations at the end-of-instruction tokens: dataset means, the global r_UV
       (reference - forget, as LUNAR) and linear separability of pairs of question sets at every layer
       -> activations.pt, activations_summary.json.

    This step is an ANALYSIS of one model. The r_UV / layer selection / W_down chain (src/METHOD/NEAR/r_uv.py,
    src/METHOD/NEAR/select_layer.py, src/METHOD/NEAR/w_down.py) does not read its outputs.

    Reading the AUCs (see the `note` of activations_summary.json): they measure how well two question sets can be
    told apart, which is dominated by WHAT the questions are about (other authors, other templates, other length),
    not by whether the model knows them. Only compare pairs whose questions are matched (forget | neighbour), and
    compare the SAME pair across models (full - retain, `src/METHOD/NEAR/neighbor_compare.py`).

    Args:
        cfg (TrackingConfig): `experiment=unlearn/tofu/NEAR`
    """
    # cuda device check
    logger.info(f"CUDA_VISIBLE_DEVICES: {get_cuda_visible_devices()}")
    # Set seed for reproducibility
    seed = cfg["seed"]
    set_seed(seed)

    model_cfg = cfg["model"]
    template_args = model_cfg["template_args"]
    # 1. Load model and tokenizer
    with step_logging(logger, "<1/2>", "model & tokenizer", mode="load", cfg=model_cfg):
        model, tokenizer = get_model_and_tokenizer(model_cfg)
        model.eval()
    # 2. Load Data (left padded, prompt only)
    with step_logging(logger, "<2/2>", "data", mode="load", cfg=cfg["data"], ensure_cfg=True) as data_cfg:
        split_data = SplitData(
            data_cfg,
            tokenizer=tokenizer,
            template_args=template_args
        )

    # ==========
    output_dir = Path(str(cfg["paths"]["output_dir"]))
    probe_cfg = cfg["probe"]
    # Model responses on a FIXED random subset of each split (the same subset for every model)
    with step_logging(logger, "#1", "calculate responses", cfg=probe_cfg.get("responses", None)) as res_cfg:
        if res_cfg:
            responses_dir = output_dir / "responses"
            # set dataloader config
            split_data.set_split_loader(
                batch_size=probe_cfg["batch_size"],
                max_samples=res_cfg.get("max_gen_samples", None),
                sampling_seed=seed,
                shuffle=False,
                num_workers=0
            )
            res_collector = ResponseCollector(
                model, tokenizer, cfg["generation_args"],
                output_dir=responses_dir,
                text_metrics="ALL",
                hallucination_rouge=float(res_cfg["hallucination_rouge"]),
            ).set_split_data(split_data)
            summaries = res_collector.multi_eval(
                desc_exp=r"%(split)s",
                peek=2,
                records_name_exp=r"responses_%(split)s",
            )
            save_logs(
                {"model": str(model_cfg["pretrained"]["name_or_path"]), "splits": summaries},
                file_path=responses_dir / "responses_summary.json"
            )

            for split, s in summaries.items():
                logger.info(
                    f"[{split:>8}] rougeL_recall={s['rougeL_recall']:.3f}  answer_prob={s.get('answer_prob', float('nan')):.3f}  "
                    f"refusal={s['refusal_rate']:.2f}  hallucination={s['hallucination_rate']:.2f}  degenerate={s['degenerate_rate']:.2f}"
                )

    # Per-layer activations: means, global r_UV, separability of pairs of question sets
    with step_logging(logger, "#2", "calculate activations", cfg=probe_cfg.get("activations", None)) as act_cfg:
        if act_cfg:
            forget, reference = act_cfg["forget"], act_cfg["reference"]
            # hook point / positions / layers / author numbering: of the config (analysis of an original model, before any
            # r_UV exists), or of an r_uv.pt (`from_r_uv`: the probe of an edited model, to be compared with that run)
            positions = act_cfg.get("positions", None)
            layers = act_cfg.get("layers", None)
            point = act_cfg.get("point", "block_out")
            qa_per_author, author_offset = int(act_cfg["qa_per_author"]), int(act_cfg["author_offset"])
            r_uv_path = act_cfg.get("from_r_uv", None)
            if r_uv_path is not None:
                ref = torch.load(str(r_uv_path), weights_only=False)
                check_r_uv_identity(
                    ref,
                    forget_split=cfg.get("forget_split", None),
                    num_forget=len(split_data[forget].dataset),
                    source=str(r_uv_path),
                )
                from_file = {
                    "positions": list(ref["positions"]), "layers": list(ref["layers"]), "point": ref["point"],
                    "qa_per_author": int(ref["qa_per_author"]), "author_offset": int(ref["author_offset"]),
                }
                given = {
                    "positions": None if positions is None else list(positions),
                    "layers": None if layers is None else list(layers),
                    "point": point, "qa_per_author": qa_per_author, "author_offset": author_offset,
                }
                for key, value in given.items():
                    if value is not None and value != from_file[key]:
                        logger.warning(f"probe.activations.{key}={value} is ignored: {r_uv_path} was computed with {from_file[key]}.")
                positions, layers, point = from_file["positions"], from_file["layers"], from_file["point"]
                qa_per_author, author_offset = from_file["qa_per_author"], from_file["author_offset"]
            act_collector = ActivationCollector(
                model,
                positions=list(positions) if positions is not None else get_eoi_positions(tokenizer, template_args),
                layers=layers,
                point=point
            )
            logger.info(f"Hook point: `{act_collector.point}`, positions: {act_collector.positions}")

            split_data.set_split_loader(
                probe_cfg["batch_size"],
                max_samples=act_cfg.get("max_act_samples", None),
                sampling_seed=seed
            )
            result = analyze_activations(
                act_collector, split_data, forget, reference,
                seen=act_cfg["seen"], unseen=act_cfg["unseen"],
                qa_per_author=qa_per_author, author_offset=author_offset, seed=seed,
            )
            save_activation_outputs(result, act_collector, output_dir)
            logger.info(f"Saved r_UV {tuple(result['r_uv'].shape)} (n_pos, n_layers, d) and per-layer diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
