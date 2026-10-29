# ==== OUTPUT FILES of this script ====================================================
# output_dir = ${paths.output_dir} = <root_dir>/saves/<mode>/<task_name>        (configs/paths/default.yaml)
#   root_dir  : "."
#   mode      : "NEAR"      (set by `experiment=generate/NEAR`; the "generate" of configs/generate.yaml is overridden)
#   task_name : the `task_name=...` override of the shell script  ->  scripts/NEAR/0_probe.sh : 0_probe/<MODEL>                   (retain model;  MODEL = tofu_<BASE>_<retain split>)
#                  scripts/NEAR/0_probe.sh : 0_probe/<MODEL>/<FORGET_SPLIT>    (full model;   MODEL = tofu_<BASE>_full)
#                  scripts/NEAR/4_eval.sh  : probe/probe_near_tofu_<MODEL>_<FORGET_SPLIT>_<TAG>   (edited model)
#                  ->  saves/NEAR/0_probe/tofu_Llama-3.2-1B-Instruct_retain90/  or  saves/NEAR/probe/probe_near_tofu_...
# To move the outputs: pass `task_name=...` (or edit TASK_NAME in scripts/NEAR/_common.sh), change `mode` in
# configs/experiment/generate/NEAR.yaml, or override `paths.output_dir=...` (configs/paths/default.yaml).
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
import itertools
import logging
import torch
import torch.nn.functional as F
from typing import Any, Dict, Hashable, TYPE_CHECKING

from data import SplitData
from data.datasets.datasets_utils import get_eoi_positions
from model import get_model_and_tokenizer
from model.hooks.collector import ActivationCollector
from evals.response.collector import ResponseCollector
from utils.common import get_cuda_visible_devices, save_logs, set_seed
from utils.log import step_logging
from utils.config import hydra_main
from METHOD._common.activations import unlearning_vector
from METHOD.NEAR._utils.activations import mean_diff_auc

if TYPE_CHECKING:
    from utils.config import TrackingConfig

logger = logging.getLogger("main(NEAR-probe)")

_META_KEYS = ("author_id", "neighbor_id")


def sample_group(split: str, index: int, meta: Dict[str, Any], forget_split: str, qa_per_author: int, author_offset: int) -> Hashable:
    """Group (= author or neighbour) of a question, the unit of the group-wise cross-validation.

    - neighbour rows: the author they were generated for (`author_id`), so a forget author and his neighbours share
      a group (and thus a fold); the neighbour csv also has `neighbor_id`, the rows of one neighbour share
      `(author_id, neighbor_id)` but the author is what matters for the leakage;
    - every other split (TOFU stores the 20 QAs of an author contiguously): `index // qa_per_author`,
      shifted by `author_offset` for the forget split so that it matches the `author_id` of the neighbour csv."""
    if "author_id" in meta:
        return f"author:{meta['author_id']}"
    author = index // qa_per_author
    if split == forget_split:
        return f"author:{author + author_offset}"
    return f"{split}:{author}"


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
        cfg (TrackingConfig): `experiment=generate/NEAR`
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
                seed=seed,
                shuffle=False,
                num_workers=0
            )
            res_collector = ResponseCollector(
                model, split_data, tokenizer, cfg["generation_args"],
                output_dir=responses_dir,
                text_metrics="ALL",
                hallucination_rouge=float(res_cfg["hallucination_rouge"]),
            )
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
            # collector
            _positions = act_cfg.get("positions", None)
            act_collector = ActivationCollector(
                model,
                positions=list(_positions) if _positions is not None else get_eoi_positions(tokenizer, template_args),
                layers=act_cfg.get("layers", None),
                point=act_cfg.get("point", "block_out")
            )
            logger.info(f"Hook point: `{act_collector.point}`, positions: {act_collector.positions}")

            qa_per_author = int(act_cfg["qa_per_author"])
            author_offset = int(act_cfg["author_offset"])
            forget, reference = act_cfg["forget"], act_cfg["reference"]

            split_data.set_split_loader(
                probe_cfg["batch_size"],
                max_samples=act_cfg.get("max_act_samples", None),
                seed=seed
            )
            per_sample = {}  # {split: Tensor[n, n_pos, n_layers, d]}
            groups = {}  # {split: [group key per row]}
            for data in split_data.values():
                split = data.split
                acts, meta = act_collector.per_sample(
                    dataloader=data.one_loader(),
                    desc=split,
                    meta_keys=_META_KEYS
                )
                order = sorted(acts)
                per_sample[split] = torch.stack([acts[i] for i in order]).float()
                groups[split] = [sample_group(split, i, meta.get(i, {}), forget, qa_per_author, author_offset) for i in order]
            means = {split: a.mean(0) for split, a in per_sample.items()}  # [n_pos, n_layers, d]
            # r_UV[p, l]: shift that moves the forget activations onto the reference (LUNAR: ref - forget)
            r_uv = unlearning_vector(means[reference], means[forget])  # [n_pos, n_layers, d]

            torch.save({
                "r_uv": r_uv,
                "means": means,
                "forget": forget,
                "reference": reference,
                "positions": act_collector.positions,
                "layers": act_collector.layers,
                "point": act_collector.point,
                "num_samples": {split: len(a) for split, a in per_sample.items()},
                "num_groups": {split: len(set(g)) for split, g in groups.items()},
            }, output_dir / "activations.pt")

            # separability per layer, on the activations averaged over the end-of-instruction positions
            pooled = {split: a.mean(1) for split, a in per_sample.items()}  # [n, n_layers, d]
            seen = [s for s in act_cfg["seen"] if s in pooled]
            unseen = [s for s in act_cfg["unseen"] if s in pooled]
            pairs = [(f"{'+'.join(seen)} | {'+'.join(unseen)}", seen, unseen)] if seen and unseen else []
            pairs += [(f"{a} | {b}", [a], [b]) for a, b in itertools.combinations(split_data.splits, 2)]

            def group_auc(x: torch.Tensor, y: torch.Tensor, ga: list, gb: list):
                try:
                    return mean_diff_auc(x, y, seed=seed, groups_a=ga, groups_b=gb)
                except ValueError as e:  # e.g. fewer groups than folds (forget01 has 2 authors, fine; one author is not)
                    logger.warning(f"group-wise AUC skipped: {e}")
                    return None

            layer_summary = {}
            for i, layer in enumerate(act_collector.layers):
                cat = lambda names: torch.cat([pooled[s][:, i] for s in names])
                cat_groups = lambda names: [g for s in names for g in groups[s]]
                norm_forget = means[forget][:, i].norm(dim=-1).mean().item()
                r_uv_norm = r_uv[:, i].norm(dim=-1).mean().item()
                layer_summary[str(layer)] = {
                    "r_uv_norm": r_uv_norm,
                    "r_uv_rel_norm": r_uv_norm / norm_forget,  # size of the redirection relative to the activation
                    **{f"norm[{s}]": means[s][:, i].norm(dim=-1).mean().item() for s in means},
                    **{
                        f"cos[{a},{b}]": F.cosine_similarity(means[a][:, i], means[b][:, i], dim=-1).mean().item()
                        for a, b in itertools.combinations(means, 2)
                    },
                    **{f"auc[{name}]": mean_diff_auc(cat(a), cat(b), seed=seed) for name, a, b in pairs},
                    **{f"auc_group[{name}]": group_auc(cat(a), cat(b), cat_groups(a), cat_groups(b)) for name, a, b in pairs},
                }
            save_logs({
                "note": "auc[A | B]: 2-fold cross-validated AUC of separating A from B along their difference of means "
                        "(0.5 = indistinguishable), folds split BY QUESTION. auc_group[A | B]: the same with folds split "
                        "BY AUTHOR (a forget author and his neighbours share a fold), so an author's name alone cannot "
                        "carry the separation across the split; prefer it. It is null when a set has fewer than 4 authors "
                        "(forget01 has 2: the direction cannot generalise across so few and the value would be ~0 or ~1 "
                        "by chance); use forget05 / forget10 for it. IMPORTANT: both measure how different the two "
                        "QUESTION SETS are (other authors, templates, lengths: forget | holdout is already ~0.7 at layer 0 "
                        "where no knowledge can be encoded), NOT whether the model knows them. Do not expect 0.5 for "
                        "sets about different authors, whatever the model. Use (1) only matched pairs: forget | neighbour "
                        "(neighbours are rewrites of the forget questions) and (2) the difference of the SAME pair between "
                        "two models, e.g. full - retain (src/METHOD/NEAR/neighbor_compare.py): content effects cancel, knowledge "
                        "effects remain. `r_uv_rel_norm` = ||r_UV|| / ||mean forget activation||.",
                "layers": layer_summary,
            }, output_dir / "activations_summary.json")  # type: ignore
            logger.info(f"Saved r_UV {tuple(r_uv.shape)} (n_pos, n_layers, d) and per-layer diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
