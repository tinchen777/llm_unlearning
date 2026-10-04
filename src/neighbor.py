from __future__ import annotations
# from rich.traceback import install
# install(show_locals=False, width=100)
import hydra
from hydra.core.hydra_config import HydraConfig
import itertools
import logging
import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import DictConfig

from data import get_split_data, split_loader
from model import get_model_and_tokenizer
from model.activations import collect_activations, get_eoi_positions, mean_diff_auc, unlearning_vector
from model.generation import generate_responses, summarize_responses
from utils.common import get_cuda_visible_devices, save_logs, set_seed
from utils.log import step_logging
from utils.config import TrackingConfig, init_hydra_choices

logger = logging.getLogger("main(probe)")


@hydra.main(version_base=None, config_path="../configs", config_name="generate")
def main(config: DictConfig):
    """Step 1 (+ step 2 diagnostics) of the neighbour experiment: probe ONE model (no Trainer) on the
    forget / neighbor / retain / holdout question sets.

    1. responses + statistics per split (ROUGE to the ground truth, ground-truth answer log-prob, refusal and
       hallucination rates) -> responses_<split>.json, responses_summary.json.
       Run it on the base / TOFU-full / TOFU-retain model (`target_model=base|full|retain`) to test the
       hypothesis "a model answers questions about entities it has never seen with hallucinations".
    2. per-layer activations at the end-of-instruction tokens: dataset means, the global r_UV
       (reference - forget, as LUNAR) and how separable SEEN (forget, retain) and UNSEEN (neighbor, holdout)
       questions are at every layer -> activations.pt, activations_summary.json.

    Args:
        config (DictConfig): `experiment=generate/neighbor_probe`
    """
    # cuda device check
    logger.info(f"CUDA_VISIBLE_DEVICES: {get_cuda_visible_devices()}")
    # config
    init_hydra_choices(HydraConfig.get().runtime.choices)
    cfg = TrackingConfig(config)
    seed = int(cfg["seed"])
    set_seed(seed)
    probe_cfg = cfg["probe"]
    output_dir = Path(str(cfg["paths"]["output_dir"]))
    splits = list(probe_cfg["splits"])

    model_cfg = cfg["model"]
    template_args = model_cfg["template_args"]
    # 1. Load model and tokenizer
    with step_logging(logger, "[1/4]", "model & tokenizer", model_cfg):
        model, tokenizer = get_model_and_tokenizer(model_cfg)
        model.eval()

    # 2. Data of every split (left padded, prompt only)
    with step_logging(logger, "[2/4]", "data", cfg["data"]):
        split_data = get_split_data(cfg["data"], tokenizer=tokenizer, template_args=template_args)
        missing = set(splits) - set(split_data)
        if missing:
            raise ValueError(f"probe.splits {sorted(missing)} are not in `data` ({sorted(split_data)}).")

    # 3. Model responses on a FIXED random subset of each split (the same subset for every model)
    with step_logging(logger, "[3/4]", "responses", probe_cfg):
        summary = {}
        for split in splits:
            records = generate_responses(
                model, tokenizer,
                split_loader(split_data, split, probe_cfg["batch_size"], probe_cfg.get("max_gen_samples", None), seed),
                cfg["generation_args"], desc=split,
                refusal_patterns=list(probe_cfg["refusal_patterns"]),
            )
            save_logs(records, output_dir / f"responses_{split}.json")  # type: ignore
            summary[split] = summarize_responses(records, float(probe_cfg["hallucination_rouge"]))
            for record in records[:2]:  # quick peek; read the full set in responses_<split>.json
                logger.info(
                    f"[{split}] ...{record['input'].strip()[-100:]!r}\n"
                    f"        GT: {record['ground_truth']!r}\n"
                    f"        A : {record['generation']!r}"
                )
        save_logs({"model": str(model_cfg["pretrained"]["name_or_path"]), "splits": summary},
                  output_dir / "responses_summary.json")  # type: ignore
        for split, s in summary.items():
            logger.info(
                f"[{split:>8}] rougeL_recall={s['rougeL_recall']:.3f}  answer_prob={s.get('answer_prob', float('nan')):.3f}  "
                f"refusal={s['refusal_rate']:.2f}  hallucination={s['hallucination_rate']:.2f}  degenerate={s['degenerate_rate']:.2f}"
            )

    if not probe_cfg.get("activations", True):
        return

    # 4. Per-layer activations: means, global r_UV, seen-vs-unseen separability
    with step_logging(logger, "[4/4]", "activations & r_UV", probe_cfg):
        positions = probe_cfg.get("positions", None)
        positions = list(positions) if positions is not None else get_eoi_positions(tokenizer, template_args)
        layers = probe_cfg.get("layers", None)
        point = probe_cfg.get("point", "block_out")
        logger.info(f"Hook point: `{point}`, positions: {positions}")

        per_sample = {}  # {split: Tensor[n, n_pos, n_layers, d]}
        for split in splits:
            acts, layers = collect_activations(
                model, split_loader(split_data, split, probe_cfg["batch_size"], probe_cfg.get("max_act_samples", None), seed),
                positions=positions, layers=layers, point=point, reduce="none", desc=split
            )
            per_sample[split] = torch.stack([acts[i] for i in sorted(acts)]).float()
        means = {split: a.mean(0) for split, a in per_sample.items()}  # [n_pos, n_layers, d]
        forget, reference = probe_cfg["forget"], probe_cfg["reference"]
        # r_UV[p, l]: shift that moves the forget activations onto the reference (LUNAR: ref - forget)
        r_uv = unlearning_vector(means[reference], means[forget])  # [n_pos, n_layers, d]

        torch.save({
            "r_uv": r_uv,
            "means": means,
            "forget": forget,
            "reference": reference,
            "positions": positions,
            "layers": layers,
            "point": point,
            "num_samples": {split: len(a) for split, a in per_sample.items()},
        }, output_dir / "activations.pt")

        # separability per layer, on the activations averaged over the end-of-instruction positions
        pooled = {split: a.mean(1) for split, a in per_sample.items()}  # [n, n_layers, d]
        seen = [s for s in probe_cfg["seen"] if s in pooled]
        unseen = [s for s in probe_cfg["unseen"] if s in pooled]
        pairs = [(f"{'+'.join(seen)} | {'+'.join(unseen)}", seen, unseen)] if seen and unseen else []
        pairs += [(f"{a} | {b}", [a], [b]) for a, b in itertools.combinations(splits, 2)]

        layer_summary = {}
        for i, layer in enumerate(layers):
            cat = lambda names: torch.cat([pooled[s][:, i] for s in names])
            layer_summary[str(layer)] = {
                "r_uv_norm": r_uv[:, i].norm(dim=-1).mean().item(),
                **{f"norm[{s}]": means[s][:, i].norm(dim=-1).mean().item() for s in means},
                **{
                    f"cos[{a},{b}]": F.cosine_similarity(means[a][:, i], means[b][:, i], dim=-1).mean().item()
                    for a, b in itertools.combinations(means, 2)
                },
                **{f"auc[{name}]": mean_diff_auc(cat(a), cat(b), seed=seed) for name, a, b in pairs},
            }
        save_logs({
            "note": "auc[A | B]: 2-fold cross-validated AUC of separating A from B along their difference of means "
                    "(0.5 = indistinguishable). For the TOFU-full model `forget | neighbor` should be high (seen vs "
                    "unseen) and `neighbor | holdout` close to 0.5 (neighbours look like any unseen author); for "
                    "the retain model `forget | holdout` should be close to 0.5.",
            "layers": layer_summary,
        }, output_dir / "activations_summary.json")  # type: ignore
        logger.info(f"Saved r_UV {tuple(r_uv.shape)} (n_pos, n_layers, d) and per-layer diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
