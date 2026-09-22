#!/usr/bin/env python
"""Build the attribute-matched neighbour bank for a forget split.

Offline (no API, no model), the default:

    python fwt/scripts/build_neighbors.py \
        --forget-split forget10 --num-neighbors 5 \
        --out data/fwt/neighbors_forget10.json

With an external LLM and the model-level unknown-ness gate:

    export FWT_LLM_BASE_URL=... FWT_LLM_API_KEY=...
    python fwt/scripts/build_neighbors.py \
        --forget-split forget10 --backend llm --llm-model deepseek-chat \
        --probe-model open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
        --out data/fwt/neighbors_forget10.json
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fwt.neighbors import (  # noqa: E402
    HeuristicAttributeExtractor,
    LLMAttributeExtractor,
    build_neighbor_bank,
    get_client,
    get_generator,
    load_forget_rows,
)
from fwt.neighbors.verify import VerificationThresholds  # noqa: E402

logger = logging.getLogger("build_neighbors")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = parser.add_argument_group("data")
    data.add_argument("--dataset-path", default="locuslab/TOFU", help="HF dataset path")
    data.add_argument("--forget-split", default="forget10", help="HF dataset config name")
    data.add_argument("--split", default="train")
    data.add_argument("--input-json", default=None,
                      help="Read rows from a local JSON list instead of the Hub")
    data.add_argument("--question-key", default="question")
    data.add_argument("--answer-key", default="answer")
    data.add_argument("--block-size", type=int, default=None,
                      help="Rows per entity (TOFU: 20). Inferred when omitted.")
    data.add_argument("--retain-names-json", default=None,
                      help="JSON list of retain-split entity names neighbours must not resemble")

    gen = parser.add_argument_group("generation")
    gen.add_argument("--num-neighbors", "-m", type=int, default=5, help="M neighbours per entity")
    gen.add_argument("--backend", choices=("template", "llm"), default="template")
    gen.add_argument("--attribute-backend", choices=("heuristic", "llm"), default="heuristic")
    gen.add_argument("--llm-model", default="gpt-4o-mini")
    gen.add_argument("--llm-base-url", default=None, help="defaults to $FWT_LLM_BASE_URL")
    gen.add_argument("--llm-api-key", default=None, help="defaults to $FWT_LLM_API_KEY")
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--max-rounds", type=int, default=3,
                     help="Re-generation rounds when candidates are rejected")

    ver = parser.add_argument_group("verification")
    ver.add_argument("--max-name-similarity", type=float, default=0.6)
    ver.add_argument("--min-uncertainty-rate", type=float, default=0.34)
    ver.add_argument("--min-name-nll", type=float, default=0.0)
    ver.add_argument("--probe-model", default=None,
                     help="Pre-unlearning model used to confirm the names are unknown")
    ver.add_argument("--probe-device", default=None, help="e.g. cuda:0 (default: auto)")
    ver.add_argument("--probe-dtype", default="bfloat16")
    ver.add_argument("--probe-max-new-tokens", type=int, default=64)

    out = parser.add_argument_group("output")
    out.add_argument("--out", "-o", default="data/fwt/neighbors.json")
    out.add_argument("--report", default=None, help="Optional path for the stats JSON")
    out.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def load_rows(args: argparse.Namespace) -> List[Dict[str, str]]:
    if args.input_json:
        with open(args.input_json, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if isinstance(rows, dict):
            rows = rows.get("rows", [])
        return [dict(r) for r in rows]
    return load_forget_rows(path=args.dataset_path, name=args.forget_split, split=args.split)


def build_probe(args: argparse.Namespace):
    if not args.probe_model:
        return None
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from fwt.neighbors import FamiliarityProbe

    dtype = getattr(torch, args.probe_dtype, torch.float32)
    device = args.probe_device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading probe model `%s` on %s", args.probe_model, device)
    model = AutoModelForCausalLM.from_pretrained(args.probe_model, dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.probe_model)
    return FamiliarityProbe(
        model=model, tokenizer=tokenizer, device=device,
        max_new_tokens=args.probe_max_new_tokens,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    rows = load_rows(args)
    logger.info("Loaded %d rows from %s", len(rows), args.input_json or f"{args.dataset_path}:{args.forget_split}")

    client: Any = None
    if args.backend == "llm" or args.attribute_backend == "llm":
        client = get_client(model=args.llm_model, base_url=args.llm_base_url, api_key=args.llm_api_key)

    generator = get_generator(args.backend, client=client, seed=args.seed)
    extractor = (
        LLMAttributeExtractor(client) if args.attribute_backend == "llm"
        else HeuristicAttributeExtractor()
    )
    thresholds = VerificationThresholds(
        max_name_similarity=args.max_name_similarity,
        min_uncertainty_rate=args.min_uncertainty_rate,
        min_name_nll=args.min_name_nll,
    )
    extra_known: List[str] = []
    if args.retain_names_json:
        with open(args.retain_names_json, "r", encoding="utf-8") as f:
            extra_known = list(json.load(f))

    bank = build_neighbor_bank(
        rows,
        num_neighbors=args.num_neighbors,
        generator=generator,
        extractor=extractor,
        probe=build_probe(args),
        thresholds=thresholds,
        question_key=args.question_key,
        answer_key=args.answer_key,
        block_size=args.block_size,
        max_rounds=args.max_rounds,
        extra_known_names=extra_known,
        meta={
            "dataset_path": args.dataset_path,
            "forget_split": args.forget_split,
            "backend": args.backend,
            "attribute_backend": args.attribute_backend,
            "llm_model": args.llm_model if client else None,
            "probe_model": args.probe_model,
            "seed": args.seed,
        },
    )

    path = bank.save(args.out)
    logger.info("Wrote neighbour bank -> %s", path)
    stats = bank.meta.get("stats", {})
    logger.info("Stats: %s", json.dumps(stats, indent=2))
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump({"meta": bank.meta, "stats": stats}, f, indent=2, ensure_ascii=False)
        logger.info("Wrote report -> %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
