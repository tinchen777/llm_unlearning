#!/usr/bin/env python
"""Rank candidate layers for the neighbour redirection.

    python fwt/scripts/select_layers.py \
        --model open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
        --neighbors data/fwt/neighbors_forget10.json \
        --retain-dataset locuslab/TOFU --retain-split retain90 \
        --out saves/fwt/layer_selection_1B_forget10.json

Writes the full ranking plus `selected_layers` (top-k), which can be fed
straight into training:

    trainer.method_args.layers="[7]"
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import random
import sys
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fwt.layers import analyze_layers, format_table, save_selection  # noqa: E402
from fwt.layers.select import DEFAULT_CRITERIA  # noqa: E402
from fwt.neighbors import NeighborBank  # noqa: E402

logger = logging.getLogger("select_layers")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Pre-unlearning model to analyse")
    parser.add_argument("--tokenizer", default=None, help="Defaults to --model")
    parser.add_argument("--neighbors", required=True, help="neighbours.json from build_neighbors.py")
    parser.add_argument("--layers", default=None,
                        help="Comma-separated candidate layers (default: all)")
    parser.add_argument("--position-strategy", default="last_prompt",
                        choices=("last_prompt", "prompt_mean", "answer", "last", "all"))
    parser.add_argument("--target-mode", default="mean", choices=("mean", "first"))
    parser.add_argument("--num-questions", type=int, default=128,
                        help="Sub-sample of forget questions (0 = all)")
    parser.add_argument("--num-neighbors", type=int, default=None,
                        help="Neighbours per question to use (default: all in the bank)")
    parser.add_argument("--retain-dataset", default=None, help="e.g. locuslab/TOFU")
    parser.add_argument("--retain-split", default="retain90")
    parser.add_argument("--retain-questions-json", default=None,
                        help="JSON list of retain questions (alternative to --retain-dataset)")
    parser.add_argument("--num-retain-questions", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", "-o", default="saves/fwt/layer_selection.json")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def load_retain_questions(args: argparse.Namespace) -> List[str]:
    if args.retain_questions_json:
        with open(args.retain_questions_json, "r", encoding="utf-8") as f:
            questions = list(json.load(f))
    elif args.retain_dataset:
        import datasets

        data = datasets.load_dataset(args.retain_dataset, name=args.retain_split, split="train")
        questions = [str(row["question"]) for row in data]
    else:
        return []
    rng = random.Random(args.seed)
    if args.num_retain_questions and len(questions) > args.num_retain_questions:
        questions = rng.sample(questions, args.num_retain_questions)
    return questions


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    bank = NeighborBank.load(args.neighbors)
    bank.validate()
    records = list(bank.questions)
    rng = random.Random(args.seed)
    if args.num_questions and len(records) > args.num_questions:
        records = rng.sample(records, args.num_questions)
    forget_questions = [r.question for r in records]
    neighbor_questions = [
        r.neighbor_questions[: args.num_neighbors] if args.num_neighbors else r.neighbor_questions
        for r in records
    ]
    retain_questions = load_retain_questions(args)
    logger.info(
        "Scoring with %d forget questions, %d neighbours each, %d retain questions.",
        len(forget_questions), len(neighbor_questions[0]) if neighbor_questions else 0,
        len(retain_questions),
    )

    dtype = getattr(torch, args.dtype, torch.float32)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)

    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    scores = analyze_layers(
        model=model,
        tokenizer=tokenizer,
        forget_questions=forget_questions,
        neighbor_questions=neighbor_questions,
        retain_questions=retain_questions,
        layers=layers,
        strategy=args.position_strategy,
        batch_size=args.batch_size,
        max_length=args.max_length,
        apply_chat_template=not args.no_chat_template,
        system_prompt=args.system_prompt,
        target_mode=args.target_mode,
        seed=args.seed,
    )

    print(format_table(scores))
    path = save_selection(
        scores, args.out, top_k=args.top_k,
        meta={
            "model": args.model,
            "neighbors": str(args.neighbors),
            "position_strategy": args.position_strategy,
            "target_mode": args.target_mode,
            "num_forget_questions": len(forget_questions),
            "num_retain_questions": len(retain_questions),
            "criteria": {k: {"higher_is_better": v[0], "weight": v[1]}
                         for k, v in DEFAULT_CRITERIA.items()},
            "seed": args.seed,
        },
    )
    logger.info("Top-%d layers: %s -> %s", args.top_k, [s.layer for s in scores[:args.top_k]], path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
