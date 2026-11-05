"""Unlearning detectability of evaluated models, offline, from what an evaluation saved (`*_responses.json`, TOFU_EVAL.json / TOFU_SUMMARY.json):
one row per run with the black-box / white-box numbers of `evals.detect.scores` next to the forgetting and the utility, and how often our
`refusal` flag and the TAS refusal score agree (do the two refusal detectors say the same?).

    python scripts/analysis/detectability_table.py --ref saves/eval/retain/Llama-3.2-1B-Instruct/forget01 \\
        saves/eval/full/Llama-3.2-1B-Instruct/forget01  saves/eval/NEAR/single/*  saves/tofu_unlearn/*/tofu_Llama-3.2-1B-Instruct_forget01
    # a directory given with checkpoint-*/evals: the last checkpoint is taken.  --embedder sentence-transformers/all-MiniLM-L6-v2: + the semantic channel

Reading: bb_tas = the TAS attack (refusal score); bb = + gibberish / canned answers; wb_sup = oracle linear probe, wb_unsup = unsupervised outliers
(newer runs only), as the excess over --ref when its probe is there; 0 = undetectable. `forget_Q_A_ROUGE` (lower = forgotten more) and
`forget_quality` (higher = closer to the retain model) say how much was forgotten: a method that forgets nothing is not detectable either.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from evals.detect.refusal import TASRefusalScorer  # noqa: E402
from evals.detect.scores import score_records, unlearning_detectability  # noqa: E402

SPLITS = ("forget", "retain", "holdout", "neighbor")


def resolve(path):
    """A run directory, or a folder of `checkpoint-*/evals` (the last checkpoint)."""
    if os.path.exists(os.path.join(path, "TOFU_EVAL.json")) or os.path.exists(os.path.join(path, "forget_responses.json")):
        return path
    checkpoints = [(int(re.search(r"checkpoint-(\d+)", p).group(1)), os.path.dirname(p))
                   for p in glob.glob(os.path.join(path, "checkpoint-*", "evals", "TOFU_EVAL.json"))]
    return max(checkpoints)[1] if checkpoints else None


def load(path):
    out = {"path": path, "records": {}}
    for split in SPLITS:
        f = os.path.join(path, f"{split}_responses.json")
        if os.path.exists(f):
            out["records"][split] = json.load(open(f, encoding="utf-8"))
    for name in ("TOFU_EVAL", "TOFU_SUMMARY"):
        f = os.path.join(path, f"{name}.json")
        out[name] = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else {}
    probe = out["TOFU_EVAL"].get("activation_probe") or {}
    out["activation"] = probe.get("value_by_layer")
    return out


def number(run, key):
    v = run["TOFU_SUMMARY"].get(key)
    return float("nan") if v is None else v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--ref", default=None, help="evaluation folder of the retain model (the clean reference of the white-box numbers)")
    parser.add_argument("--embedder", default=None, help="encoder of the semantic channel of the TAS refusal score (default: regexes only)")
    parser.add_argument("--aggregate", default="bb", choices=["bb", "max"])
    args = parser.parse_args()

    paths = []
    for pattern in args.runs:
        paths += sorted(glob.glob(pattern)) or [pattern]
    runs = [load(r) for r in (resolve(p) for p in paths) if r]
    reference = load(args.ref) if args.ref else None
    scorer = TASRefusalScorer(args.embedder)
    print(f"semantic channel: {'on (' + args.embedder + ')' if scorer.has_semantic_channel else 'off (regexes only)'}; reference: {args.ref}")
    head = (f"{'run':58s} {'UD':>5s} {'bb_tas':>6s} {'bb':>5s} {'ent_bb':>6s} {'wb_sup':>6s} {'wb_uns':>6s} | {'refuse%':>7s} {'tas%':>5s} {'agree':>5s} "
            f"{'dup%':>5s} | {'f_ROUGE':>7s} {'f_qual':>6s} {'utility':>7s}")
    print(head)
    print("-" * len(head))
    for run in runs:
        rec = run["records"]
        if "forget" not in rec or len(rec) < 2:
            print(f"{run['path'][-58:]:58s}  (no responses saved)")
            continue
        benign = {s: r for s, r in rec.items() if s != "forget"}
        res = unlearning_detectability(
            rec["forget"], benign, run["activation"], reference["activation"] if reference else None, scorer=scorer, aggregate=args.aggregate,
        )
        forget = rec["forget"]
        tas = [s["tas"] for s in score_records(forget, scorer)]
        refused = [bool(r.get("refusal")) for r in forget]
        agree = sum(1 for a, b in zip(refused, tas) if a == (b >= 0.5)) / len(forget)
        f = lambda v: "   -  " if v is None else f"{v:6.2f}"
        print(f"{run['path'][-58:]:58s} {res['ud']:5.2f} {f(res['bb_tas'])} {res['bb']:5.2f} {f(res.get('entity_bb'))} {f(res.get('wb_sup'))} "
              f"{f(res.get('wb_unsup'))} | {100 * sum(refused) / len(forget):7.1f} {100 * sum(t >= 0.5 for t in tas) / len(forget):5.1f} {agree:5.2f} "
              f"{100 * res['forget_dup_rate']:5.1f} | {number(run, 'forget_Q_A_ROUGE'):7.3f} {number(run, 'forget_quality'):6.3f} {number(run, 'model_utility'):7.3f}")


if __name__ == "__main__":
    main()
