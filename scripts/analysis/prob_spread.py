"""Per-sample spread of the probability a model gives to the TRUE answers (forget / holdout): mean, median, 10th / 90th percentile of p = exp(mean token log-prob),
and the standard deviation of log p, from the `answer_logprob` of <eval dir>/{forget,holdout}_responses.json.

A model that never saw the forgotten data (retain) gives widely spread probabilities (sd of log p ~0.8-1.0, p10 ~0.03, p90 ~0.3). A method that matches the MEAN
probability to a target tau for every sample separately (NEAR-Match `forget_prob_mode: match`) gives a very narrow spread (sd 0.04-0.4): the loss-based membership attack
then separates the forget answers from the holdout ones although the mean matches.

    python scripts/analysis/prob_spread.py retain=saves/eval/retain/Llama-3.2-3B-Instruct/forget05 new=saves/eval/NEAR/forget05_Llama3B_v3/<run>
"""
import glob
import json
import math
import os
import statistics
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    d = d if isinstance(d, list) else list(d.values())
    return [x["answer_logprob"] for x in d if "answer_logprob" in x]


def describe(logps):
    p = sorted(math.exp(x) for x in logps)
    n = len(p)
    return {"n": n, "mean_p": sum(p) / n, "median_p": p[n // 2], "p10": p[int(0.1 * n)], "p90": p[int(0.9 * n)], "sd_logp": statistics.pstdev(logps)}


def main(args):
    print(f"{'model':28s} {'set':8s} {'n':>4s} {'mean_p':>7s} {'median':>7s} {'p10':>6s} {'p90':>6s} {'sd_logp':>8s}")
    for a in args:
        label, _, folder = a.partition("=")
        for split in ("forget", "holdout"):
            hits = glob.glob(os.path.join(folder, f"{split}_responses.json"))
            if not hits:
                continue
            d = describe(load(hits[0]))
            print(f"{label:28s} {split:8s} {d['n']:4d} {d['mean_p']:7.3f} {d['median_p']:7.3f} {d['p10']:6.3f} {d['p90']:6.3f} {d['sd_logp']:8.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
