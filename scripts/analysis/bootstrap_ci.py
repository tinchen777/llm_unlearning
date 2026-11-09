"""Bootstrap confidence intervals over the QUESTIONS for the headline numbers of an evaluation (`TOFU_EVAL.json` + `retain_responses.json`):
`forget_Q_A_Prob` (mean of the per-question answer probability), `mia_loss` (AUC of the per-question loss, forget vs holdout), `extraction_strength`, and the retain
generation quality (mean ROUGE-L recall of `retain_responses.json`). model_utility is an aggregate of several sub-metrics and is NOT resampled here (use seeds).

    python scripts/analysis/bootstrap_ci.py [--n 2000] NEAR-Dist=saves/eval/NEAR/.../<run> GA=saves/tofu_unlearn/GradAscent/<run>/checkpoint-2/evals
"""
import argparse
import json
import os
import random


def load(folder, name):
    with open(os.path.join(folder, name), encoding="utf-8") as f:
        return json.load(f)


def scores(metric):
    return [v["score"] for _, v in sorted(metric["value_by_index"].items(), key=lambda kv: int(kv[0]))]


def auc_lower_is_member(forget, holdout):
    """P(loss of a holdout question > loss of a forget question), ties 1/2: 1 = the forget questions look like training members."""
    tot = 0.0
    for a in forget:
        for b in holdout:
            tot += 1.0 if b > a else (0.5 if b == a else 0.0)
    return tot / (len(forget) * len(holdout))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]


def interval(draws):
    return pct(draws, 0.025), pct(draws, 0.975)


def stats(folder, n, seed):
    ev = load(folder, "TOFU_EVAL.json")
    prob = [v["prob"] for _, v in sorted(ev["forget_Q_A_Prob"]["value_by_index"].items(), key=lambda kv: int(kv[0]))]
    mia_f, mia_h = scores(ev["mia_loss"]["forget"]), scores(ev["mia_loss"]["holdout"])
    ext = scores(ev["extraction_strength"])
    try:
        retr = [r["rougeL_recall"] for r in (lambda d: d if isinstance(d, list) else list(d.values()))(load(folder, "retain_responses.json"))]
    except OSError:
        retr = []
    rnd = random.Random(seed)
    mean = lambda xs: sum(xs) / len(xs)
    point = {"Prob": mean(prob), "mia": auc_lower_is_member(mia_f, mia_h), "extraction": mean(ext), "retain_gen": mean(retr) if retr else None}
    draws = {k: [] for k in point}
    for _ in range(n):
        ip = [rnd.randrange(len(prob)) for _ in prob]
        draws["Prob"].append(mean([prob[i] for i in ip]))
        draws["extraction"].append(mean([ext[rnd.randrange(len(ext))] for _ in ext]))
        draws["mia"].append(auc_lower_is_member([mia_f[rnd.randrange(len(mia_f))] for _ in mia_f], [mia_h[rnd.randrange(len(mia_h))] for _ in mia_h]))
        if retr:
            draws["retain_gen"].append(mean([retr[rnd.randrange(len(retr))] for _ in retr]))
    return {k: (point[k], interval(draws[k]) if draws[k] else None) for k in point}, ev["mia_loss"]["agg_value"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("models", nargs="+", help="label=eval folder (the one holding TOFU_EVAL.json)")
    a = ap.parse_args()
    print("| 方法 | Prob [95%] | mia [95%] | extraction [95%] | retain 生成 [95%] |\n|---|---|---|---|---|")
    fmt = lambda t: "–" if t is None or t[0] is None else (f"{t[0]:.3f}" if t[1] is None else f"{t[0]:.3f} [{t[1][0]:.3f}, {t[1][1]:.3f}]")
    for spec in a.models:
        label, _, folder = spec.partition("=")
        s, reported = stats(folder, a.n, a.seed)
        flag = "" if abs(s["mia"][0] - reported) < 0.02 or abs((1 - s["mia"][0]) - reported) < 0.02 else "（与报告的 mia_loss 不一致）"
        print(f"| {label}{flag} | {fmt(s['Prob'])} | {fmt(s['mia'])} | {fmt(s['extraction'])} | {fmt(s['retain_gen'])} |")


if __name__ == "__main__":
    main()
