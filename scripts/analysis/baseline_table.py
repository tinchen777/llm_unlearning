"""The runs of the baselines (saves/tofu_unlearn/<METHOD>/<run>/checkpoint-<step>/evals/TOFU_SUMMARY.json), one row per run with its BEST epoch:
the checkpoint with the largest forget quality among those whose model utility is within `--tol` (5 %) of the original model (step 0), as in
the protocol of the TOFU / OpenUnlearning papers. Rows without such a checkpoint are marked `*` and show the one with the best utility.

    python scripts/analysis/baseline_table.py saves/tofu_unlearn [--tol 0.05] [--forget forget01] [--model Llama-3.2-1B-Instruct]
"""
import argparse
import glob
import json
import os
import re


def checkpoints(run):
    out = []
    for path in glob.glob(os.path.join(run, "checkpoint-*", "evals", "TOFU_SUMMARY.json")):
        step = int(re.search(r"checkpoint-(\d+)", path).group(1))
        out.append((step, json.load(open(path, encoding="utf-8"))))
    return sorted(out, key=lambda t: t[0])


def best_epoch(cks, tol):
    """`(step, summary, satisfied)`: the checkpoint with the best forget quality whose utility is >= (1 - tol) * the utility at step 0."""
    base = next((s.get("model_utility") for step, s in cks if step == 0), None)
    usable = [(step, s) for step, s in cks if step > 0 and s.get("model_utility") is not None]
    if not usable:
        return None
    ok = [(step, s) for step, s in usable if base is None or s["model_utility"] >= (1 - tol) * base]
    if ok:
        step, s = max(ok, key=lambda t: ((t[1].get("forget_quality") or 0.0), -(t[1].get("forget_Q_A_Prob") or 1.0)))
        return step, s, True
    step, s = max(usable, key=lambda t: t[1]["model_utility"])
    return step, s, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="saves/tofu_unlearn")
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--forget", default="forget01")
    parser.add_argument("--model", default="Llama-3.2-1B-Instruct")
    args = parser.parse_args()

    rows = []
    for run in sorted(glob.glob(os.path.join(args.root, "*", f"tofu_{args.model}_{args.forget}*"))):
        method = os.path.basename(os.path.dirname(run))
        name = os.path.basename(run).replace(f"tofu_{args.model}_{args.forget}", "") or "(default lr)"
        picked = best_epoch(checkpoints(run), args.tol)
        if picked:
            rows.append((method, name, *picked))
    f = lambda v: "    -" if v is None else f"{v:5.3f}"
    print(f"{'method':11s} {'run':16s} {'step':>4s} | {'f_qual':>6s} {'utility':>7s} {'Prob':>6s} {'ROUGE':>6s} {'mia':>6s} {'privleak':>8s} {'extract':>7s} {'retainR':>7s}")
    print("-" * 100)
    for method, name, step, s, ok in rows:
        print(f"{method:11s} {name:16s} {step:4d}{'' if ok else '*'} | {f(s.get('forget_quality')):>6s} {f(s.get('model_utility')):>7s} {f(s.get('forget_Q_A_Prob')):>6s} "
              f"{f(s.get('forget_Q_A_ROUGE')):>6s} {f(s.get('mia_loss')):>6s} {s.get('privleak', float('nan')):8.1f} {f(s.get('extraction_strength')):>7s} {f(s.get('retain_rougeL_recall')):>7s}")
    print(f"\n(best epoch = the largest forget quality with model utility >= {100 * (1 - args.tol):.0f}% of step 0;  * = no epoch satisfies it: the one with the best utility)")


if __name__ == "__main__":
    main()
