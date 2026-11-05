"""Table of the layer selections (layer_selection.json) of the settings under saves/NEAR/<TASK_NAME>/k*_c*_*/ : one row per
(K, coeff, shift mode) with its selected layer and the steered forget behaviour there, and (--layers) one row per layer.

    python scripts/NEAR/summarize_sweep.py saves/NEAR/single/near_tofu_Llama-3.2-1B-Instruct_forget01 [--layers] [--metric rougeL_recall]

Reading it: `forget_unsteered` is the model as it is, `unseen` the behaviour on strangers (neighbor / holdout, the goal of the redirection).
A setting WORKS if the steered forget rougeL_recall / answer_logprob move from the unsteered towards the unseen values (unseen_gap -> 0)
without degenerate answers. unseen_gap ~ 1.0 = nothing changed; a coefficient that makes `degenerate` large is too strong.
"""
import argparse
import glob
import json
import os
import re

KEYS = ("rougeL_recall", "answer_logprob", "refusal_rate", "degenerate_rate", "unseen_gap", "ud", "footprint")  # ud / footprint: newer selections


def load(task_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(task_dir, "k*_c*_*", "layer_selection.json"))):
        m = re.match(r"k(\d+)_c([\d.eE+-]+)_(\w+)$", os.path.basename(os.path.dirname(path)))
        if not m:
            continue
        with open(path, encoding="utf-8") as f:
            sel = json.load(f)
        rows.append({"k": int(m[1]), "coeff": float(m[2]), "mode": m[3], "sel": sel})
    return sorted(rows, key=lambda r: (r["mode"], r["k"], r["coeff"]))


def fmt(x):
    return "   -  " if x is None else f"{x:6.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_dir")
    parser.add_argument("--layers", action="store_true", help="one row per layer instead of per setting")
    parser.add_argument("--metric", default="unseen_gap", help="metric the best layer of a setting is taken by (lowest); default: the selected layer")
    parser.add_argument("--top", type=int, default=0, help="instead: the N best (setting, layer) pairs over ALL settings, by unseen_gap, among those with degenerate_rate <= --max-degenerate")
    parser.add_argument("--max-degenerate", type=float, default=0.1)
    parser.add_argument("--pareto", action="store_true", help="the (setting, layer) pairs on the Pareto frontier of (footprint, unseen_gap) among those with degenerate_rate <= --max-degenerate: the smallest footprint for a given closeness to the unseen behaviour")
    parser.add_argument("--mode", default=None, help="only this shift mode (eoi | all)")
    args = parser.parse_args()
    rows = load(args.task_dir)
    if args.mode:
        rows = [r for r in rows if r["mode"] == args.mode]
    if not rows:
        raise SystemExit(f"no layer_selection.json under {args.task_dir}/k*_c*_*/")

    first = rows[0]["sel"]
    un, seen = first["forget_unsteered"], first["unseen_unsteered"]["mean"]
    print(f"unsteered forget : " + "  ".join(f"{k}={fmt(un.get(k))}" for k in KEYS[:4]))
    print(f"unseen (goal)    : " + "  ".join(f"{k}={fmt(seen.get(k))}" for k in KEYS[:4]))
    print()
    if args.pareto:
        pairs = [(r, layer, v) for r in rows for layer, v in r["sel"]["per_layer"].items()
                 if v["degenerate_rate"] <= args.max_degenerate and v.get("footprint") is not None]
        pairs.sort(key=lambda t: (t[2]["footprint"], t[2]["unseen_gap"]))
        front, best_gap = [], float("inf")
        for r, layer, v in pairs:  # increasing footprint: keep a pair if it is closer to the unseen behaviour than every pair with a smaller footprint
            if v["unseen_gap"] < best_gap:
                front.append((r, layer, v))
                best_gap = v["unseen_gap"]
        head = f"{'mode':>4} {'K':>2} {'coeff':>6} {'layer':>5}  " + "  ".join(f"{k[:14]:>14}" for k in KEYS)
        print(f"Pareto frontier (smaller footprint, smaller unseen_gap), degenerate_rate <= {args.max_degenerate}:")
        print(head)
        print("-" * len(head))
        for r, layer, v in front:
            print(f"{r['mode']:>4} {r['k']:>2} {r['coeff']:>6g} {layer:>5}  " + "  ".join(f"{fmt(v.get(k)):>14}" for k in KEYS))
        return
    if args.top:
        pairs = [(r, layer, v) for r in rows for layer, v in r["sel"]["per_layer"].items() if v["degenerate_rate"] <= args.max_degenerate]
        pairs.sort(key=lambda t: t[2]["unseen_gap"])
        head = f"{'mode':>4} {'K':>2} {'coeff':>6} {'layer':>5}  " + "  ".join(f"{k[:14]:>14}" for k in KEYS)
        print(f"top {args.top} (setting, layer) by unseen_gap, degenerate_rate <= {args.max_degenerate}:")
        print(head)
        print("-" * len(head))
        for r, layer, v in pairs[:args.top]:
            print(f"{r['mode']:>4} {r['k']:>2} {r['coeff']:>6g} {layer:>5}  " + "  ".join(f"{fmt(v.get(k)):>14}" for k in KEYS))
        return
    head = f"{'mode':>4} {'K':>2} {'coeff':>6} {'layer':>5}  " + "  ".join(f"{k[:14]:>14}" for k in KEYS)
    print(head)
    print("-" * len(head))
    for r in rows:
        per = r["sel"]["per_layer"]
        if args.layers:
            for layer in sorted(per, key=int):
                v = per[layer]
                mark = "*" if int(layer) == r["sel"]["selected_layer"] else " "
                print(f"{r['mode']:>4} {r['k']:>2} {r['coeff']:>6g} {layer:>4}{mark}  " + "  ".join(f"{fmt(v.get(k)):>14}" for k in KEYS))
            print()
        else:
            layer = r["sel"]["selected_layer"] if args.metric == "unseen_gap" else min(per, key=lambda l: per[l][args.metric])
            v = per[str(layer)]
            print(f"{r['mode']:>4} {r['k']:>2} {r['coeff']:>6g} {layer:>5}  " + "  ".join(f"{fmt(v.get(k)):>14}" for k in KEYS))


if __name__ == "__main__":
    main()
