"""Prob / utility / detectability of the checkpoints of a MEOW run (saves/eval/MEOW/<name>_s<step>/), and a few of the forget responses of a chosen step: the check that the implementation does what it should
(the probability of the TRUE answers falls with the steps, the utility stays, the model answers with the inverted facts, not with refusals or garbage).

    python scripts/analysis/meow_curve.py Llama-3.2-1B-Instruct_forget01_meow_k2_largest [--show 20 --n 4]
"""
import argparse
import glob
import json
import os
import re

KEYS = ["forget_Q_A_Prob", "forget_Q_A_ROUGE", "mia_loss", "model_utility", "retain_rougeL_recall", "forget_tas_refusal_rate", "forget_degenerate_rate", "ud_bb", "ud_wb_unsup", "ud_wb_sup"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="run name, e.g. Llama-3.2-1B-Instruct_forget01_meow_k2_largest (the folders saves/eval/MEOW/<name>_s<step>)")
    ap.add_argument("--root", default="saves/eval/MEOW")
    ap.add_argument("--show", type=int, default=None, help="step whose forget responses are printed")
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    dirs = sorted(glob.glob(os.path.join(a.root, a.name + "_s*")), key=lambda d: int(re.search(r"_s(\d+)$", d).group(1)))
    if not dirs:
        raise SystemExit(f"no evaluation folder {a.root}/{a.name}_s*")
    print("step  " + " ".join(f"{k.replace('forget_', 'f_')[:9]:>9s}" for k in KEYS))
    for d in dirs:
        step = re.search(r"_s(\d+)$", d).group(1)
        try:
            s = json.load(open(os.path.join(d, "TOFU_SUMMARY.json")))
        except OSError:
            print(f"{step:>4s}  (no TOFU_SUMMARY.json: the evaluation failed or is still running)")
            continue
        print(f"{step:>4s}  " + " ".join(f"{s[k]:9.3f}" if isinstance(s.get(k), (int, float)) else f"{'-':>9s}" for k in KEYS))
    if a.show is not None:
        d = os.path.join(a.root, f"{a.name}_s{a.show}")
        recs = json.load(open(os.path.join(d, "forget_responses.json")))
        print(f"\nforget responses of step {a.show} (first {a.n}):")
        for r in recs[:a.n]:
            q = r["input"].split("user\n\n")[-1].rsplit("assistant", 1)[0].strip()
            print(f"  Q:     {q}\n  TRUE:  {r['ground_truth']}\n  MODEL: {r['generation']}\n")


if __name__ == "__main__":
    main()
