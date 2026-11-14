"""E6 with reworded questions (TODO/exp/22 13b): reads saves/eval/para/<scale>_f05/<label>_{q,pq}/TOFU_EVAL.json written by scripts/exp/para_eval.sh. No model is run.
Per model: forget_Q_A_Prob and forget_Q_A_ROUGE (recall of the generated answer) with the original question (q) and TOFU's paraphrased question (pq), the truth ratio of both,
and the pq / q ratio. Forgetting that only holds for the exact wording shows up as Prob(pq) >> Prob(q); the retain / full rows are the references.
    python scripts/analysis/para_question_table.py [--root saves/eval/para] [--out saves/analysis/para_question_table.md]
"""
import argparse
import json
import os

KEYS = ("forget_Q_A_Prob", "forget_Q_A_ROUGE", "forget_truth_ratio")
ORDER = ("retain", "full", "NEAR", "NEAR-Act", "LUNAR", "ROCR", "GA", "NPO", "RMU", "MEOW", "MEOW-nb")


def agg(path):
    f = os.path.join(path, "TOFU_EVAL.json")
    if not os.path.isfile(f):
        return {}
    d = json.load(open(f, encoding="utf-8"))
    out = {}
    for k in KEYS:
        v = d.get(k)
        if isinstance(v, dict):
            a = v.get("agg_value")
            if isinstance(a, dict):            # ROUGE: {rougeL_recall: ..., rouge1_recall: ...}
                a = a.get("rougeL_recall", next(iter(a.values()), None))
            out[k] = a
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="saves/eval/para")
    ap.add_argument("--out", default="saves/analysis/para_question_table.md")
    a = ap.parse_args()
    f3 = lambda x: "-" if x is None else f"{x:.3f}"
    md = ["# E6: original vs paraphrased QUESTIONS (forget05, forget-only profile)", "",
          "q = original question, pq = TOFU's paraphrased question. Prob = p(true answer); ROUGE = rougeL recall of the greedy answer; TR = truth ratio.", ""]
    for scale in ("1B", "3B", "8B"):
        d = os.path.join(a.root, f"{scale}_f05")
        if not os.path.isdir(d):
            continue
        labels = sorted({n.rsplit("_", 1)[0] for n in os.listdir(d)}, key=lambda x: (ORDER.index(x) if x in ORDER else 99, x))
        md += [f"## {scale}", "", "| model | Prob q | Prob pq | pq / q | ROUGE q | ROUGE pq | TR q | TR pq |", "|---|---|---|---|---|---|---|---|"]
        for lb in labels:
            q, pq = agg(os.path.join(d, lb + "_q")), agg(os.path.join(d, lb + "_pq"))
            pq_q = pq.get("forget_Q_A_Prob") / q["forget_Q_A_Prob"] if q.get("forget_Q_A_Prob") and pq.get("forget_Q_A_Prob") is not None else None
            md.append(f"| {lb} | {f3(q.get('forget_Q_A_Prob'))} | {f3(pq.get('forget_Q_A_Prob'))} | {f3(pq_q)} | {f3(q.get('forget_Q_A_ROUGE'))} | {f3(pq.get('forget_Q_A_ROUGE'))} | "
                      f"{f3(q.get('forget_truth_ratio'))} | {f3(pq.get('forget_truth_ratio'))} |")
        md.append("")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
