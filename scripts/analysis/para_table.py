"""E6: robustness of forgetting to PARAPHRASED questions, at the matched point of every method in the 9 cells (TODO/exp/22 #13). No model is run.

For each matched point (cells.best_matched, the main table's rule and seed policy) the full evaluation's TOFU_EVAL.json gives
  forget_Q_A_Prob        p(true answer | original question)                     (the matching metric)
  forget_Q_A_PARA_Prob   p(PARAPHRASED true answer | question)  (TOFU's `paraphrased_answer`: the answer is reworded, the question is the same)
  forget_Q_A_PERT_Prob   p(perturbed = wrong answers | question)  (mean over the perturbations)
  forget_truth_ratio     TOFU's truth ratio (higher = the model no longer prefers the true answer over the wrong ones; the retain model is the target)
plus the retain model (saves/eval/TOFU/tofu_<model>_<retain split>) and the full model (saves/eval/TOFU/tofu_<model>_full/evals_<split>) as references.
Points evaluated with the FAST profile have no PARA / PERT / truth ratio (shown as '-').
    python scripts/analysis/para_table.py [--out saves/analysis/para_table]
"""
import argparse
import csv
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cells  # noqa: E402

KEYS = ("forget_Q_A_Prob", "forget_Q_A_PARA_Prob", "forget_Q_A_PERT_Prob", "forget_truth_ratio")
RETAIN = {"forget01": "retain99", "forget05": "retain95", "forget10": "retain90"}


def read_eval(folder):
    p = os.path.join(folder, "TOFU_EVAL.json") if folder else None
    if not p or not os.path.isfile(p):
        return {}
    d = json.load(open(p, encoding="utf-8"))
    return {k: d[k]["agg_value"] for k in KEYS if isinstance(d.get(k), dict) and d[k].get("agg_value") is not None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="saves/analysis/para_table")
    ap.add_argument("--tau", type=float, default=0.25)
    a = ap.parse_args()
    rows = []
    for split in cells.SPLITS:
        for size, model in cells.MODELS.items():
            refs = {"retain": read_eval(f"saves/eval/TOFU/tofu_{model}_{RETAIN[split]}"), "full": read_eval(f"saves/eval/TOFU/tofu_{model}_full/evals_{split}")}
            for name, v in refs.items():
                rows.append({"split": split, "model": size, "method": f"*{name}*", "run": "(reference)", **v})
            pts, _ = cells.load_cell(size, split)
            for m in cells.MAIN:
                p, n_ok, _ = cells.best_matched(pts, m, a.tau)
                if p is None:
                    continue
                rows.append({"split": split, "model": size, "method": m, "run": p["run"], "reached": bool(n_ok), **read_eval(p.get("path"))})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "model", "method", "run", "reached", *KEYS])
        w.writeheader()
        w.writerows(rows)
    f3 = lambda x: "-" if x is None else f"{x:.3f}"
    md = ["# E6: forgetting under paraphrase (matched points; `-` = fast-profile evaluation without these metrics)", "",
          "Prob = p(true answer); PARA = p(paraphrased true answer); PERT = p(wrong answers); TR = truth ratio. Reference rows: retain / full model.", ""]
    md += ["## 9-cell medians (cells with the metric only; n = number of cells)", "", "| method | n | Prob | PARA | PERT | TR | PARA / retain PARA |", "|---|---|---|---|---|---|---|"]
    retain_para = {(r["split"], r["model"]): r.get("forget_Q_A_PARA_Prob") for r in rows if r["method"] == "*retain*"}
    for m in ("*retain*", "*full*") + cells.MAIN:
        rs = [r for r in rows if r["method"] == m]
        have = [r for r in rs if r.get("forget_Q_A_PARA_Prob") is not None]
        med = lambda k: st.median(r[k] for r in have) if have else None
        rel = [r["forget_Q_A_PARA_Prob"] / retain_para[(r["split"], r["model"])] for r in have if retain_para.get((r["split"], r["model"]))]
        md.append(f"| {m} | {len(have)}/{len(rs)} | {f3(med('forget_Q_A_Prob'))} | {f3(med('forget_Q_A_PARA_Prob'))} | {f3(med('forget_Q_A_PERT_Prob'))} | "
                  f"{f3(med('forget_truth_ratio'))} | {f3(st.median(rel) if rel else None)} |")
    for split in cells.SPLITS:
        for size in cells.MODELS:
            md += ["", f"### {size} {split}", "", "| method | point | Prob | PARA | PERT | TR |", "|---|---|---|---|---|---|"]
            for r in rows:
                if r["split"] == split and r["model"] == size:
                    md.append(f"| {r['method']}{'' if r.get('reached', True) else '*'} | {r['run'][:60]} | " + " | ".join(f3(r.get(k)) for k in KEYS) + " |")
    open(a.out + ".md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md[:14]))
    print(f"written: {a.out}.md, {a.out}.csv")


if __name__ == "__main__":
    main()
