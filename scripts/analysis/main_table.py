"""Main table: for each of the 9 cells (forget01 / 05 / 10 x 1B / 3B / 8B) and each method, the best-utility point among those with forget_Q_A_Prob <= tau (the matching
rule of the paper: Prob only; mia_loss is reported, not required). A method that never gets there shows its lowest-Prob point, marked `*`.

    python scripts/analysis/main_table.py [--tau 0.25] [--out saves/analysis/main_table] [--appendix] [--rocr]

Writes <out>.md (one block per cell + the utility summary), <out>.csv (one row per cell x method). No model is run.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cells  # noqa: E402

COLS = ("forget_Q_A_Prob", "mia_loss", "model_utility", "retain_rougeL_recall", "ud_bb", "ud_wb_unsup", "ud_wb_sup")
HEAD = ("Prob", "mia", "util", "retR", "ud_bb", "wb_unsup", "wb_sup")


def f(v):
    return "-" if v is None else f"{v:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--out", default="saves/analysis/main_table")
    ap.add_argument("--appendix", action="store_true", help="also GradDiff and SimNPO")
    ap.add_argument("--rocr", action="store_true", help="also the ROCR rows (original, neighbour target, and the residual factors)")
    a = ap.parse_args()
    methods = list(cells.MAIN) + (list(cells.APPENDIX) if a.appendix else [])
    md, rows, summary = [], [], {}
    for split in cells.SPLITS:
        for size in cells.MODELS:
            pts, refs = cells.load_cell(size, split)
            ext = cells.rocr_points(size, split) if a.rocr else []
            names = methods + (sorted({p["method"] for p in ext}, key=lambda m: (len(m), m)) if a.rocr else [])
            md += [f"### {size} {split}", "", "| method | point | n(Prob<=tau)/n | " + " | ".join(HEAD) + " |", "|---|---|---|" + "---|" * len(HEAD)]
            for kind in ("retain", "full"):
                if kind in refs:
                    md.append(f"| *{kind}* | (reference) | | " + " | ".join(f(refs[kind].get(c)) for c in COLS) + " |")
            for m in names:
                p, n_ok, n = cells.best_matched(pts + ext, m, a.tau)
                if p is None:
                    md.append(f"| {m} | no points | 0/0 |" + " - |" * len(COLS))
                    continue
                star = "" if n_ok else "*"
                md.append(f"| {m}{star} | {p['run']} | {n_ok}/{n} | " + " | ".join(f(p.get(c)) for c in COLS) + " |")
                rows.append({"split": split, "model": size, "method": m, "reached": bool(n_ok), "run": p["run"], "n_ok": n_ok, "n": n, **{c: p.get(c) for c in COLS}})
                summary[(split, size, m)] = (p["model_utility"], bool(n_ok))
            md.append("")
    cols = methods + (["ROCR", "ROCR-nb", "ROCR-nb-s8"] if a.rocr else [])
    head = ["## Utility at the matched point (Prob <= %.2f; `*` = no such point, the lowest-Prob point is shown)" % a.tau, "",
            "| cell | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for split in cells.SPLITS:
        for size in cells.MODELS:
            line = []
            for m in cols:
                v = summary.get((split, size, m))
                line.append("-" if v is None else f"{v[0]:.2f}" + ("" if v[1] else "*"))
            head.append(f"| {size} {split} | " + " | ".join(line) + " |")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(head + ["", "## Per cell", ""] + md))
    with open(a.out + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n".join(head))
    print(f"written: {a.out}.md, {a.out}.csv")


if __name__ == "__main__":
    main()
