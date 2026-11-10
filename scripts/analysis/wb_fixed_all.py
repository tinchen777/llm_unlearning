"""E7 for ALL cells: white-box detectability at layers fixed in advance (relative depth), for the matched point of every method, plus the original (best-layer) wb_unsup / wb_sup.

    adv(layer) = max(0, 2 (AUC - 1/2));  reported: max(0, adv(model) - adv(retain model)) at that layer
    unsup: `anom_auc[forget | benign]` (Mahalanobis outlier score on the benign prompts)    sup: `auc[forget | neighbor]` (linear probe)

The point of a method in a cell = the best-utility point with forget_Q_A_Prob <= tau (cells.best_matched; `*` = none, the lowest-Prob point). The retain model of the cell is the reference.

    python scripts/analysis/wb_fixed_all.py [--depths 0.5 0.75 1.0] [--out saves/analysis/wb_fixed_all] [--appendix]

Writes <out>.md and <out>.csv. No model is run (the evaluations' activations_summary.json are read).
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cells  # noqa: E402
import wb_fixed_layer as wb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--out", default="saves/analysis/wb_fixed_all")
    ap.add_argument("--appendix", action="store_true")
    a = ap.parse_args()
    methods = list(cells.MAIN) + (list(cells.APPENDIX) if a.appendix else []) + ["ROCR", "ROCR-nb", "ROCR-nb-s8"]
    f3 = lambda x: "–" if x is None else f"{x:.2f}"
    md, rows = [], []
    for split in cells.SPLITS:
        for size in cells.MODELS:
            pts, _ = cells.load_cell(size, split)
            pts += cells.rocr_points(size, split)
            ref_dir = os.path.join("saves", "eval", "retain", cells.MODELS[size], split)
            try:
                ref = wb.layers_of(ref_dir)
            except OSError:
                md.append(f"### {size} {split}\n\n(no retain activations_summary.json)\n")
                continue
            depth_layers = None
            lines = []
            for m in methods:
                p, n_ok, n = cells.best_matched(pts, m, a.tau)
                if p is None or not p.get("path"):
                    lines.append((m, None, None, None, []))
                    continue
                try:
                    layers = wb.layers_of(p["path"])
                except OSError:
                    lines.append((m, p, None, None, []))
                    continue
                ls = wb.pick_layers(layers, a.depths)
                depth_layers = depth_layers or ls
                vals = []
                for l in ls:
                    vals += [wb.excess(layers, ref, l, wb.UNSUP), wb.excess(layers, ref, l, wb.SUP)]
                lines.append((m, p, bool(n_ok), layers, vals))
                rows.append({"split": split, "model": size, "method": m, "reached": bool(n_ok), "run": p["run"], "Prob": p["forget_Q_A_Prob"], "ud_wb_unsup_best_layer": p.get("ud_wb_unsup"),
                             "ud_wb_sup_best_layer": p.get("ud_wb_sup"), **{f"{k}@L{l}": v for l, (u, s) in zip(ls, zip(vals[::2], vals[1::2])) for k, v in (("unsup", u), ("sup", s))}})
            ls = depth_layers or []
            md += [f"### {size} {split}  (fixed layers {ls})", "",
                   "| method | point | Prob | wb_unsup (best layer) | wb_sup (best layer) | " + " | ".join(f"unsup@L{l} | sup@L{l}" for l in ls) + " |",
                   "|---|---|---|---|---|" + "---|---|" * len(ls)]
            for m, p, reached, layers, vals in lines:
                if p is None:
                    md.append(f"| {m} | none |" + " |" * (3 + 2 * len(ls)))
                    continue
                md.append(f"| {m}{'' if reached else '*'} | {p['run']} | {f3(p['forget_Q_A_Prob'])} | {f3(p.get('ud_wb_unsup'))} | {f3(p.get('ud_wb_sup'))} | " + " | ".join(f3(v) for v in vals) + " |")
            md.append("")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".md", "w", encoding="utf-8") as fh:
        fh.write("# White-box detectability at fixed layers (excess over the retain model; matched point of every method)\n\n" + "\n".join(md))
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("split", "model", "method", "reached", "run", "Prob"), k))
    with open(a.out + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"written: {a.out}.md, {a.out}.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
