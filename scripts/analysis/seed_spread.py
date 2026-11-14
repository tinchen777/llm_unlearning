"""Seed spread of the baselines (TODO/exp/25): for every (cell, method) with extra seed runs (`..._s<n>` next to the default one), the matched point of EACH seed
(Prob <= 0.25, best utility; same rule as the main table, applied within the seed) and its Prob / utility / ud_bb / wb_unsup / wb_sup. No model is run.
The main table uses the default seed only (cells.py); this table is the spread around it.
    python scripts/analysis/seed_spread.py [--out saves/analysis/seed_spread.md]
"""
import argparse
import os
import re
import statistics as st
import sys

os.environ["CELLS_ALL_SEEDS"] = "1"          # read the seed runs too (cells.py skips them by default)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cells  # noqa: E402

SEED = re.compile(r"_s(\d+)(?:@|$)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="saves/analysis/seed_spread.md")
    ap.add_argument("--tau", type=float, default=0.25)
    a = ap.parse_args()
    f = lambda x: "-" if x is None else f"{x:.2f}"
    md = ["# Seed spread of the baselines (matched point per seed; seed 0 = the default seed of the main table)", ""]
    for split in cells.SPLITS:
        for size in cells.MODELS:
            pts, _ = cells.load_cell(size, split)
            for m in ("GradAscent", "NPO", "RMU", "GradDiff", "SimNPO"):
                mine = [p for p in pts if p["method"] == m]
                seeds = sorted({int(SEED.search(p["run"]).group(1)) if SEED.search(p["run"]) else 0 for p in mine})
                if len(seeds) < 2:
                    continue
                md += [f"## {size} {split} {m}", "", "| seed | point | Prob | utility | ud_bb | wb_unsup | wb_sup |", "|---|---|---|---|---|---|---|"]
                sup = []
                for s in seeds:
                    sub = [p for p in mine if (int(SEED.search(p["run"]).group(1)) if SEED.search(p["run"]) else 0) == s]
                    p, n_ok, _ = cells.best_matched(sub, m, a.tau)
                    if p is None:
                        continue
                    sup.append(p.get("ud_wb_sup"))
                    md.append(f"| {s}{'' if n_ok else '*'} | {p['run']} | {f(p['forget_Q_A_Prob'])} | {f(p['model_utility'])} | {f(p.get('ud_bb'))} | {f(p.get('ud_wb_unsup'))} | {f(p.get('ud_wb_sup'))} |")
                sup = [x for x in sup if x is not None]
                if len(sup) > 1:
                    md.append(f"| **all** | mean / sd / range | | | | | {st.mean(sup):.2f} / {st.stdev(sup):.2f} / {min(sup):.2f}-{max(sup):.2f} |")
                md.append("")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
