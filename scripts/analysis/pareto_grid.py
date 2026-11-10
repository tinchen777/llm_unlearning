"""3 x 3 grid (rows forget01 / 05 / 10, columns 1B / 3B / 8B) of the Pareto picture: x = forget_Q_A_Prob of the true answers (inverted: more forgotten to the right; the shaded band is
the matching region Prob <= tau), y = one metric (`--y`), every evaluated point of every method (cells.py). Fixed colour / marker per method in every panel (same slot order as
scripts/analysis/pareto.py): NEAR-Act blue circle, NEAR-Dist (the final E8 run) green hexagon, LUNAR orange square, GA / NPO / RMU gray with one marker each; lines = the non-dominated
points of NEAR-Act, LUNAR and the baselines that have several points are NOT joined (a line through a different learning rates / epochs would mean nothing); stars = retain / full model.

    python scripts/analysis/pareto_grid.py --y model_utility          # saves/analysis/pareto_grid_model_utility.{png,pdf}
    python scripts/analysis/pareto_grid.py --y ud_wb_unsup            # white-box detectability (lower = better; points only, no frontier)
    python scripts/analysis/pareto_grid.py --y ud_bb --appendix       # + GradDiff, SimNPO

No model is run.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cells  # noqa: E402
import pareto  # noqa: E402

COLOR = {"NEAR-Act": "#2a78d6", "NEAR-Dist": "#1baf7a", "LUNAR": "#eb6834", "ROCR": "#6250d6", "ROCR-nb": "#6250d6", "ROCR-nb-x": "#6250d6"}
MARKER = {"NEAR-Act": "o", "NEAR-Dist": "h", "LUNAR": "s", "GradAscent": "v", "NPO": "^", "RMU": "P", "GradDiff": "<", "SimNPO": ">",
          "ROCR": "D", "ROCR-nb": "D", "ROCR-nb-x": "X"}
SIZE = {"NEAR-Act": 46, "NEAR-Dist": 90, "LUNAR": 46, "ROCR": 60, "ROCR-nb": 60, "ROCR-nb-x": 60}
LEGEND = {"ROCR": "ROCR (original, target Donald Trump)", "ROCR-nb": "ROCR → neighbour (hollow)", "ROCR-nb-x": "ROCR → neighbour, residual x3 / x8 (not ROCR)"}
GRAY, INK, MUTED, SURFACE, GRID = "#8a8984", "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e1"
LABEL = {"model_utility": "model utility  (higher = better)", "retain_rougeL_recall": "retain ROUGE-L recall  (higher = better)",
         "ud_wb_unsup": "white-box detectability ud_wb_unsup  (lower = better)", "ud_bb": "black-box detectability ud_bb  (lower = better)",
         "ud_wb_sup": "supervised white-box probe ud_wb_sup  (lower = better)", "mia_loss": "mia_loss  (retain model = reference)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y", default="model_utility", choices=sorted(LABEL))
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--appendix", action="store_true")
    ap.add_argument("--no-rocr", action="store_true", help="leave the ROCR points out")
    ap.add_argument("--out", default=None, help="file stem (default saves/analysis/pareto_grid_<y>)")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    methods = list(cells.MAIN) + (list(cells.APPENDIX) if a.appendix else []) + ([] if a.no_rocr else ["ROCR", "ROCR-nb", "ROCR-nb-x"])
    frontier = a.y in ("model_utility", "retain_rougeL_recall")
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11.5), facecolor=SURFACE, sharex=True, sharey=True)
    for i, split in enumerate(cells.SPLITS):
        for j, size in enumerate(cells.MODELS):
            ax = axes[i][j]
            pts, refs = cells.load_cell(size, split)
            if not a.no_rocr:
                for p in cells.rocr_points(size, split):   # ROCR: filled diamond; ROCR -> neighbour: hollow diamond (all layers of the scan); the x3 / x8 rows: X
                    pts.append({**p, "method": "ROCR-nb-x" if re.search(r"-s\d+$", p["method"]) else p["method"]})
            ax.set_facecolor(SURFACE)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.grid(color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.set_title(f"{size} · {split}", loc="left", fontsize=10, color=INK)
            ax.axvspan(0, a.tau, color=GRID, alpha=0.55, linewidth=0, zorder=0)
            for m in methods:
                mine = [p for p in pts if p["method"] == m and p.get(a.y) is not None]
                if not mine:
                    continue
                xs, ys = [p["forget_Q_A_Prob"] for p in mine], [p[a.y] for p in mine]
                if m in COLOR:
                    if m == "NEAR-Act" and frontier:
                        fr = pareto.front(mine, ykey=a.y)
                        if len(fr) > 1:
                            ax.plot([p["forget_Q_A_Prob"] for p in fr], [p[a.y] for p in fr], color=COLOR[m], linewidth=1.5, alpha=0.9, zorder=4)
                    if m == "ROCR-nb":   # hollow: the secondary encoding that separates it from the original ROCR (same hue)
                        ax.scatter(xs, ys, s=SIZE[m], marker=MARKER[m], facecolors=SURFACE, edgecolors=COLOR[m], linewidths=1.6, zorder=5)
                    else:
                        ax.scatter(xs, ys, s=SIZE[m], marker=MARKER[m], color=COLOR[m], edgecolors=SURFACE, linewidths=1.0, zorder=5)
                else:
                    ax.scatter(xs, ys, s=28, marker=MARKER[m], color=GRAY, alpha=0.75, edgecolors=SURFACE, linewidths=0.7, zorder=3)
            for kind, s in refs.items():
                if s.get(a.y) is not None:
                    ax.scatter(s["forget_Q_A_Prob"], s[a.y], s=140, marker="*", color=INK if kind == "retain" else SURFACE, edgecolors=INK, linewidths=1.1, zorder=6)
            ax.invert_xaxis()
            if j == 0:
                ax.set_ylabel(LABEL[a.y].split("  (")[0], fontsize=9, color=MUTED)
            if i == 2:
                ax.set_xlabel("forget_Q_A_Prob  (more forgotten →)", fontsize=9, color=MUTED)
    handles = [Line2D([], [], marker=MARKER[m], color=SURFACE, markerfacecolor=SURFACE if m == "ROCR-nb" else COLOR.get(m, GRAY),
                      markeredgecolor=COLOR[m] if m == "ROCR-nb" else SURFACE, markersize=9 if m in COLOR else 8, label=LEGEND.get(m, m)) for m in methods]
    handles += [Line2D([], [], marker="*", color=SURFACE, markerfacecolor=INK, markeredgecolor=INK, markersize=12, label="retain model"),
                Line2D([], [], marker="*", color=SURFACE, markerfacecolor=SURFACE, markeredgecolor=INK, markersize=12, label="full model")]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(f"{LABEL[a.y]} vs forgetting  (shaded: Prob <= {a.tau}, the matching region; each point = one run / epoch; NEAR-Dist = the final E8 run)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    stem = a.out or f"saves/analysis/pareto_grid_{a.y}"
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    fig.savefig(stem + ".png", dpi=160, facecolor=SURFACE)
    fig.savefig(stem + ".pdf", facecolor=SURFACE)
    print("written:", stem + ".png /.pdf")


if __name__ == "__main__":
    main()
