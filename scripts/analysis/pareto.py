"""Forgetting vs utility (and vs detectability) of NEAR, NEAR + KL, LUNAR and the baselines in ONE picture and ONE table, from what the evaluations
saved (TOFU_SUMMARY.json): no model is run.

    python scripts/analysis/pareto.py [--model Llama-3.2-1B-Instruct] [--forget forget01] [--tau 0.25] [--out saves/analysis/pareto]
        [--near saves/eval/NEAR/single_v2] [--lunar saves/eval/LUNAR/single] [--baselines saves/tofu_unlearn]

Points
- NEAR / NEAR+KL: every run folder of `--near` whose name has `_kl<w>` (w > 0) is NEAR+KL, the others are NEAR (kl0 = end-to-end training without the KL term).
- LUNAR: the run folders of `--lunar` of this forget split (the coefficient sweep).
- baselines: EVERY epoch checkpoint (step > 0) of every run of `--baselines` (a run `X` is replaced by its re-run `X_ud`, which has the probe / ud_*) (so a method with several learning rates / epochs is a cloud of points, not
  the one "best epoch" of baseline_table.py).
- references: the retain and the full model (saves/eval/{retain,full}/<model>/<forget>).

Axes. x = forget_Q_A_Prob, the probability of the true answers (the retain model has ~0.17: that is "forgotten like a model that never saw it";
the axis is INVERTED so that more forgetting is to the right and the best place is the top right). y = model_utility, retain ROUGE (the generation
quality on the retain set), `ud_wb_unsup` / `ud_bb` (detectability: only runs evaluated with the probe have them; the baselines do not yet).

Table (`--tau`, `--mia-tol`): per method, among the "retain-like" points (forget_Q_A_Prob <= tau, the default 0.25 being ~1.5 x the retain model, AND
mia_loss within `--mia-tol` of the retain model's, so over-forgetting late epochs of NPO / GradAscent do not count) the one with the best utility; a method
that never gets there shows the point closest to it. A method that forgets nothing is not a
fair comparison however high its utility, so read the utility only at a matched Prob. n=40 and one run per point: differences of a few 0.01 are noise.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

KEYS = ("forget_Q_A_Prob", "mia_loss", "extraction_strength", "model_utility", "retain_rougeL_recall", "ud_bb", "ud_wb_unsup", "ud_wb_sup")
BASELINES = ("GradAscent", "GradDiff", "NPO", "SimNPO", "IdkDPO", "RMU")


def read_summary(path):
    try:
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
    except (OSError, ValueError):
        return None
    out = {k: s.get(k) for k in KEYS}
    return out if out["forget_Q_A_Prob"] is not None and out["model_utility"] is not None else None


def collect(near, lunar, baselines, model, forget):
    """List of points `{family, method, run, <KEYS>}`; family is NEAR | NEAR+KL | LUNAR | baseline."""
    points = []
    for d in sorted(glob.glob(os.path.join(near, "*"))):
        s = read_summary(os.path.join(d, "TOFU_SUMMARY.json"))
        name = os.path.basename(d)
        if s is None or f"_k5_" not in name and "_k" not in name:
            continue
        kl = re.search(r"_kl(\d+(?:\.\d+)?)", name)
        family = "NEAR+KL" if kl and float(kl.group(1)) > 0 else "NEAR"
        points.append({"family": family, "method": family, "run": name.split("_k5_")[0], **s})
    for d in sorted(glob.glob(os.path.join(lunar, f"*_{forget}_*"))):
        s = read_summary(os.path.join(d, "TOFU_SUMMARY.json"))
        if s is not None:
            m = re.search(r"_c([\d.]+)_l", os.path.basename(d))
            points.append({"family": "LUNAR", "method": "LUNAR", "run": f"c{m.group(1)}" if m else os.path.basename(d), **s})
    for run in sorted(glob.glob(os.path.join(baselines, "*", f"tofu_{model}_{forget}*"))):
        if os.path.isdir(run + "_ud"):  # the re-run of this setting with the probe (TAG=_ud, with ud_*) replaces the TOFU-only run
            continue
        method = os.path.basename(os.path.dirname(run))
        suffix = os.path.basename(run).replace(f"tofu_{model}_{forget}", "") or "_default"
        for path in glob.glob(os.path.join(run, "checkpoint-*", "evals", "TOFU_SUMMARY.json")):
            step = int(re.search(r"checkpoint-(\d+)", path).group(1))
            s = read_summary(path)
            if step > 0 and s is not None:
                points.append({"family": "baseline", "method": method, "run": f"{suffix.lstrip('_')}@{step}", **s})
    return points


def references(model, forget):
    out = {}
    for kind in ("retain", "full"):
        s = read_summary(os.path.join("saves", "eval", kind, model, forget, "TOFU_SUMMARY.json"))
        if s is not None:
            out[kind] = s
    return out


def dominated(p, others, xkey="forget_Q_A_Prob", ykey="model_utility"):
    """p is dominated if another point forgets at least as much (lower Prob) AND has at least the utility, and is strictly better in one."""
    return any(o[xkey] <= p[xkey] and o[ykey] >= p[ykey] and (o[xkey] < p[xkey] or o[ykey] > p[ykey]) for o in others if o is not p)


def front(points, xkey="forget_Q_A_Prob", ykey="model_utility"):
    pts = [p for p in points if p.get(ykey) is not None]
    return sorted((p for p in pts if not dominated(p, pts, xkey, ykey)), key=lambda p: p[xkey])


def retain_like(p, tau, mia_centre, mia_tol):
    """Forgot at least as much as `tau` AND not far beyond the retain model: mia_loss within `mia_tol` of the retain model's (a model that
    over-forgets, e.g. NPO / GradAscent late epochs with mia ~0, is detectable in its own way and not the 'forget like retain' target)."""
    mia = p.get("mia_loss")
    return p["forget_Q_A_Prob"] <= tau and (mia is None or abs(mia - mia_centre) <= mia_tol)


def matched(points, tau, mia_centre=0.5, mia_tol=0.25):
    """Per method: the best-utility retain-like point (`reached` True), else the point with the lowest Prob (`reached` False)."""
    rows = []
    for method in dict.fromkeys(p["method"] for p in points):
        mine = [p for p in points if p["method"] == method]
        ok = [p for p in mine if retain_like(p, tau, mia_centre, mia_tol)]
        if ok:
            rows.append({**max(ok, key=lambda p: p["model_utility"]), "reached": True, "n": len(mine), "n_ok": len(ok)})
        else:  # the point that misses the retain-like region by the least (Prob above tau, mia outside the band)
            miss = lambda p: max(p["forget_Q_A_Prob"] - tau, 0.0) + (max(abs(p["mia_loss"] - mia_centre) - mia_tol, 0.0) if p.get("mia_loss") is not None else 0.0)
            rows.append({**min(mine, key=miss), "reached": False, "n": len(mine), "n_ok": 0})
    return rows


def fmt(v, digits=3):
    return "  -  " if v is None else f"{v:.{digits}f}"


def print_table(rows, refs, tau, mia_centre, mia_tol):
    print(f"Best utility among the retain-like points: forget_Q_A_Prob <= {tau} and |mia_loss - {mia_centre:.3f}| <= {mia_tol}  "
          "(* = the method never gets there: the point closest to it)")
    print(f"{'method':11s} {'run':34s} {'pts':>7s} | {'Prob':>5s} {'mia':>5s} {'extr':>5s} {'util':>5s} {'retR':>5s} {'ud_bb':>5s} {'wbU':>5s}")
    print("-" * 104)
    for kind, s in refs.items():
        print(f"{kind:11s} {'(reference)':34s} {'':>7s} | {fmt(s['forget_Q_A_Prob']):>5s} {fmt(s['mia_loss']):>5s} {fmt(s['extraction_strength']):>5s} "
              f"{fmt(s['model_utility']):>5s} {fmt(s['retain_rougeL_recall']):>5s} {fmt(s['ud_bb']):>5s} {fmt(s['ud_wb_unsup']):>5s}")
    for r in sorted(rows, key=lambda r: (not r["reached"], -r["model_utility"])):
        print(f"{r['method'] + ('' if r['reached'] else '*'):11s} {r['run'][:34]:34s} {r['n_ok']:>3d}/{r['n']:<3d} | {fmt(r['forget_Q_A_Prob']):>5s} "
              f"{fmt(r['mia_loss']):>5s} {fmt(r['extraction_strength']):>5s} {fmt(r['model_utility']):>5s} {fmt(r['retain_rougeL_recall']):>5s} "
              f"{fmt(r['ud_bb']):>5s} {fmt(r['ud_wb_unsup']):>5s}")


def write_csv(points, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["family", "method", "run", *KEYS])
        w.writeheader()
        w.writerows(points)


# colours: the first three slots of the reference palette (they validate under all-pairs, light surface); the baselines are neutral gray
# with one marker shape per method (secondary encoding); NEAR and NEAR+KL share the blue family: filled circle vs diamond.
STYLE = {
    "NEAR": dict(color="#2a78d6", marker="o"),
    "NEAR+KL": dict(color="#1baf7a", marker="D"),
    "LUNAR": dict(color="#eb6834", marker="s"),
}
BASE_MARKERS = dict(zip(BASELINES, ["v", "^", "<", ">", "P", "X"]))
GRAY, INK, MUTED, SURFACE, GRID = "#8a8984", "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e1"


def plot(points, refs, tau, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    panels = [("model_utility", "model_utility  (higher = better)", "Utility"),
              ("retain_rougeL_recall", "retain ROUGE-L recall  (generation quality on retain)", "Retain generation"),
              ("ud_wb_unsup", "ud_wb_unsup  (white-box detectability, lower = better)", "White-box detectability"),
              ("ud_bb", "ud_bb  (black-box detectability, lower = better)", "Black-box detectability")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), facecolor=SURFACE)
    for ax, (ykey, ylabel, title) in zip(axes.flat, panels):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("forget_Q_A_Prob of the true answers  (axis inverted: more forgotten →)", fontsize=9, color=MUTED)
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
        ax.axvspan(0, tau, color=GRID, alpha=0.55, linewidth=0, zorder=0)  # "forgot at least as much as tau" (the table also bounds mia_loss)
        for p in points:
            if p.get(ykey) is None:
                continue
            if p["family"] == "baseline":
                ax.scatter(p["forget_Q_A_Prob"], p[ykey], s=34, marker=BASE_MARKERS.get(p["method"], "o"), color=GRAY, alpha=0.8,
                           edgecolors=SURFACE, linewidths=0.8, zorder=3)
        for fam, st in STYLE.items():
            mine = [p for p in points if p["family"] == fam and p.get(ykey) is not None]
            # lines only where "higher is better" gives a frontier; detectability is shown as points (a line through them would mean nothing)
            fr = front(mine, ykey=ykey) if ykey in ("model_utility", "retain_rougeL_recall") else []
            if len(fr) > 1:
                ax.plot([p["forget_Q_A_Prob"] for p in fr], [p[ykey] for p in fr], color=st["color"], linewidth=1.6, alpha=0.9, zorder=4)
            ax.scatter([p["forget_Q_A_Prob"] for p in mine], [p[ykey] for p in mine], s=58, marker=st["marker"], color=st["color"],
                       edgecolors=SURFACE, linewidths=1.2, zorder=5)
        for kind, s in refs.items():
            if s.get(ykey) is not None:
                ax.scatter(s["forget_Q_A_Prob"], s[ykey], s=150, marker="*", color=INK if kind == "retain" else SURFACE, edgecolors=INK,
                           linewidths=1.2, zorder=6)
        ax.invert_xaxis()
    handles = [Line2D([], [], marker=st["marker"], color=SURFACE, markerfacecolor=st["color"], markersize=9, label=fam) for fam, st in STYLE.items()]
    present = [m for m in BASELINES if any(p["method"] == m for p in points)]
    handles += [Line2D([], [], marker=BASE_MARKERS[m], color=SURFACE, markerfacecolor=GRAY, markersize=8, label=m) for m in present]
    handles += [Line2D([], [], marker="*", color=SURFACE, markerfacecolor=INK, markeredgecolor=INK, markersize=12, label="retain model"),
                Line2D([], [], marker="*", color=SURFACE, markerfacecolor=SURFACE, markeredgecolor=INK, markersize=12, label="full model")]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(f"Forgetting vs utility and detectability  (shaded: forget Prob <= {tau}; lines: non-dominated points; the table also bounds mia_loss)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.99, 0.012, "forget01, 1B; one run per point, n=40", ha="right", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(out_png, dpi=160, facecolor=SURFACE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Llama-3.2-1B-Instruct")
    ap.add_argument("--forget", default="forget01")
    ap.add_argument("--near", default="saves/eval/NEAR/single_v2")
    ap.add_argument("--lunar", default="saves/eval/LUNAR/single")
    ap.add_argument("--baselines", default="saves/tofu_unlearn")
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--mia-tol", type=float, default=0.25, help="retain-like: |mia_loss - mia_loss of the retain model| <= this")
    ap.add_argument("--out", default=None, help="folder for pareto.png, points.csv, matched.md (default: only the table)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    points = collect(args.near, args.lunar, args.baselines, args.model, args.forget)
    if not points:
        sys.exit("no points found")
    refs = references(args.model, args.forget)
    centre = (refs.get("retain") or {}).get("mia_loss") or 0.5
    rows = matched(points, args.tau, centre, args.mia_tol)
    print_table(rows, refs, args.tau, centre, args.mia_tol)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        write_csv(points, os.path.join(args.out, "points.csv"))
        if not args.no_plot:
            plot(points, refs, args.tau, os.path.join(args.out, "pareto.png"))
            print(f"\nwritten: {args.out}/pareto.png, points.csv")


if __name__ == "__main__":
    main()
