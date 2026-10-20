"""Ablation figures from the exported TOFU csv files (same inputs as tofu_from_csv.py).

    python plots/ablation_from_csv.py [--data plots/data] [--out plots/out]

  ablation_k_<MODEL>.pdf      NEAR: number of neighbours K (1 / 5 / 15); solid = eoi, dashed = all, dotted = Original model
  ablation_mode_<MODEL>.pdf   shift position (eoi vs all) for NEAR (K=5) and LUNAR, per forget set, next to Original / Retain
Metrics: forget-set ROUGE and probability (lower = more forgotten), model utility (higher = better).
Note: every run has its own selected layer (column `layer` in tofu_all.csv), so the ablations also differ in layer.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402

from tofu_from_csv import FORGETS, K_COLOR, LUNAR_COLOR  # noqa: E402
from vis.plotter import grouped_bar_panel, line_panel  # noqa: E402
from vis.style import INK_PRIMARY, OTHER_GRAY, apply_style, series_color  # noqa: E402
from tofu_from_csv import load  # noqa: E402

PANELS = [("forget_Q_A_ROUGE", "forget-set ROUGE (lower: more forgotten)"), ("forget_Q_A_Prob", "forget-set probability (lower: more forgotten)"),
          ("model_utility", "model utility (higher: better)")]
FORGET_COLOR = dict(zip(FORGETS, [series_color(0), series_color(2), series_color(3)]))


def ablation_k(df: pd.DataFrame, out: Path):
    near = df[df["method"] == "NEAR"]
    for model, dm in near.groupby("model"):
        orig = df[(df["model"] == model) & (df["method"] == "Original")].set_index("forget")
        fig, axes = plt.subplots(1, len(PANELS), figsize=(4.6 * len(PANELS), 3.8), squeeze=False)
        for ax, (metric, label) in zip(axes[0], PANELS):
            lines, hl = [], []
            for forget in FORGETS:
                for mode, ls, marker in (("eoi", "-", "o"), ("all", "--", "s")):
                    d = dm[(dm["forget"] == forget) & (dm["mode"] == mode)].sort_values("k")
                    if d.empty:
                        continue
                    lines.append({"x": d["k"].astype(int).astype(str), "y": d[metric], "label": f"{forget} {mode}", "color": FORGET_COLOR[forget],
                                  "linestyle": ls, "marker": marker})
                if forget in orig.index:
                    hl.append((float(orig.loc[forget, metric]), FORGET_COLOR[forget]))
            line_panel(ax, lines, "number of neighbours K", label, "", hlines=hl)
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=6, fontsize=7, frameon=False)
        fig.suptitle(f"NEAR: effect of K - {model}  (dotted: Original model)")
        fig.tight_layout(rect=(0, 0.08, 1, 0.94))
        fig.savefig(out / f"ablation_k_{model}.pdf")
        plt.close(fig)


def ablation_mode(df: pd.DataFrame, out: Path, near_k: int = 5):
    series_def = [("Original", OTHER_GRAY, lambda d: d["method"] == "Original"),
                  ("NEAR eoi", K_COLOR[1], lambda d: (d["method"] == "NEAR") & (d["mode"] == "eoi") & (d["k"] == near_k)),
                  ("NEAR all", K_COLOR[5], lambda d: (d["method"] == "NEAR") & (d["mode"] == "all") & (d["k"] == near_k)),
                  ("LUNAR eoi", LUNAR_COLOR, lambda d: (d["method"] == "LUNAR") & (d["mode"] == "eoi")),
                  ("LUNAR all", series_color(7), lambda d: (d["method"] == "LUNAR") & (d["mode"] == "all")),
                  ("Retain", INK_PRIMARY, lambda d: d["method"] == "Retain")]
    for model, dm in df.groupby("model"):
        if not ((dm["method"] == "NEAR") | (dm["method"] == "LUNAR")).any():
            continue
        fig, axes = plt.subplots(1, len(PANELS), figsize=(4.8 * len(PANELS), 3.8), squeeze=False)
        for ax, (metric, label) in zip(axes[0], PANELS):
            series = []
            for name, color, sel in series_def:
                sub = dm[sel(dm)].set_index("forget")[metric]
                if sub.empty:
                    continue
                series.append({"label": name, "color": color, "values": [sub.get(f, float("nan")) for f in FORGETS]})
            grouped_bar_panel(ax, FORGETS, series, label, "", value_labels=False)
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=7.5, frameon=False)
        fig.suptitle(f"Shift position eoi vs all (NEAR K={near_k}, LUNAR) - {model}")
        fig.tight_layout(rect=(0, 0.08, 1, 0.94))
        fig.savefig(out / f"ablation_mode_{model}.pdf")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(Path(__file__).parent / "data"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "out"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    apply_style()
    df = load(Path(a.data))
    ablation_k(df, out)
    ablation_mode(df, out)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
