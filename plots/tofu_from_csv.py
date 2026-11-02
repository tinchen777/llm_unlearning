"""TOFU figure + LaTeX tables from the three exported csv files (plots/data/{TOFU,NEAR_TOFU,LUNAR_TOFU}.csv).

    python plots/tofu_from_csv.py [--data plots/data] [--out plots/out]

Outputs: <out>/tofu_tradeoff_<MODEL>.pdf (forget quality / forget ROUGE vs model utility, one column per forget set),
<out>/table_tofu_<MODEL>.tex (booktabs), <out>/tofu_all.csv (the parsed rows).
Name conventions: `tofu_<M>_full/evals_<F>` (original), `tofu_<M>_retain<NN>` (retain model; forget01/05/10 <-> retain99/95/90),
`near_tofu_<M>_<F>_k<K>_l<L>_c<C>_<mode>`, `lunar_tofu_<M>_<F>_<REF>_c<C>_l<L>_<mode>`.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib.pyplot as plt  # noqa: E402

from vis.plotter import scatter_panel  # noqa: E402
from vis.style import INK_PRIMARY, OTHER_GRAY, apply_style, series_color  # noqa: E402

RETAIN_TO_FORGET = {"retain99": "forget01", "retain95": "forget05", "retain90": "forget10"}
NEAR = re.compile(r"near_tofu_(?P<model>.+)_(?P<forget>forget\d+)_k(?P<k>\d+)_l(?P<layer>\d+)_c(?P<c>[\d.]+)_(?P<mode>eoi|all)$")
LUNAR = re.compile(r"lunar_tofu_(?P<model>.+)_(?P<forget>forget\d+)_(?P<ref>LUNAR_\w+)_c(?P<c>[\d.]+)_l(?P<layer>\d+)_(?P<mode>eoi|all)$")
ORIG = re.compile(r"tofu_(?P<model>.+)_full/evals_(?P<forget>forget\d+)$")
RET = re.compile(r"tofu_(?P<model>.+)_(?P<retain>retain\d+)$")
METRICS = ["forget_quality", "model_utility", "forget_Q_A_ROUGE", "forget_Q_A_Prob", "forget_truth_ratio", "privleak"]
FORGETS = ["forget01", "forget05", "forget10"]
K_COLOR = {1: series_color(0), 5: series_color(2), 15: series_color(3)}
LUNAR_COLOR = series_color(5)


def load(data: Path) -> pd.DataFrame:
    rows = []
    for _, r in pd.read_csv(data / "TOFU.csv").iterrows():
        n = r["name"]
        if m := ORIG.match(n):
            rows.append({**m.groupdict(), "method": "Original", "variant": ""})
        elif m := RET.match(n):
            rows.append({"model": m["model"], "forget": RETAIN_TO_FORGET[m["retain"]], "method": "Retain", "variant": ""})
        else:
            continue
        rows[-1].update({k: r.get(k) for k in METRICS})
    for file, regex, method in (("NEAR_TOFU.csv", NEAR, "NEAR"), ("LUNAR_TOFU.csv", LUNAR, "LUNAR")):
        for _, r in pd.read_csv(data / file).iterrows():
            m = regex.match(r["name"])
            if m is None:
                print(f"skip {r['name']}")
                continue
            d = m.groupdict()
            variant = f"K={d['k']}, {d['mode']}" if method == "NEAR" else d["mode"]
            rows.append({"model": d["model"], "forget": d["forget"], "method": method, "variant": variant, "mode": d["mode"],
                         "k": int(d["k"]) if "k" in d else None, "layer": int(d["layer"]), **{k: r.get(k) for k in METRICS}})
    return pd.DataFrame(rows)


def style_of(r):
    if r["method"] == "Original":
        return OTHER_GRAY, "*", 150, "Original"
    if r["method"] == "Retain":
        return INK_PRIMARY, "D", 70, "Retain"
    marker = "o" if r["mode"] == "eoi" else "s"
    if r["method"] == "NEAR":
        return K_COLOR[int(r["k"])], marker, 55, f"NEAR K={int(r['k'])}"
    return LUNAR_COLOR, marker, 55, "LUNAR"


def draw(df: pd.DataFrame, out: Path):
    panels = [("forget_quality", "forget quality (p-value)", "log"), ("forget_Q_A_ROUGE", "forget-set ROUGE", None)]
    for model, dm in df.groupby("model"):
        fig, axes = plt.subplots(len(panels), len(FORGETS), figsize=(4.6 * len(FORGETS), 3.6 * len(panels)), squeeze=False)
        for j, forget in enumerate(FORGETS):
            d = dm[dm["forget"] == forget]
            for i, (metric, ylabel, scale) in enumerate(panels):
                pts = []
                for _, r in d.iterrows():
                    if pd.isna(r[metric]) or pd.isna(r["model_utility"]):
                        continue
                    color, marker, size, label = style_of(r)
                    pts.append({"x": r["model_utility"], "y": max(float(r[metric]), 1e-30) if scale else r[metric],
                                "label": label, "color": color, "marker": marker, "size": size})
                scatter_panel(axes[i][j], pts, "model utility", ylabel if j == 0 else "", forget if i == 0 else "", yscale=scale)
        h, l = axes[0][0].get_legend_handles_labels()
        h2, l2 = axes[1][0].get_legend_handles_labels()
        merged = dict(zip(l + l2, h + h2))
        fig.legend(merged.values(), merged.keys(), loc="lower center", ncol=len(merged), fontsize=8, frameon=False)
        fig.suptitle(f"{model}  (o = eoi, s = all; Retain has no forget quality)")
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        fig.savefig(out / f"tofu_tradeoff_{model}.pdf")
        plt.close(fig)


def table(df: pd.DataFrame, out: Path):
    cols = [("layer", "L"), ("forget_quality", "FQ$\\uparrow$"), ("model_utility", "MU$\\uparrow$"), ("forget_Q_A_ROUGE", "F-ROUGE$\\downarrow$"),
            ("forget_Q_A_Prob", "F-Prob$\\downarrow$"), ("privleak", "PrivLeak")]

    def fmt(c, v):
        if v is None or pd.isna(v):
            return "--"
        if c == "layer":
            return str(int(v))
        if c == "forget_quality":
            return f"{v:.1e}".replace("e-0", "e-").replace("e+0", "e+")
        return f"{v:.1f}" if c == "privleak" else f"{v:.3f}"

    order = {"Original": 0, "Retain": 1, "NEAR": 2, "LUNAR": 3}
    for model, dm in df.groupby("model"):
        lines = ["% generated by plots/tofu_from_csv.py", "\\begin{table*}[t]", "\\centering\\small",
                 f"\\caption{{TOFU results on {model.replace('_', '-')}. FQ: forget quality, MU: model utility.}}",
                 "\\begin{tabular}{ll" + "r" * len(cols) + "}", "\\toprule", "Forget set & Method & " + " & ".join(h for _, h in cols) + " \\\\"]
        for forget in FORGETS:
            d = dm[dm["forget"] == forget].copy()
            if d.empty:
                continue
            d["o"] = d["method"].map(order)
            d = d.sort_values(["o", "variant"], key=lambda s: s if s.name != "variant" else s.str.replace("K=", "").str.zfill(8))
            lines.append("\\midrule")
            for n, (_, r) in enumerate(d.iterrows()):
                name = r["method"] + (f" ({r['variant']})" if r["variant"] else "")
                lines.append(f"{forget if n == 0 else ''} & {name} & " + " & ".join(fmt(c, r.get(c)) for c, _ in cols) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", f"\\label{{tab:tofu-{model}}}", "\\end{table*}"]
        (out / f"table_tofu_{model}.tex").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(Path(__file__).parent / "data"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "out"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    apply_style()
    df = load(Path(a.data))
    df.to_csv(out / "tofu_all.csv", index=False)
    draw(df, out)
    table(df, out)
    print(f"{len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
