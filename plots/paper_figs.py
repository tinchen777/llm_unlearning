"""Figures and tables of the experiment section (A - E), drawn from the saves of the NEAR / LUNAR / baseline runs.

Flow (the same for every figure):   vis.loader (extract)  ->  tidy csv in <out>/data/  ->  vis.plotter panels  ->  png / pdf
    A  TOFU forgetting vs utility       (+ table 1)    saves/eval/**/TOFU_SUMMARY.json, saves/unlearn/tofu_*/<method>/
    B  response behaviour per split     refusal / hallucination / ROUGE   <NEAR|LUNAR task>/<setting>/l*/w_down_summary.json
    C  layer selection curves           NEAR unseen_gap, LUNAR s1 - s2    <NEAR|LUNAR task>/<setting>/layer_selection.json
    D  K saturation of r_UV             cos(r_UV[K], r_UV[Kmax]), ||r_UV||   <NEAR task>/r_uv_summary.json
    E  activation separability          AUC(forget | neighbor / holdout) per layer, Original / Retain / NEAR / LUNAR (probe)

    python plots/paper_figs.py                         # every figure for every model / forget set found under saves/
    python plots/paper_figs.py --figs A B --models Llama-3.2-1B-Instruct --forgets forget05
    python plots/paper_figs.py --saves saves --out saves/plots/paper --near-k 5

A figure whose data is missing is skipped with a warning (the paths it looked for are logged), so add runs and re-run.
Add / delete a figure: write `extract_x(ctx) -> DataFrame` + `draw_x(ctx, df)` and register it in FIGURES at the end.
The conventions of the folder names live in the CONVENTIONS block below: edit it if you rename the outputs.
"""

from __future__ import annotations
import argparse
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib.pyplot as plt  # noqa: E402

from vis.loader import ExperimentLoader, MethodRunLoader, ProbeRunLoader, match_dirs  # noqa: E402
from vis.plotter import grouped_bar_panel, line_panel, scatter_panel  # noqa: E402
from vis.style import CATEGORICAL, INK_PRIMARY, OTHER_GRAY, apply_style, series_color  # noqa: E402

logger = logging.getLogger("plots")

# =============================================================================================
# CONVENTIONS: where the runs are (relative to --saves) and how their folders are named
# =============================================================================================
FORGET_TO_RETAIN = {"forget01": "retain99", "forget05": "retain95", "forget10": "retain90"}
# NEAR / LUNAR task dirs (scripts/*/_common.sh TASK_NAME):  saves/NEAR/near_tofu_<MODEL>_<FORGET>/
NEAR_TASK = r"near_tofu_(?P<model>.+)_(?P<forget>forget\d+)"
#                                                         saves/LUNAR/lunar_tofu_<MODEL>_<FORGET>_<REF>/
LUNAR_TASK = r"lunar_tofu_(?P<model>.+)_(?P<forget>forget\d+)_(?P<ref>LUNAR_[A-Za-z]+)"
# TOFU evaluations of the edited models (scripts/*/4_eval.sh task_name, mode `eval` -> saves/eval/<task_name>)
NEAR_EVAL = (r"eval_near_tofu_(?P<model>.+)_(?P<forget>forget\d+)_"
             r"k(?P<k>\d+)_l(?P<layers>[0-9-]+)_c(?P<coeff>[0-9.eE+-]+?)_(?P<mode>eoi|all)(?P<cf>_cf)?")
LUNAR_EVAL = (r"eval_lunar_tofu_(?P<model>.+)_(?P<forget>forget\d+)_(?P<ref>LUNAR_[A-Za-z]+)_"
              r"c(?P<coeff>[0-9.eE+-]+?)_l(?P<layers>[0-9-]+)_(?P<mode>eoi|all)(?P<cf>_cf)?")
# baselines (scripts/unlearn): saves/unlearn/tofu_<MODEL>_<FORGET>/<METHOD>[_lr.._ep..]/
UNLEARN_TASK = r"tofu_(?P<model>.+)_(?P<forget>forget\d+)"
# probes (scripts/NEAR/0_probe.sh, 4_eval.sh): the first existing candidate is used ({model}, {forget}, {retain}, {k}, {mode})
PROBE_ORIGINAL = ["NEAR/0_probe/tofu_{model}_full/{forget}", "NEAR/0_probe/tofu_{model}_full_{forget}", "NEAR/1_probe/tofu_{model}_full_{forget}"]
PROBE_RETAIN = ["NEAR/0_probe/tofu_{model}_{retain}", "NEAR/0_probe/tofu_{model}_{retain}/{forget}", "NEAR/1_probe/tofu_{model}_{retain}_{forget}"]
PROBE_NEAR = ["NEAR/probe/probe_near_tofu_{model}_{forget}_k{k}_l*_c*_{mode}"]
PROBE_LUNAR = ["NEAR/probe/probe_lunar_tofu_{model}_{forget}_*_{mode}", "LUNAR/probe/probe_lunar_tofu_{model}_{forget}_*_{mode}"]
# the reference models of TOFU (official logs, scripts: python setup/setup_data.py --eval_logs)
EVAL_ORIGINAL = ["eval/tofu_{model}_full/evals_{forget}", "eval/tofu_{model}_full"]
EVAL_RETAIN = ["eval/tofu_{model}_{retain}"]

TABLE_METRICS = ["forget_quality", "model_utility", "forget_Q_A_ROUGE", "forget_Q_A_Prob", "forget_truth_ratio", "privleak"]
SPLITS = ["forget", "neighbor", "retain", "holdout"]
PROBE_PAIRS = ["forget | neighbor", "forget | holdout"]

# colours: fixed slot per method (vis.style palette)
METHOD_COLORS = {
    "NEAR": series_color(0), "LUNAR": series_color(5), "Original": OTHER_GRAY, "Retain": INK_PRIMARY,
    "GradAscent": series_color(1), "GradDiff": series_color(2), "NPO": series_color(3), "SimNPO": series_color(4),
    "IdkDPO": series_color(6), "RMU": series_color(7),
}


@dataclass
class Context:
    saves: Path
    out: Path
    models: Optional[Sequence[str]] = None
    forgets: Optional[Sequence[str]] = None
    near_k: int = 5  # the K of the NEAR setting shown in B, C, E (D shows every K)
    lunar_ref: str = "LUNAR_harmful"
    skipped: List[str] = field(default_factory=list)

    def wanted(self, model: str, forget: str) -> bool:
        return (self.models is None or model in self.models) and (self.forgets is None or forget in self.forgets)

    def path(self, template: str, **kw) -> Optional[Path]:
        """The first existing directory of the glob `template` (relative to saves) after filling the placeholders."""
        pattern = template.format(**kw)
        hits = sorted(self.saves.glob(pattern))
        hits = [p for p in hits if p.is_dir()]
        return hits[0] if hits else None

    def first(self, templates: Sequence[str], **kw) -> Optional[Path]:
        for template in templates:
            found = self.path(template, **kw)
            if found is not None:
                return found
        return None

    def write(self, name: str, df: pd.DataFrame) -> Path:
        path = self.out / "data" / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"wrote {path} ({len(df)} rows)")
        return path

    def save_fig(self, fig, name: str):
        self.out.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(self.out / f"{name}.{ext}", dpi=200)
        plt.close(fig)
        logger.info(f"saved {self.out / name}.png / .pdf")


# =============================================================================================
# discovery
# =============================================================================================
def near_tasks(ctx: Context) -> Dict[Tuple[str, str], MethodRunLoader]:
    return {(d["model"], d["forget"]): MethodRunLoader(p) for d, p in match_dirs(ctx.saves / "NEAR", NEAR_TASK)
            if ctx.wanted(d["model"], d["forget"])}


def lunar_tasks(ctx: Context) -> Dict[Tuple[str, str], MethodRunLoader]:
    return {(d["model"], d["forget"]): MethodRunLoader(p) for d, p in match_dirs(ctx.saves / "LUNAR", LUNAR_TASK)
            if d["ref"] == ctx.lunar_ref and ctx.wanted(d["model"], d["forget"])}


def models_of(*groups) -> List[str]:
    return sorted({key[0] for g in groups for key in g})


def forgets_of(*groups) -> List[str]:
    return sorted({key[1] for g in groups for key in g})


def selected_run(task: MethodRunLoader, setting: str) -> Optional[Dict[str, Any]]:
    """The AdamW run of the setting that trained the layer chosen by select_layer.py (else any run of the setting)."""
    selection = task.selection(setting)
    runs = [r for r in task.runs if r["setting"] == setting]
    if selection:
        exact = [r for r in runs if r["layers"] == [int(selection["selected_layer"])] and r["solver"] == "adam"]
        runs = exact or runs
    return runs[0] if runs else None


def warn_skip(ctx: Context, what: str):
    ctx.skipped.append(what)
    logger.warning(f"skipped: {what}")


# =============================================================================================
# A  TOFU forgetting vs utility  (+ table 1)
# =============================================================================================
def _final_tofu(path: Optional[Path]) -> Optional[Dict[str, float]]:
    if path is None:
        return None
    try:
        summary = ExperimentLoader(path).eval_final_summaries.get("TOFU")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"cannot read {path}: {e}")
        return None
    return {k: v for k, v in summary.items() if isinstance(v, (int, float))} if summary else None


def extract_tofu(ctx: Context) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def add(model, forget, method, setting, path, **extra):
        metrics = _final_tofu(path)
        if metrics is not None:
            rows.append({"model": model, "forget": forget, "method": method, "setting": setting, "source": str(path),
                         **extra, **{m: metrics.get(m) for m in TABLE_METRICS}})

    for d, path in match_dirs(ctx.saves / "eval" / "NEAR", NEAR_EVAL):
        if ctx.wanted(d["model"], d["forget"]):
            add(d["model"], d["forget"], "NEAR", f"k{d['k']}_{d['mode']}{d['cf'] or ''}", path, k=int(d["k"]), mode=d["mode"])
    for d, path in match_dirs(ctx.saves / "eval" / "LUNAR", LUNAR_EVAL):
        if d["ref"] == ctx.lunar_ref and ctx.wanted(d["model"], d["forget"]):
            add(d["model"], d["forget"], "LUNAR", f"{d['mode']}{d['cf'] or ''}", path, mode=d["mode"])
    for d, task in match_dirs(ctx.saves / "unlearn", UNLEARN_TASK):
        if ctx.wanted(d["model"], d["forget"]):
            for _, run in match_dirs(task, r".+"):
                add(d["model"], d["forget"], re.split(r"_lr|_ep", run.name)[0], run.name, run)
    models = {r["model"] for r in rows}
    for model in sorted(models):
        for forget in sorted({r["forget"] for r in rows if r["model"] == model}):
            retain = FORGET_TO_RETAIN.get(forget)
            add(model, forget, "Original", "", ctx.first(EVAL_ORIGINAL, model=model, forget=forget))
            add(model, forget, "Retain", "", ctx.first(EVAL_RETAIN, model=model, retain=retain))
    return pd.DataFrame(rows)


def _color(method: str) -> str:
    return METHOD_COLORS.get(method, series_color(len(METHOD_COLORS) % len(CATEGORICAL)))


def draw_tofu(ctx: Context, df: pd.DataFrame):
    markers = {"eoi": "o", "all": "s"}
    for model, dm in df.groupby("model"):
        forgets = sorted(dm["forget"].unique())
        fig, axes = plt.subplots(1, len(forgets), figsize=(5.2 * len(forgets), 4.4), squeeze=False)
        for ax, forget in zip(axes[0], forgets):
            points = []
            for _, r in dm[dm["forget"] == forget].iterrows():
                if pd.isna(r["model_utility"]) or pd.isna(r["forget_quality"]):
                    continue
                marker = "*" if r["method"] == "Original" else "D" if r["method"] == "Retain" else markers.get(str(r.get("mode")), "o")
                points.append({"x": r["model_utility"], "y": max(float(r["forget_quality"]), 1e-12), "label": r["method"],
                               "color": _color(r["method"]), "marker": marker, "size": 120 if marker in "*D" else 46})
            scatter_panel(ax, points, "model utility", "forget quality (p-value)", forget, yscale="log")
            ax.legend(fontsize=7, ncol=2)
        fig.suptitle(f"TOFU forgetting vs utility - {model}  (marker: o = eoi, s = all; up and to the right is better)")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        ctx.save_fig(fig, f"figA_tofu_tradeoff_{model}")


def table_tofu(ctx: Context, df: pd.DataFrame):
    """Table 1: one row per (model, forget set, method / setting); csv + markdown."""
    cols = ["model", "forget", "method", "setting"] + [m for m in TABLE_METRICS if m in df]
    table = df[cols].sort_values(["model", "forget", "method", "setting"]).reset_index(drop=True)
    ctx.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(ctx.out / "table1_tofu.csv", index=False)
    fmt = lambda v: "" if pd.isna(v) else (f"{v:.3g}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in table.itertuples(index=False)]
    (ctx.out / "table1_tofu.md").write_text("\n".join(lines) + "\n")
    logger.info(f"saved {ctx.out / 'table1_tofu.csv'} / .md")


# =============================================================================================
# B  response behaviour per split
# =============================================================================================
BEHAVIOR_METRICS = [("refusal_rate", "refusal rate"), ("hallucination_rate", "hallucination rate"), ("rougeL_recall", "ROUGE-L recall")]


def extract_behavior(ctx: Context) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    near, lunar = near_tasks(ctx), lunar_tasks(ctx)
    for method, tasks in (("NEAR", near), ("LUNAR", lunar)):
        for (model, forget), task in tasks.items():
            for setting, s in task.settings.items():
                if method == "NEAR" and s["k"] != ctx.near_k:
                    continue
                run = selected_run(task, setting)
                if run is None:
                    continue
                df = task.behavior_df
                df = df[(df["setting"] == setting) & (df["run"] == run["run"])].copy()
                df["model"], df["forget"], df["method"] = model, forget, method
                rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["series"] = [f"{m}-{mode}" if ph == "after" else "Original" for m, mode, ph in zip(out["method"], out["mode"], out["phase"])]
    # the ORIGINAL model is the same in every run of a (model, forget): keep one copy, from NEAR (whole forget set)
    original = out[(out["phase"] == "before")]
    original = original[original["method"] == ("NEAR" if (original["method"] == "NEAR").any() else "LUNAR")]
    original = original.drop_duplicates(["model", "forget", "split"])
    return pd.concat([original, out[out["phase"] == "after"]], ignore_index=True)


def draw_behavior(ctx: Context, df: pd.DataFrame):
    order = ["Original", "NEAR-eoi", "NEAR-all", "LUNAR-eoi", "LUNAR-all"]
    colors = {"Original": OTHER_GRAY, "NEAR-eoi": series_color(0), "NEAR-all": series_color(2),
              "LUNAR-eoi": series_color(5), "LUNAR-all": series_color(7)}
    for model, dm in df.groupby("model"):
        forgets = sorted(dm["forget"].unique())
        fig, axes = plt.subplots(len(forgets), len(BEHAVIOR_METRICS), figsize=(4.6 * len(BEHAVIOR_METRICS), 3.3 * len(forgets)), squeeze=False)
        for i, forget in enumerate(forgets):
            d = dm[dm["forget"] == forget]
            for j, (metric, label) in enumerate(BEHAVIOR_METRICS):
                series = []
                for name in order:
                    sub = d[d["series"] == name].set_index("split")
                    if sub.empty or metric not in sub:
                        continue
                    series.append({"label": name, "color": colors[name], "values": [sub[metric].get(s, float("nan")) for s in SPLITS]})
                grouped_bar_panel(axes[i][j], SPLITS, series, label, f"{forget}: {label}", value_labels=False)
            axes[i][0].legend(fontsize=7)
        fig.suptitle(f"Responses per split, before / after the edit - {model}  (NEAR: K={ctx.near_k}; LUNAR forgets the first 20 QAs only)")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        ctx.save_fig(fig, f"figB_behavior_{model}")


# =============================================================================================
# C  layer selection curves
# =============================================================================================
def extract_layer_curves(ctx: Context) -> pd.DataFrame:
    rows = []
    for method, tasks, column in (("NEAR", near_tasks(ctx), "unseen_gap"), ("LUNAR", lunar_tasks(ctx), "score")):
        for (model, forget), task in tasks.items():
            df = task.layer_selection_df
            if df.empty:
                continue
            if method == "NEAR":
                df = df[df["k"] == ctx.near_k]
            df = df.assign(model=model, forget=forget, method=method, value=df[column], metric=column)
            rows.append(df[["model", "forget", "method", "setting", "mode", "layer", "metric", "value", "selected"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def draw_layer_curves(ctx: Context, df: pd.DataFrame):
    titles = {"NEAR": "NEAR: unseen_gap (lower is better)", "LUNAR": "LUNAR: s1 - s2 (higher is better)"}
    for model, dm in df.groupby("model"):
        forgets = sorted(dm["forget"].unique())
        methods = [m for m in ("NEAR", "LUNAR") if (dm["method"] == m).any()]
        fig, axes = plt.subplots(len(methods), len(forgets), figsize=(4.8 * len(forgets), 3.4 * len(methods)), squeeze=False)
        for i, method in enumerate(methods):
            for j, forget in enumerate(forgets):
                d = dm[(dm["method"] == method) & (dm["forget"] == forget)]
                lines = []
                for mode, color in (("eoi", series_color(0)), ("all", series_color(2))):
                    sub = d[d["mode"] == mode].sort_values("layer")
                    if sub.empty:
                        continue
                    chosen = sub[sub["selected"]]
                    lines.append({"x": sub["layer"], "y": sub["value"], "label": mode, "color": color, "marker": "o",
                                  "highlight": int(chosen["layer"].iloc[0]) if len(chosen) else None})
                line_panel(axes[i][j], lines, "layer", "unseen_gap" if method == "NEAR" else "s1 - s2", f"{forget} - {titles[method]}")
                if lines:
                    axes[i][j].legend(fontsize=7)
        fig.suptitle(f"Layer selection - {model}  (star = selected layer)")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        ctx.save_fig(fig, f"figC_layer_selection_{model}")


# =============================================================================================
# D  K saturation of r_UV (NEAR)
# =============================================================================================
def extract_k_saturation(ctx: Context) -> pd.DataFrame:
    rows = []
    for (model, forget), task in near_tasks(ctx).items():
        df = task.r_uv_df
        if not df.empty:
            rows.append(df.assign(model=model, forget=forget))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def draw_k_saturation(ctx: Context, df: pd.DataFrame):
    for model, dm in df.groupby("model"):
        forgets = sorted(dm["forget"].unique())
        fig, axes = plt.subplots(2, len(forgets), figsize=(4.8 * len(forgets), 6.4), squeeze=False)
        ks = sorted(dm["k"].unique())
        for j, forget in enumerate(forgets):
            d = dm[dm["forget"] == forget]
            cos_lines, norm_lines = [], []
            for n, k in enumerate(ks):
                sub = d[d["k"] == k].sort_values("layer")
                norm_lines.append({"x": sub["layer"], "y": sub["rel_norm"], "label": f"K={k}", "color": series_color(n), "marker": "o"})
                if k != max(ks):
                    cos_lines.append({"x": sub["layer"], "y": sub["cos_to_kmax"], "label": f"K={k}", "color": series_color(n), "marker": "o"})
            line_panel(axes[0][j], cos_lines, "layer", "cos(r_UV[K], r_UV[Kmax])", f"{forget}: does K already point where K={max(ks)} points?", hlines=[(1.0, "0.7")])
            line_panel(axes[1][j], norm_lines, "layer", "||r_UV|| / ||forget activation||", f"{forget}: size of the shift")
            axes[0][j].legend(fontsize=7)
            axes[1][j].legend(fontsize=7)
        fig.suptitle(f"r_UV against the number of neighbours K - {model}")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        ctx.save_fig(fig, f"figD_k_saturation_{model}")


# =============================================================================================
# E  activation separability (probe AUC)
# =============================================================================================
def _auc(probe: ProbeRunLoader, pair: str) -> pd.Series:
    by_author = probe.auc(pair, by_author=True)
    return by_author if by_author.notna().any() else probe.auc(pair)  # by author needs >= 4 authors (forget05 / forget10)


def extract_probe_auc(ctx: Context) -> pd.DataFrame:
    rows = []
    keys = set(near_tasks(ctx)) | set(lunar_tasks(ctx))
    for model, forget in sorted(keys):
        retain = FORGET_TO_RETAIN.get(forget)
        sources = {
            "Original": ctx.first(PROBE_ORIGINAL, model=model, forget=forget),
            "Retain": ctx.first(PROBE_RETAIN, model=model, forget=forget, retain=retain),
        }
        for mode in ("eoi", "all"):
            sources[f"NEAR-{mode}"] = ctx.first(PROBE_NEAR, model=model, forget=forget, k=ctx.near_k, mode=mode)
            sources[f"LUNAR-{mode}"] = ctx.first(PROBE_LUNAR, model=model, forget=forget, mode=mode)
        for series, path in sources.items():
            if path is None or not (path / "activations_summary.json").exists():
                continue
            probe = ProbeRunLoader(path)
            for pair in PROBE_PAIRS:
                auc = _auc(probe, pair)
                rows += [{"model": model, "forget": forget, "series": series, "pair": pair, "layer": int(l), "auc": float(v),
                          "source": str(path)} for l, v in auc.items() if pd.notna(v)]
    return pd.DataFrame(rows)


def draw_probe_auc(ctx: Context, df: pd.DataFrame):
    colors = {"Original": OTHER_GRAY, "Retain": INK_PRIMARY, "NEAR-eoi": series_color(0), "NEAR-all": series_color(2),
              "LUNAR-eoi": series_color(5), "LUNAR-all": series_color(7)}
    for model, dm in df.groupby("model"):
        forgets = sorted(dm["forget"].unique())
        fig, axes = plt.subplots(len(PROBE_PAIRS), len(forgets), figsize=(4.8 * len(forgets), 3.5 * len(PROBE_PAIRS)), squeeze=False)
        for i, pair in enumerate(PROBE_PAIRS):
            for j, forget in enumerate(forgets):
                d = dm[(dm["forget"] == forget) & (dm["pair"] == pair)]
                lines = []
                for series in colors:
                    sub = d[d["series"] == series].sort_values("layer")
                    if not sub.empty:
                        lines.append({"x": sub["layer"], "y": sub["auc"], "label": series, "color": colors[series],
                                      "linestyle": "--" if series in ("Original", "Retain") else "-"})
                line_panel(axes[i][j], lines, "layer", f"AUC ({pair})", f"{forget}: {pair}", hlines=[(0.5, "0.75")])
                if lines:
                    axes[i][j].legend(fontsize=7)
        fig.suptitle(f"How well the question sets separate in the activations - {model}  (closer to Retain is better; 0.5 = indistinguishable)")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        ctx.save_fig(fig, f"figE_probe_auc_{model}")


# =============================================================================================
# registry
# =============================================================================================
FIGURES: Dict[str, Tuple[Callable[[Context], pd.DataFrame], Callable[[Context, pd.DataFrame], None], str]] = {
    "A": (extract_tofu, lambda ctx, df: (draw_tofu(ctx, df), table_tofu(ctx, df)), "figA_tofu_metrics"),
    "B": (extract_behavior, draw_behavior, "figB_behavior"),
    "C": (extract_layer_curves, draw_layer_curves, "figC_layer_selection"),
    "D": (extract_k_saturation, draw_k_saturation, "figD_k_saturation"),
    "E": (extract_probe_auc, draw_probe_auc, "figE_probe_auc"),
}


def run(ctx: Context, figs: Sequence[str]) -> List[str]:
    apply_style()
    made = []
    for name in figs:
        extract, draw, csv_name = FIGURES[name]
        df = extract(ctx)
        if df is None or df.empty:
            warn_skip(ctx, f"figure {name}: no data found under {ctx.saves}")
            continue
        path = ctx.write(csv_name, df)
        draw(ctx, pd.read_csv(path))  # draw from the csv: what is plotted is what is on disk
        made.append(name)
    return made


def main(argv: Optional[Sequence[str]] = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saves", default="saves", help="root of the outputs (default: saves)")
    ap.add_argument("--out", default=None, help="where figures / tables / data csv go (default: <saves>/plots/paper)")
    ap.add_argument("--figs", nargs="+", default=list(FIGURES), choices=list(FIGURES))
    ap.add_argument("--models", nargs="+", default=None, help="base models, e.g. Llama-3.2-1B-Instruct (default: all found)")
    ap.add_argument("--forgets", nargs="+", default=None, help="forget1 / forget05 / forget10 (default: all found)")
    ap.add_argument("--near-k", type=int, default=5, help="the K of the NEAR setting shown in B, C, E")
    ap.add_argument("--lunar-ref", default="LUNAR_harmful", help="reference set of the LUNAR runs to show")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(name)s][%(levelname)s] - %(message)s")
    saves = Path(args.saves)
    ctx = Context(saves=saves, out=Path(args.out) if args.out else saves / "plots" / "paper", models=args.models,
                  forgets=args.forgets, near_k=args.near_k, lunar_ref=args.lunar_ref)
    made = run(ctx, args.figs)
    logger.info(f"done: {made}; skipped: {ctx.skipped}")
    return made


if __name__ == "__main__":
    main()
