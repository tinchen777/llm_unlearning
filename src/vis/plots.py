"""Plot functions for experiment outputs.

Every function takes one or more run directories, draws with the shared style
(fixed-order colorblind-safe palette, recessive chrome, one y-axis per panel)
and returns the matplotlib Figure so callers can save or further tweak it.
"""

from __future__ import annotations
import os
import math
import matplotlib
import matplotlib.pyplot as plt
import logging
from typing import List, Optional, Sequence, Dict, Any, Tuple

from .style import apply_style, series_color, legend_if_multi, INK_SECONDARY
from .loaders import ExperimentLoader

matplotlib.use("Agg")

logger = logging.getLogger("vis")

# Metrics that live on (0, 1] but span orders of magnitude (p-values)
LOG_SCALE_METRICS = {"forget_quality"}


def _grid_axes(
    n_panels: int,
    max_ncols: int = 3,
    panel_size: Tuple[float, float] = (3.6, 2.6)
):
    ncols = min(max_ncols, max(n_panels, 1))
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        squeeze=False,
    )
    flat = axes.ravel()
    for ax in flat[n_panels:]:
        ax.set_visible(False)
    return fig, flat[:n_panels]


class Ploter:
    def __init__(self, run_dirs: List[str]):
        self.runs = [ExperimentLoader(run) for run in run_dirs]
        apply_style()

    def plot_training_curves(
        self,
        keys: Optional[Sequence[str]] = None,
        max_ncols: int = 3
    ):
        """Loss curves from `trainer_state.json` log_history, one panel per key,
        one line per run. `keys=None` auto-discovers every logged loss key
        (total `loss` plus custom per-component losses like `forget_pull_loss`)."""
        # # load all log_history entries of trainer_state.json` for each run
        # run_histories: Dict[str, Any] = {}
        # for run in run_dirs:
        #     state = load_trainer_state(run)
        #     if state:
        #         run_histories[run] = state.get("log_history", [])

        # if keys is None:
        #     discovered: List[str] = []
        #     eval_metric_keys = set()
        #     for run, run_history in run_histories.items():
        #         for step_metrics in load_checkpoint_summaries(run).values():
        #             eval_metric_keys.update(step_metrics)
        #         for k in list_numeric_keys(run_history):
        #             if k not in discovered and k not in _TRAIN_SKIP_KEYS and k not in eval_metric_keys:
        #                 discovered.append(k)
        #     keys = discovered
        # if not keys:
        #     raise ValueError("No plottable training keys found in trainer_state.json")
        # plot
        keys = 
        
        
        
        fig, axes = _grid_axes(len(keys), max_ncols)
        for ax, key in zip(axes, keys):
            n = 0
            for i, (run, run_history) in enumerate(run_histories.items()):
                steps, values = collect_series(run_history, key)
                if not steps:
                    continue
                ax.plot(steps, values, color=series_color(i), label=run_label(run))
                n += 1
            ax.set_title(key)
            ax.set_xlabel("step")
            legend_if_multi(ax, n)
        fig.tight_layout()
        return fig


    def plot_metric_trajectories(
        self,
        metrics: Optional[Sequence[str]] = None,
        ncols: int = 3,
    ):
        """Eval metric trajectories over training, from every
        `checkpoint-*/evals/*_SUMMARY.json`. One panel per metric, one line per
        run (markers mark the actual eval checkpoints)."""
        all_summaries = {run: load_checkpoint_summaries(run) for run in run_dirs}

        if metrics is None:
            metrics = []
            for summaries in all_summaries.values():
                for step_metrics in summaries.values():
                    for m in step_metrics:
                        if m not in metrics:
                            metrics.append(m)
        if not metrics:
            raise ValueError("No eval summaries found under the given runs")

        fig, axes = _grid_axes(len(metrics), ncols)
        for ax, metric in zip(axes, metrics):
            n = 0
            for i, (run, summaries) in enumerate(all_summaries.items()):
                pts = [
                    (step, vals[metric])
                    for step, vals in summaries.items()
                    if step >= 0 and isinstance(vals.get(metric), (int, float))
                ]
                if not pts:
                    continue
                xs, ys = zip(*sorted(pts))
                ax.plot(xs, ys, marker="o", color=series_color(i), label=run_label(run))
                n += 1
            ax.set_title(metric)
            ax.set_xlabel("step")
            if metric in LOG_SCALE_METRICS:
                ax.set_yscale("log")
            legend_if_multi(ax, n)
        fig.tight_layout()
        return fig


    def plot_method_comparison(
        self,
        metrics: Optional[Sequence[str]] = None,
        ncols: int = 3,
    ):
        """Final (last-eval) metric values compared across runs, one bar panel
        per metric. Bars keep each run's fixed series color."""
        finals = {run: load_final_summary(run) for run in run_dirs}

        if metrics is None:
            metrics = []
            for summary in finals.values():
                for m in summary:
                    if m not in metrics and isinstance(summary[m], (int, float)):
                        metrics.append(m)
        if not metrics:
            raise ValueError("No final summaries found under the given runs")

        labels = [run_label(r) for r in run_dirs]
        fig, axes = _grid_axes(len(metrics), ncols)
        for ax, metric in zip(axes, metrics):
            values, colors, xticklabels = [], [], []
            for i, run in enumerate(run_dirs):
                val = finals[run].get(metric)
                if isinstance(val, (int, float)):
                    values.append(val)
                    colors.append(series_color(i))
                    xticklabels.append(labels[i])
            xs = range(len(values))
            ax.bar(xs, values, width=0.62, color=colors)
            ax.set_xticks(list(xs))
            ax.set_xticklabels(xticklabels, rotation=20, ha="right", fontsize=7)
            # direct value labels (bars are few; text wears ink, not series color)
            for x, v in zip(xs, values):
                ax.annotate(
                    f"{v:.3g}", (x, v), ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=7, color=INK_SECONDARY,
                )
            ax.set_title(metric)
            if metric in LOG_SCALE_METRICS and all(v > 0 for v in values):
                ax.set_yscale("log")
            ax.grid(axis="x", visible=False)
        fig.tight_layout()
        return fig


    def plot_tradeoff(
        self,
        x_metric: str = "model_utility",
        y_metric: str = "forget_quality",
    ):
        """Forget/utility trade-off scatter: one labeled point per run, from each
        run's final summary. The usual reading: up and to the right is better."""
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        drawn = 0
        for i, run in enumerate(run_dirs):
            summary = load_final_summary(run)
            x, y = summary.get(x_metric), summary.get(y_metric)
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                logger.warning(f"Run {run} misses {x_metric}/{y_metric}, skipped")
                continue
            ax.scatter([x], [y], s=64, color=series_color(i), zorder=3)
            ax.annotate(
                run_label(run), (x, y), xytext=(6, 4), textcoords="offset points",
                fontsize=8, color=INK_SECONDARY,
            )
            drawn += 1
        if not drawn:
            raise ValueError(f"No run provides both `{x_metric}` and `{y_metric}`")
        ax.set_xlabel(x_metric)
        ax.set_ylabel(y_metric)
        ax.set_title(f"{y_metric} vs {x_metric}")
        if y_metric in LOG_SCALE_METRICS:
            ax.set_yscale("log")
        fig.tight_layout()
        return fig


    def plot_stat_distribution(
        self,
        metric: str,
        stat: Optional[str] = None,
        step: Optional[int] = None,
        bins: int = 30,
    ):
        """Histogram of a metric's per-sample values (`value_by_index`) from
        `*_EVAL.json`, overlaid across runs (step outlines keep overlaps legible).
        `run_dirs` entries may be run directories or direct *_EVAL.json paths."""
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        n = 0
        for i, run in enumerate(run_dirs):
            details = load_eval_details(run, step=step)
            if not details:
                continue
            values = extract_stat_values(details, metric, stat)
            if not values:
                logger.warning(f"No `{stat or 'numeric'}` values for `{metric}` in {run}")
                continue
            ax.hist(
                values, bins=bins, histtype="step", linewidth=2.0,
                color=series_color(i), label=run_label(run),
            )
            n += 1
        if n == 0:
            raise ValueError(f"No per-sample values found for metric `{metric}`")
        ax.set_title(f"{metric}" + (f" [{stat}]" if stat else ""))
        ax.set_xlabel(stat or "value")
        ax.set_ylabel("count")
        legend_if_multi(ax, n)
        fig.tight_layout()
        return fig


    def save_fig(self, fig, out_path: str):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
        logger.info(f"Saved plot: {out_path}")
