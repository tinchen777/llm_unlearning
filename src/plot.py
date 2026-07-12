"""Entry point to plot saved experiment outputs as charts.

Reads what training/eval runs already wrote to disk (`trainer_state.json`,
`checkpoint-*/evals/*_{EVAL,SUMMARY}.json`) -- no model or GPU needed.

Usage examples
--------------
# everything at once, into <first-run>/plots (or --out DIR):
python src/plot.py report saves/unlearn/NCU saves/unlearn/GradDiff

# individual charts:
python src/plot.py train saves/unlearn/NCU --keys loss forget_pull_loss
python src/plot.py evals saves/unlearn/NCU saves/unlearn/GradDiff --metrics forget_quality model_utility
python src/plot.py compare saves/unlearn/* --metrics forget_quality model_utility
python src/plot.py tradeoff saves/unlearn/* --x model_utility --y forget_quality
python src/plot.py dist saves/unlearn/NCU --metric forget_Q_A_Prob --stat prob
"""

from __future__ import annotations
import os
import argparse
import logging

from vis import plots

logging.basicConfig(level=logging.INFO, format="[%(name)s][%(levelname)s] - %(message)s")
logger = logging.getLogger("vis")


def _add_common(p: argparse.ArgumentParser):
    p.add_argument("runs", nargs="+", help="run directories (saves/{mode}/{task_name})")
    p.add_argument("--out", "-o", default=None,
                   help="output directory for PNGs (default: <first-run>/plots)")


def _out_dir(args) -> str:
    return args.out or os.path.join(args.runs[0], "plots")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train", help="training loss curves from trainer_state.json")
    _add_common(p)
    p.add_argument("--keys", nargs="*", default=None,
                   help="log_history keys to plot (default: auto-discover)")

    p = sub.add_parser("evals", help="eval metric trajectories over checkpoints")
    _add_common(p)
    p.add_argument("--metrics", nargs="*", default=None)

    p = sub.add_parser("compare", help="final metrics compared across runs (bars)")
    _add_common(p)
    p.add_argument("--metrics", nargs="*", default=None)

    p = sub.add_parser("tradeoff", help="scatter of two final metrics across runs")
    _add_common(p)
    p.add_argument("--x", default="model_utility")
    p.add_argument("--y", default="forget_quality")

    p = sub.add_parser("dist", help="per-sample value distribution from *_EVAL.json")
    _add_common(p)
    p.add_argument("--metric", required=True, help="metric name, e.g. forget_Q_A_Prob")
    p.add_argument("--stat", default=None, help="stat key, e.g. prob/score (default: auto)")
    p.add_argument("--step", type=int, default=None, help="checkpoint step (default: last)")
    p.add_argument("--bins", type=int, default=30)

    p = sub.add_parser("report", help="generate all applicable charts at once")
    _add_common(p)

    args = parser.parse_args()
    out_dir = _out_dir(args)

    if args.cmd == "train":
        plots.save_fig(plots.plot_training_curves(args.runs, keys=args.keys),
                       os.path.join(out_dir, "training_curves.png"))
    elif args.cmd == "evals":
        plots.save_fig(plots.plot_metric_trajectories(args.runs, metrics=args.metrics),
                       os.path.join(out_dir, "metric_trajectories.png"))
    elif args.cmd == "compare":
        plots.save_fig(plots.plot_method_comparison(args.runs, metrics=args.metrics),
                       os.path.join(out_dir, "method_comparison.png"))
    elif args.cmd == "tradeoff":
        plots.save_fig(plots.plot_tradeoff(args.runs, x_metric=args.x, y_metric=args.y),
                       os.path.join(out_dir, "tradeoff.png"))
    elif args.cmd == "dist":
        plots.save_fig(
            plots.plot_stat_distribution(args.runs, metric=args.metric, stat=args.stat,
                                         step=args.step, bins=args.bins),
            os.path.join(out_dir, f"dist_{args.metric}.png"))
    elif args.cmd == "report":
        made = []
        for name, fn in [
            ("training_curves.png", lambda: plots.plot_training_curves(args.runs)),
            ("metric_trajectories.png", lambda: plots.plot_metric_trajectories(args.runs)),
            ("method_comparison.png", lambda: plots.plot_method_comparison(args.runs)),
            ("tradeoff.png", lambda: plots.plot_tradeoff(args.runs)),
        ]:
            try:
                plots.save_fig(fn(), os.path.join(out_dir, name))
                made.append(name)
            except Exception as e:
                logger.warning(f"Skipping {name}: {e}")
        if not made:
            raise SystemExit("report: nothing could be plotted from the given runs")
        logger.info(f"Report done ({len(made)} charts) -> {out_dir}")


if __name__ == "__main__":
    main()
