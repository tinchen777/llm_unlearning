
from __future__ import annotations
import sys
sys.path.append("/data/tianzhen/my_projects/LLM/llm_unlearning")  # noqa

import os
import logging

from src.vis import plots

logging.basicConfig(level=logging.INFO, format="[%(name)s][%(levelname)s] - %(message)s")
logger = logging.getLogger("vis")


RUN_DIRS = [
    "saves/finetune/test/tofu_phi-1_5_full",
    "saves/finetune/test/tofu_phi-1_5_retain90",
    "saves/finetune/test/tofu_phi-1_5_retain95",
    "saves/finetune/test/tofu_phi-1_5_retain99"
]
OUT_DIR = "saves/plots/test/phi-1_5"

made = []
for name, fn in [
    ("training_curves.png", lambda: plots.plot_training_curves(RUN_DIRS)),
    ("metric_trajectories.png", lambda: plots.plot_metric_trajectories(RUN_DIRS)),
    ("method_comparison.png", lambda: plots.plot_method_comparison(RUN_DIRS)),
    ("tradeoff.png", lambda: plots.plot_tradeoff(RUN_DIRS)),
]:
    try:
        plots.save_fig(fn(), os.path.join(OUT_DIR, name))
        made.append(name)
    except Exception as e:
        logger.warning(f"Skipping {name}: {e}")
if not made:
    raise SystemExit("report: nothing could be plotted from the given runs")
logger.info(f"Report done ({len(made)} charts) -> {OUT_DIR}")
