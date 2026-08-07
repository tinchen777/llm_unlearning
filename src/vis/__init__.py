
from __future__ import annotations
from pathlib import Path
import logging
from typing import Sequence, Dict, Optional, Union, TYPE_CHECKING

from .plotter import Plotter

if TYPE_CHECKING:
    from utils.config import TrackingConfig
    from matplotlib.figure import Figure
    from os import PathLike

logger = logging.getLogger("vis")

_DEFAULT_OUT_DIR = object()

PLOT_REGISTRY: Sequence[str] = [
    "training_curves",
    "metric_trajectories",
    "method_comparison",
    "tradeoff",
]


def plot_figures(
    run_dirs: Sequence[Union[PathLike, str]],
    vis_cfg: TrackingConfig,
    out_dir: Optional[Union[PathLike, str]] = _DEFAULT_OUT_DIR  # type: ignore
):
    # create plotter
    plotter = Plotter(*run_dirs)
    # determine output directory
    if out_dir is not None:
        # with save
        _out_dir = plotter.default_out_dir if out_dir is _DEFAULT_OUT_DIR else Path(out_dir)
        _out_dir.mkdir(parents=True, exist_ok=True)
    else:
        # without save
        _out_dir = None
    # save args
    save_args = vis_cfg.get("save_args", {}, check_none=True)
    # plot each figure
    plot_figs: Dict[str, Dict[str, Figure]] = {}
    for plot_name, plot_cfg in vis_cfg["plots"].items():
        if plot_name not in PLOT_REGISTRY:
            logger.warning(f"Unknown plot name: {plot_name}, skipped.")
            continue
        # plot
        plot_func = getattr(plotter, plot_cfg["handler"])
        figs = plot_func(**plot_cfg.get("args", {}, check_none=True))
        # save
        if _out_dir is not None:
            file_type = plot_cfg.get("file_type", "png", check_none=True)
            for name, fig in figs.items():
                file_name = f"{name}-{plot_name}" if name else plot_name
                save_path = _out_dir / f"{file_name}.{file_type}"
                fig.savefig(save_path, **save_args)
                logger.info(f"Saved plot '{file_name}' to {save_path}")
        # update dict
        plot_figs[plot_name] = figs

    return plot_figs
