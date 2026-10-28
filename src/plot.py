from __future__ import annotations
import logging
from typing import TYPE_CHECKING

from vis import plot_figures
from utils.log import step_logging
from utils.config import hydra_main

if TYPE_CHECKING:
    from utils.config import TrackingConfig

logger = logging.getLogger("main(plot)")


@hydra_main(config_path="../configs", config_name="plot")
def main(cfg: TrackingConfig):
    """Entry point of the code to plot figures
    Args:
        cfg (TrackingConfig): Config for plotting
    """
    # plot figures
    with step_logging(logger, "#1", "plotting", cfg=cfg["vis"], ensure_cfg=True) as vis_cfg:
        plot_figures(
            *vis_cfg.get("run_dirs", []),
            run_folders=vis_cfg.get("run_folders", None),
            vis_cfg=vis_cfg,
            out_dir=vis_cfg["out_dir"]
        )


if __name__ == "__main__":
    main()
