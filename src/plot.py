from __future__ import annotations
# from rich.traceback import install
# install(show_locals=False, width=100)
import hydra
import logging
from hydra.core.hydra_config import HydraConfig
from typing import TYPE_CHECKING

from vis import plot_figures
from utils.log import step_logging
from utils.config import TrackingConfig, init_hydra_choices

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger("main(plot)")


@hydra.main(version_base=None, config_path="../configs", config_name="plot")
def main(config: DictConfig):
    """Entry point of the code to plot figures
    Args:
        config (DictConfig): Config for plotting
    """
    # config
    init_hydra_choices(HydraConfig.get().runtime.choices)

    # plot figures
    with step_logging(logger, "#1", "plotting", cfg=TrackingConfig(config)["vis"], ensure_cfg=True) as vis_cfg:
        plot_figures(
            *vis_cfg.get("run_dirs", []),
            run_folders=vis_cfg.get("run_folders", None),
            vis_cfg=vis_cfg,
            out_dir=vis_cfg["out_dir"]
        )


if __name__ == "__main__":
    main()
