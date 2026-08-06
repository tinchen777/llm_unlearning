from __future__ import annotations
# from rich.traceback import install
# install(show_locals=False, width=100)
import hydra
from hydra.core.hydra_config import HydraConfig
from typing import TYPE_CHECKING

from vis import plot_figures
from utils.config import TrackingConfig, init_hydra_choices

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="plot")
def main(config: DictConfig):
    """Entry point of the code to plot figures
    Args:
        config (DictConfig): Config for plotting
    """
    # config
    init_hydra_choices(HydraConfig.get().runtime.choices)
    vis_cfg = TrackingConfig(config)["vis"]
    # plot figures
    plot_figures(vis_cfg["run_dirs"], vis_cfg, out_dir=vis_cfg["out_dir"])


if __name__ == "__main__":
    main()
