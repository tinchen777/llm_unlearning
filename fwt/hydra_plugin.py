"""Expose `fwt/configs` to Hydra without copying anything into `configs/`.

Hydra composes from the project's own `configs/` directory; registering this
search-path plugin adds `fwt/configs` next to it, so `trainer=NeighborRedirect`
and `experiment=unlearn/tofu/fwt` resolve while the original tree stays clean.
"""

from __future__ import annotations
import logging

from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin

from fwt._paths import FWT_CONFIG_DIR

logger = logging.getLogger(__name__)
_REGISTERED = False


class FwtSearchPathPlugin(SearchPathPlugin):
    """Appends `fwt/configs` to Hydra's search path."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="fwt", path=f"file://{FWT_CONFIG_DIR}")


def register_fwt_configs() -> None:
    """Idempotently register the plugin (call before `hydra.main` runs)."""
    global _REGISTERED
    if _REGISTERED:
        return
    Plugins.instance().register(FwtSearchPathPlugin)
    _REGISTERED = True
    logger.debug("Hydra search path extended with %s", FWT_CONFIG_DIR)
