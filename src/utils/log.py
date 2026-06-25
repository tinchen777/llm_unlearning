
from __future__ import annotations
import logging
from rich.markup import escape
from rich import print
from cobra_color import cstr
from omegaconf import OmegaConf
from contextlib import contextmanager
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import TrackingConfig


class RichNameFormatter(logging.Formatter):
    """Formatter for use with RichHandler(markup=True).

    Wraps the logger name in a Rich style tag so it gets its own color, while
    escaping the actual log message so any brackets in it (URLs, list reprs like
    ``[q_proj, v_proj]``) are NOT parsed as Rich markup.

    Configured from Hydra's job_logging dictConfig via the ``()`` key, e.g.:

        formatters:
          rich:
            (): log_utils.RichNameFormatter
            name_style: "bold cyan"
    """

    def __init__(self, name_style: str = "bold cyan", **kwargs):
        super().__init__(**kwargs)
        self.name_style = name_style

    def format(self, record: logging.LogRecord) -> str:
        message = escape(record.getMessage())
        name = f"[{self.name_style}]{escape(record.name)}[/] "
        return name + message


@contextmanager
def step_logging(logger: logging.Logger, step: str, name: str, cfg: Optional[TrackingConfig] = None, is_skip: bool = False):
    print()
    # loc of config
    loc = f" `@{cfg.loc_choices}`" if cfg is not None else ""

    if is_skip:
        logger.info(cstr(step, fg="y", styles="bold") + cstr(f" Skipping {name}{loc}.", styles="bold"))
        yield
    else:
        logger.info(cstr(step, fg="y", styles="bold") + cstr(f" Loading {name}{loc} ...", styles="bold"))
        try:
            yield
        except Exception as e:
            raise RuntimeError(f"Error loading {name}{loc}.") from e
        else:
            logger.info(cstr(step, fg="g", styles="bold") + cstr(f" Loaded {name}{loc}.", styles="bold"))
