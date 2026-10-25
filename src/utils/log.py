
from __future__ import annotations
import logging
from rich.markup import escape
from rich import print
from cobra_color import cstr
from contextlib import contextmanager
from typing import Any, Optional, Literal, overload, Generator

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


CONFIG_PLACEHOLDER = "<CONFIG_PLACEHOLDER>"
LOGGING_MODE = {
    "load": ("Loading", "Loaded"),
    "step": ("Start", "End"),
}

@overload
@contextmanager
def step_logging(
    logger: logging.Logger,
    step_tag: str,
    name: str,
    *,
    mode: Literal["load", "step"] = ...,
    cfg: Optional[TrackingConfig] = ...,
    ensure_cfg: Literal[True],
) -> Generator[TrackingConfig, Any, None]: ...
@overload
@contextmanager
def step_logging(
    logger: logging.Logger,
    step_tag: str,
    name: str,
    *,
    mode: Literal["load", "step"] = ...,
    cfg: Optional[TrackingConfig] = ...,
    ensure_cfg: Literal[False] = ...,
) -> Generator[Optional[TrackingConfig], Any, None]: ...
@overload
@contextmanager
def step_logging(
    logger: logging.Logger,
    step_tag: str,
    name: str,
    *,
    mode: Literal["load", "step"] = ...,
    cfg: Optional[TrackingConfig] = ...,
    ensure_cfg: bool,
) -> Generator[Optional[TrackingConfig], Any, None]: ...


@contextmanager
def step_logging(
    logger: logging.Logger,
    step_tag: str,
    name: str,
    *,
    mode: Literal["load", "step"] = "step",
    cfg: Optional[TrackingConfig] = CONFIG_PLACEHOLDER,  # type: ignore
    ensure_cfg: bool = False,
) -> Generator[Optional[TrackingConfig], Any, None]:
    # name
    name += f" `@{cfg.loc_choices}`" if isinstance(cfg, TrackingConfig) else ""
    # mode
    header, footer = LOGGING_MODE[mode]

    if cfg is None:
        logger.info(
            cstr(step_tag, fg="g", styles="bold") +
            cstr(f" Skipping {name}.", styles=("bold", "del"))
        )
        yield
    else:
        logger.info(
            cstr(step_tag, fg="d", bg="y", styles="bold") +
            cstr(f" {header} {name} ...", styles="bold")
        )
        try:
            if cfg is CONFIG_PLACEHOLDER or cfg is None:
                if ensure_cfg:
                    raise ValueError(f"`cfg` must be a `TrackingConfig`, got {type(cfg)}.")
                else:
                    yield None
            else:
                yield cfg
        except Exception as e:
            raise RuntimeError(f"Error for {name}!") from e
        else:
            logger.info(
                cstr(step_tag, fg="d", bg="g", styles="bold") +
                cstr(f" {footer} {name}.", styles="bold")
            )
    print()
