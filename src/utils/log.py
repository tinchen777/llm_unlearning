
from __future__ import annotations
import logging
from rich.markup import escape
from rich import print
from cobra_color import cstr
from contextlib import contextmanager
from typing import Any, Optional, Literal, TypeVar, overload, Generator

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

_T = TypeVar("_T")
_Mode = Literal["load", "step"]


@overload
@contextmanager
def step_logging(
    logger: logging.Logger,
    step_tag: str,
    name: str,
    *,
    mode: _Mode = ...,
    cfg: Optional[_T],
    ensure_cfg: Literal[True],
) -> Generator[_T, Any, None]: ...
@overload
@contextmanager
def step_logging(
    logger: logging.Logger,
    step_tag: str,
    name: str,
    *,
    mode: _Mode = ...,
    cfg: _T,
    ensure_cfg: Literal[False] = ...,
) -> Generator[_T, Any, None]: ...
@overload
@contextmanager
def step_logging(
    logger: logging.Logger,
    step_tag: str,
    name: str,
    *,
    mode: _Mode = ...,
    ensure_cfg: Literal[False] = ...,
) -> Generator[None, Any, None]: ...
@overload
@contextmanager
def step_logging(
    logger: logging.Logger,
    step_tag: str,
    name: str,
    *,
    mode: _Mode = ...,
    cfg: Any = ...,
    ensure_cfg: bool,
) -> Generator[Any, Any, None]: ...


@contextmanager
def step_logging(
    logger: logging.Logger,
    step_tag: str,
    name: str,
    *,
    mode: _Mode = "step",
    cfg: Any = CONFIG_PLACEHOLDER,
    ensure_cfg: bool = False,
) -> Generator[Any, Any, None]:
    """
    Log the start / end of a step; yields `cfg` (`None` if it is not given).

    - `cfg` is `None` -> the step is logged as skipped (the body still runs and receives `None`).
    - `cfg` is omitted -> no config is attached to the step (the body receives `None`).
    - `ensure_cfg` -> `cfg` must be given and not `None`, else `ValueError`.
    `cfg` can be any config value (a `TrackingConfig`, a path, a list ...); only a `TrackingConfig` adds its location.
    """
    # name
    name += f" `@{cfg.loc_choices}`" if isinstance(cfg, TrackingConfig) else ""
    # mode
    header, footer = LOGGING_MODE[mode]

    if ensure_cfg and (cfg is CONFIG_PLACEHOLDER or cfg is None):
        raise ValueError(f"`cfg` is required for {name} (`ensure_cfg=True`), got {None if cfg is None else 'nothing'}.")

    if cfg is None:
        logger.info(
            cstr(step_tag, fg="g", styles="bold") +
            cstr(f" Skipping {name}.", styles=("bold", "del"))
        )
        yield None
    else:
        logger.info(
            cstr(step_tag, fg="d", bg="y", styles="bold") +
            cstr(f" {header} {name} ...", styles="bold")
        )
        try:
            yield None if cfg is CONFIG_PLACEHOLDER else cfg
        except Exception as e:
            raise RuntimeError(f"Error for {name}!") from e
        else:
            logger.info(
                cstr(step_tag, fg="d", bg="g", styles="bold") +
                cstr(f" {footer} {name}.", styles="bold")
            )
    print()
