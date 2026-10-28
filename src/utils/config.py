
from __future__ import annotations
import sys
from collections.abc import Mapping, Iterator, ItemsView
from omegaconf import DictConfig, open_dict, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import reprlib
from functools import wraps
from typing import Any, Dict, Sequence, Optional, Callable

reprlib.aRepr.maxlist = 5
reprlib.aRepr.maxstring = 15

_MISSING = object()
HYDRA_CHOICES: Dict[str, str] = {}


def hydra_main(
    config_path: str,
    config_name: Optional[str] = None,
    argv: Optional[Sequence[str]] = None
) -> Callable[[Callable[..., Any]], Any]:
    """Decorator for Hydra main functions that wraps the first argument as TrackingConfig.

    Args:
        config_path (str): Path to the Hydra configuration directory.
        config_name (str): Name of the Hydra configuration file.
        argv (Optional[Sequence[str]]): Additional command-line arguments to pass to Hydra.

    Returns:
        Callable: Decorated function that wraps the first argument as TrackingConfig.
    """
    def _decorator(func):
        @wraps(func)
        def _task(cfg: DictConfig, *args, **kwargs):
            """Wrap the first argument as TrackingConfig and initialize Hydra choices."""
            init_hydra_choices(HydraConfig.get().runtime.choices)
            return func(TrackingConfig(cfg), *args, **kwargs)

        hydra_main_fn = hydra.main(version_base=None, config_path=config_path, config_name=config_name)(_task)

        @wraps(func)
        def _entry(*args, **kwargs):
            """Entry point for Hydra main with TrackingConfig wrapping."""
            if argv:
                sys.argv += [a for a in argv if a not in sys.argv]
            return hydra_main_fn(*args, **kwargs)
        return _entry
    return _decorator


def init_hydra_choices(choices: Dict[str, str]):
    HYDRA_CHOICES.clear()
    for loc, choice in choices.items():
        if loc.startswith("hydra"):
            continue
        loc = loc.split("@", 1)[-1]
        HYDRA_CHOICES[loc] = choice


def _check_and_return(func):
    @wraps(func)
    def _wrapper(self, key, default=_MISSING, check_none=False):
        if key not in self._cfg:
            if default is not _MISSING:
                return default
            raise KeyError(f"`{key}` not found in `@{self._loc_choices}`.")
        val = func(self, key)
        if val is None and check_none:
            raise ValueError(f"`{key}` is None in `@{self._loc_choices}`.")

        if isinstance(val, DictConfig):
            # new_loc
            new_loc = f"{self._loc}.{key}" if self._loc else key
            # new_loc_choices
            choice = HYDRA_CHOICES.get(new_loc, "")
            _key_choice = f"{key}({choice})" if choice else key
            new_loc_choices = f"{self._loc_choices}.{_key_choice}" if self._loc_choices else _key_choice
            return TrackingConfig(val, new_loc, new_loc_choices)
        return val
    return _wrapper


class TrackingConfig(Mapping):
    def __init__(self, cfg: DictConfig, loc: str = "", loc_choices: str = ""):
        self._loc = loc
        self._loc_choices = loc_choices
        if not isinstance(cfg, DictConfig):
            raise ValueError(f"`@{self._loc_choices}` must be a DictConfig, got {type(cfg)}")
        self._cfg = cfg

    @_check_and_return
    def get(self, key: Any, default: Any = _MISSING, check_none: bool = False) -> Any:
        return self._cfg[key]

    @_check_and_return
    def pop(self, key: Any, default: Any = _MISSING, check_none: bool = False) -> Any:
        with open_dict(self._cfg):
            return self._cfg.pop(key)

    def copy(self):
        return TrackingConfig(self._cfg.copy(), self._loc, self._loc_choices)

    def to_dict(self) -> Dict[Any, Any]:
        return OmegaConf.to_container(self._cfg, resolve=True)  # type: ignore

    def items(self) -> ItemsView[str, Any]:
        return super().items()

    def __getitem__(self, key):
        return self.get(key)

    def __iter__(self) -> Iterator[str]:
        return (str(k) for k in self._cfg)

    def __len__(self):
        return len(self._cfg)

    def __setitem__(self, key: str, value: Any):
        with open_dict(self._cfg):
            self._cfg[key] = value

    def __str__(self):
        return reprlib.repr({**self._cfg})

    @property
    def cfg(self):
        return self._cfg

    @property
    def loc(self):
        return self._loc

    @property
    def loc_choices(self):
        return self._loc_choices
