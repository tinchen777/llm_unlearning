
from __future__ import annotations
from collections.abc import Mapping, Iterator, ItemsView
from omegaconf import DictConfig, open_dict
import reprlib
from functools import wraps
from typing import Any, Dict

reprlib.aRepr.maxlist = 5
reprlib.aRepr.maxstring = 15

_MISSING = object()
HYDRA_CHOICES: Dict[str, str] = {}


def init_hydra_choices(choices: Dict[str, str]):
    HYDRA_CHOICES.clear()
    for loc, choice in choices.items():
        if loc.startswith("hydra"):
            continue
        loc = loc.split("@", 1)[-1]
        HYDRA_CHOICES[loc] = choice


def check_and_return(func):
    @wraps(func)
    def wrapper(self, key, default=_MISSING, allow_none=False):
        if key not in self._cfg:
            if default is not _MISSING:
                return default
            raise KeyError(f"`{key}` not found in `@{self._loc_choices}`.")
        val = func(self, key, default, allow_none)
        if val is None and not allow_none:
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
    return wrapper


class TrackingConfig(Mapping):
    def __init__(self, cfg: DictConfig, loc: str = "", loc_choices: str = ""):
        self._loc = loc
        self._loc_choices = loc_choices
        if not isinstance(cfg, DictConfig):
            raise ValueError(f"`@{self._loc_choices}` must be a DictConfig, got {type(cfg)}")
        self._cfg = cfg

    @check_and_return
    def get(self, key: Any, default: Any = _MISSING, allow_none: bool = False) -> Any:
        return self._cfg[key]

    @check_and_return
    def pop(self, key: Any, default: Any = _MISSING, allow_none: bool = False) -> Any:
        with open_dict(self._cfg):
            return self._cfg.pop(key)

    def copy(self):
        return TrackingConfig(self._cfg.copy(), self._loc, self._loc_choices)

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
