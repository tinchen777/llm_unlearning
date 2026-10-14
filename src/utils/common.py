
from __future__ import annotations
import os
from pathlib import Path
import json
import torch
import random
import numpy as np
from typing import Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from os import PathLike


def set_seed(seed: int = 42):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cuda_visible_devices():
    num_devices = torch.cuda.device_count()
    _device = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        assert _device is not None
        devices = [int(x) for x in _device.split(',')][:num_devices]
    except Exception:
        devices = list(range(num_devices))
    return devices


def load_logs(file_path: Union[PathLike, str]) -> Any:
    """Returns the cache of existing results"""
    file_path = Path(file_path)
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_logs(logs: Any, file_path: Union[PathLike, str]):
    """Save the logs in a json file"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4, sort_keys=True, ensure_ascii=False)
    except Exception as e:
        raise RuntimeError(f"Failed to save {file_path}") from e
