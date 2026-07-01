
import os
import json
import torch
import random
import numpy as np
from typing import Any, Dict


def set_seed(seed=42):
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


def load_logs_from_file(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Returns the cache of existing results"""
    with open(file_path, "r") as f:
        return json.load(f)


def save_logs(logs: Dict[str, Any], file_path: str):
    """Save the logs in a json file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, "w") as f:
            json.dump(logs, f, indent=4, sort_keys=True)
    except Exception as e:
        raise RuntimeError(f"Failed to save {file_path}") from e
