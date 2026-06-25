
import os
import torch
import random
import numpy as np


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
