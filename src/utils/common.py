
from __future__ import annotations
import os
from pathlib import Path
import json
import torch
import random
import numpy as np
from torch.nn import functional as F
from transformers import BatchEncoding
from typing import Any, Dict, Mapping, Literal, overload, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from os import PathLike

IGNORE_INDEX = -100


def randidx(high: int) -> int:
    """Returns a random integer in the range [0, high)."""
    return int(torch.randint(high, ()).item())


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


def load_logs(file_path: Union[PathLike, str]) -> Dict[str, Any]:
    """Returns the cache of existing results"""
    file_path = Path(file_path)
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_logs(logs: Dict[str, Any], file_path: Union[PathLike, str]):
    """Save the logs in a json file"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4, sort_keys=True, ensure_ascii=False)
    except Exception as e:
        raise RuntimeError(f"Failed to save {file_path}") from e


@overload
def forward_batch(model: Any, batch: Mapping[str, torch.Tensor], ignore_labels: Literal[False] = ..., grad: bool = ...) -> Tuple[torch.Tensor, torch.Tensor, CausalLMOutputWithPast]: ...
@overload
def forward_batch(model: Any, batch: Mapping[str, torch.Tensor], ignore_labels: Literal[True], grad: bool = ...) -> Tuple[torch.Tensor, CausalLMOutputWithPast]: ...
def forward_batch(model: Any, batch: Mapping[str, torch.Tensor], ignore_labels: bool = False, grad: bool = True):
    """
    Forward a batch through the model.

    Parameters
    ----------
        model : Any
            The model to forward the batch through.

        batch : Mapping[str, torch.Tensor]
            The batch of data to forward through the model.

        ignore_labels : bool, default `False`
            - `True`, the 'labels' key in the batch will be ignored, which means the model will not compute the loss.
            - `False`, the 'labels' key in the batch will be used to compute the loss.

        grad : bool, default `True`
            - `True`, gradients will be computed.

    Returns
    -------
        For `ignore_labels`：
            - `True`: Returns a tuple of (`logits`, `outputs`) of the model.
            - `False`: Returns a tuple of (`loss`, `logits`, `outputs`)
        `logits` : torch.Tensor
            Shape as `[batch_size, seq_len, vocab_size]`.
        `loss` : torch.Tensor
            `Scalar` tensor representing the loss.
        `outputs` : CausalLMOutputWithPast
            The output of the model, which includes `logits`, `loss`, etc.
    """
    ignore_keys = ("index", "labels") if ignore_labels else ("index",)
    batch = {k: v for k, v in batch.items() if k not in ignore_keys}

    with torch.set_grad_enabled(grad):
        outputs = model(**batch)
    # logits
    logits = outputs.logits  # bsz x seq_len x vocab_size
    if logits is None:
        raise ValueError("Model output `logits` is `None`. Ensure the model returns logits.")
    if ignore_labels:
        return logits, outputs
    else:
        # loss
        loss = outputs.loss  # scalar
        if loss is None:
            raise ValueError("Model output `loss` is `None`. Ensure the batch contains `labels` and the model returns loss.")
        return loss, logits, outputs


def to_device(batch: Any, device: torch.device):
    """Move a batch of data to the specified device."""
    if isinstance(batch, dict):
        return {k: v.to(device) for k, v in batch.items()}
    elif isinstance(batch, BatchEncoding):
        return batch.to(device)
    else:
        raise ValueError(
            f"Expected batch to be a `dict` or `BatchEncoding`, but got {type(batch)}."
        )


def per_token_CE(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-token cross-entropy loss.

    Parameters
    ----------
        logits : torch.Tensor
            The logits from the model of shape `[batch_size, seq_len, vocab_size]`.

        labels : torch.Tensor
            The ground truth labels of shape `[batch_size, seq_len]`.

    Returns
    -------
    losses : torch.Tensor
        The per-token cross-entropy loss of shape `[batch_size, seq_len-1]`.
    shift_labels_mask : torch.BoolTensor
        A mask indicating valid positions (not equal to `IGNORE_INDEX`) of shape `[batch_size, seq_len-1]`.
    """
    shift_logits = logits[..., :-1, :]  # shape as [batch_size, seq_len-1, vocab_size]
    shift_labels = labels[..., 1:].to(logits.device)  # shape as [batch_size, seq_len-1]
    losses = F.cross_entropy(
        shift_logits.transpose(-1, -2),
        shift_labels,
        ignore_index=IGNORE_INDEX,
        reduction="none",
    )  # shape as [batch_size, seq_len-1]

    return losses, shift_labels != IGNORE_INDEX
