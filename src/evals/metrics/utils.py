
from __future__ import annotations
import numpy as np
import torch
from functools import wraps
from tqdm import tqdm
import logging
from typing import List, Any, Dict, Callable, Mapping

from utils.common import to_device

DATA_SPLIT_SUFFIX = "_dl"

logger = logging.getLogger("eval.metric")


def run_batchwise_evals(
    model: Any,
    dataloader: Any,
    batch_eval_fn: Callable[..., List[Dict[str, Any]]],
    batch_eval_fn_args: Dict[str, Any] = {},
    eval_name: str = "???"
):
    """Run batch-wise evaluations on a dataset using a specified evaluation function. Handles
    multi-answer datasets by organizing evaluations by answer indices and aggregating results."""
    item_sample_evals: Dict[int, Dict[int, Dict[str, Any]]] = {}
    # evals looks like {iidx0: {idx453: {prob: 0.1, loss: 1}},
    #                   iidx1: {idx453: {prob: 0.2, loss: 2}}}
    try:
        data_size = len(dataloader.dataset)
        pbar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Calculating [{eval_name}]",
            unit="batch(es)",
            colour="blue"
        )
        for batch in pbar:
            if "input_ids" in batch:
                batch = {0: batch}
            # Assume batch like {0: {"input_ids": [[]]..., index: [453, 454..]},
            #                    1: {"input_ids": [[]]..., index: [453, 454..]}..}
            for intra_item_idx, mini_batch in batch.items():
                if "input_ids" not in mini_batch:
                    raise ValueError(
                        f"Expected mini_batch to contain 'input_ids', but got {list(mini_batch)}."
                    )
                data_indices: List[int] = mini_batch.pop("index")
                batch_evals = batch_eval_fn(
                    model=model,
                    batch=mini_batch,
                    **batch_eval_fn_args
                )
                indexwise_batch_evals = dict(zip(data_indices, batch_evals))

                item_sample_evals.setdefault(intra_item_idx, {}).update(indexwise_batch_evals)
            # progress bar update
            pbar.set_postfix_str(f"[{len(item_sample_evals.get(0, {}))} / {data_size}] sample(s) evaluated")
        pbar.close()

        if len(item_sample_evals) == 1:  # normal single answer dataset, no need for list
            sample_evals = item_sample_evals[0]
        else:
            # for each index return a dict with all intra_item_idx values in list
            # after dict transpose looks like {idx453: {prob: [0.1, 0.2], loss: [1, 2]}}
            sample_evals = dict_transpose(item_sample_evals)
        return sample_evals

    except Exception as e:
        raise RuntimeError(f"Error during batch-wise evaluation with {eval_name}") from e


def batch_to_model_device(func):
    @wraps(func)
    def wrapper(model: Any, batch: Mapping[str, torch.Tensor], **kwargs):
        batch = to_device(batch, model.device)
        return func(model=model, batch=batch, **kwargs)
    return wrapper


def dict_transpose(evals: Dict[int, Dict[int, Dict[str, Any]]]) -> Dict[int, Dict[str, List[Any]]]:
    """Transpose a nested dictionary structure to group statistics by item indices."""
    # evals looks like {iidx0: {idx453: {prob: 0.1, loss: 1}},
    #                   iidx1: {idx453: {prob: 0.2, loss: 2}}}
    # multiple answers indexed by intra_item_idx, then item_idx
    # invert the dict, put outermost iidx deepest inside
    # after dict transpose looks like {idx453: {prob: [0.1, 0.2], loss: [1, 2]}}
    all_iidxs = list(evals)
    all_idxs = list(evals[all_iidxs[0]])
    all_stat_names = list(evals[all_iidxs[0]][all_idxs[0]])
    return {
        idx: {
            stat: [evals[iidx][idx][stat] for iidx in all_iidxs]
            for stat in all_stat_names
        }
        for idx in all_idxs
    }


def aggregate_to_1D(x):
    return np.mean(x, axis=tuple(range(1, x.ndim)))


def topk_mean(tensor: torch.Tensor, ratio: float, largest: bool = True, dim: int = -1):
    """Compute the mean of the top-k elements in the tensor along the specified dimension."""
    if ratio <= 0 or ratio > 1:
        raise ValueError(f"Ratio must be in the range (0, 1], but got {ratio}.")

    if tensor.numel() == 0:
        return 0.0
    k = max(1, int(tensor.size(dim) * ratio))
    topk_values = tensor.topk(k, largest=largest, dim=dim).values
    return float(topk_values.mean().item())


def to_np(tensor: torch.Tensor, is_float: bool = True):
    """Convert a PyTorch tensor to a NumPy array."""
    if is_float:
        return tensor.detach().cpu().to(torch.float32).numpy()
    else:
        return tensor.detach().cpu().numpy()
