
from __future__ import annotations
import numpy as np
import torch
from functools import wraps
from tqdm import tqdm
from transformers import BatchEncoding
import logging
from typing import List, Any, Dict, Callable, Mapping, Optional, TYPE_CHECKING

# if TYPE_CHECKING:
#     from transformers.modeling_outputs import CausalLMOutputWithPast

DATA_SPLIT_SUFFIX = "_dl"

logger = logging.getLogger("eval.metric")


def run_batchwise_evals(
    model: Any,
    dataloader: Any,
    batch_eval_fn: Callable[..., List[Dict[str, Any]]],
    batch_eval_fn_args: Dict[str, Any] = {},
    eval_msg: Optional[str] = None
):
    """Run batch-wise evaluations on a dataset using a specified evaluation function. Handles
    multi-answer datasets by organizing evaluations by answer indices and aggregating results."""
    item_sample_evals: Dict[int, Dict[int, Dict[str, Any]]] = {}
    # evals looks like {iidx0: {idx453: {prob: 0.1, loss: 1}},
    #                   iidx1: {idx453: {prob: 0.2, loss: 2}}}
    try:
        for batch in tqdm(dataloader, desc=eval_msg, total=len(dataloader)):
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

        if len(item_sample_evals) == 1:  # normal single answer dataset, no need for list
            sample_evals = next(iter(item_sample_evals.values()))
        else:
            # for each index return a dict with all intra_item_idx values in list
            # after dict transpose looks like {idx453: {prob: [0.1, 0.2], loss: [1, 2]}}
            sample_evals = dict_transpose(item_sample_evals)
        logger.info(f"Evaluated {len(sample_evals)} examples.")
        return sample_evals

    except Exception as e:
        raise RuntimeError(f"Error during batch-wise evaluation with {eval_msg}") from e


def forward_batch(model: Any, batch: Mapping[str, torch.Tensor], ignore_keys: Optional[List[str]] = ["labels"], grad: bool = False) -> torch.Tensor:
    """Forward a batch through the model and return the outputs.
    Return the logits tensor of shape (bsz, seq_len, vocab_size).
    """
    if ignore_keys:
        batch = {k: v for k, v in batch.items() if k not in ignore_keys}
    with torch.set_grad_enabled(grad):
        outputs = model(**batch)
    logits = outputs.logits  # bsz x seq_len x vocab_size
    if logits is None:
        raise ValueError("Model output logits is `None`. Ensure the model is in evaluation mode and returns logits.")
    return logits



def batch_to_model_device(func):
    @wraps(func)
    def wrapper(model: Any, batch: Mapping[str, torch.Tensor], **kwargs):
        batch = to_device(batch, model.device)
        return func(model=model, batch=batch, **kwargs)
    return wrapper


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
