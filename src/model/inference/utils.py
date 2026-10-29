
from __future__ import annotations
import torch
from functools import wraps
from transformers import BatchEncoding
from typing import Any, List, Mapping, Literal, overload, Tuple, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.config import TrackingConfig
    from transformers.modeling_outputs import CausalLMOutputWithPast

MODEL_INPUT_KEYS = ("input_ids", "attention_mask", "labels")


def batch_to_device(batch: Any, device: torch.device):
    """
    Move a batch of data to the specified device.
    Only `torch.Tensor` values are moved, other values (e.g. string metadata) are kept as they are.
    """
    if isinstance(batch, dict):
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    elif isinstance(batch, BatchEncoding):
        return batch.to(device)
    else:
        raise ValueError(
            f"Batch to device error, expected batch to be a `dict` or `BatchEncoding`, but got {type(batch)}."
        )


def get_model_device(model: Any) -> torch.device:
    """Device of a model; also for wrappers without a `device` attribute (e.g. `DistributedDataParallel`)."""
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def batch_to_model_device(func):
    @wraps(func)
    def wrapper(model: Any, batch: Any, *args, **kwargs):
        device = get_model_device(model)
        if not (isinstance(batch, BatchEncoding) and batch.device == device):
            batch = batch_to_device(batch, device=device)
        return func(model, batch, *args, **kwargs)
    return wrapper


@overload
def forward_batch(model: Any, batch: Mapping[str, torch.Tensor], ignore_labels: Literal[False] = ..., grad: bool = ...) -> Tuple[torch.Tensor, torch.Tensor, CausalLMOutputWithPast]: ...
@overload
def forward_batch(model: Any, batch: Mapping[str, torch.Tensor], ignore_labels: Literal[True], grad: bool = ...) -> Tuple[torch.Tensor, CausalLMOutputWithPast]: ...


@batch_to_model_device
def forward_batch(
    model: Any,
    batch: Mapping[str, torch.Tensor],
    ignore_labels: bool = False,
    grad: bool = True
):
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
    batch = {k: v for k, v in batch.items() if k not in ignore_keys and isinstance(v, torch.Tensor)}

    with torch.set_grad_enabled(grad):
        outputs = model(**batch)
    # logits
    logits = outputs.logits  # [bsz, seq_len, vocab_size]
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


def decode_labels(
    tokenizer: Any,
    batch: Mapping[str, torch.Tensor],
    ignore_index: int = -100,
) -> List[str]:
    """Decode non-ignored label tokens for each sample."""
    labels = batch["labels"]
    return tokenizer.batch_decode(
        [label[label != ignore_index] for label in labels],
        skip_special_tokens=True,
    )


def _cut_off(s: str, stop_strings: List[str]) -> str:
    """Cut off the string `s` at the earliest occurrence of any of the stop_strings in `seps`."""
    end = len(s)
    for sep in stop_strings:
        idx = s.find(sep)
        if idx != -1:
            end = min(end, idx)
    return s[:end].strip()


@batch_to_model_device
def generate_batch(
    model: Any,
    batch: Mapping[str, torch.Tensor],
    tokenizer: Any,
    generation_args: TrackingConfig,
    ignore_index: int = -100,
    with_meta: bool = False
):
    """
    Generate texts for a batch of inputs using the model and tokenizer.

    Parameters
    ----------
    model : Any
        The model used for generation.
    batch : Mapping[str, torch.Tensor]
        A batch of input tensors.
    tokenizer : Any
        The tokenizer used to decode the input and output tensors.
    generation_args : TrackingConfig
        The generation arguments.
    ignore_index : int, default `-100`
        The index to ignore in the labels.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries containing the input, ground truth, and generated texts.
    """
    # input_texts
    input_ids = batch["input_ids"]
    input_texts = tokenizer.batch_decode(
        input_ids,
        skip_special_tokens=True
    )

    # ground_truths
    ground_truths = decode_labels(tokenizer, batch, ignore_index=ignore_index)

    # generate texts
    initial_input_length = input_ids.shape[1]
    gen_kwargs = generation_args.to_dict()
    stop_strings = list(gen_kwargs.get("stop_strings") or [])
    # generate batch outputs
    outputs = model.generate(
        input_ids,
        attention_mask=batch["attention_mask"],
        **gen_kwargs,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_texts = tokenizer.batch_decode(
        outputs[:, initial_input_length:],
        skip_special_tokens=True
    )
    # cut off at stop_strings and strip
    gen_texts = [_cut_off(text, stop_strings) for text in gen_texts]

    def _as_list(values: Any) -> List[Any]:
        return values.tolist() if isinstance(values, torch.Tensor) else list(values)

    # meta
    batch_meta = {
        k: _as_list(v)
        for k, v in batch.items()
        if k not in MODEL_INPUT_KEYS
    } if with_meta else {}
    # create records for each sample in the batch
    records: List[Dict[str, Any]] = []
    for i, (input_text, ground_truth, gen_text) in enumerate(zip(input_texts, ground_truths, gen_texts)):
        record = {
            "input": input_text,
            "ground_truth": ground_truth,
            "generation": gen_text,
        }
        # meta
        for k, v in batch_meta.items():
            record[k] = v[i]

        records.append(record)

    return records
