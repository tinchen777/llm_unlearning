
from __future__ import annotations
import torch
from transformers import BatchEncoding
from typing import Any, List, Mapping, Literal, overload, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.config import TrackingConfig
    from transformers.modeling_outputs import CausalLMOutputWithPast


def batch_to_device(batch: Any, device: torch.device):
    """Move a batch of data to the specified device."""
    if isinstance(batch, dict):
        return {k: v.to(device) for k, v in batch.items()}
    elif isinstance(batch, BatchEncoding):
        return batch.to(device)
    else:
        raise ValueError(
            f"Expected batch to be a `dict` or `BatchEncoding`, but got {type(batch)}."
        )


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


def _cut_off_at_stopwords(s: str, seps: List[str]) -> str:
    """Cut off the string `s` at the earliest occurrence of any of the stopwords in `seps`."""
    end = len(s)
    for sep in seps:
        idx = s.find(sep)
        if idx != -1:
            end = min(end, idx)
    return s[:end].strip()


def generate_texts(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    generation_args: TrackingConfig
) -> List[str]:
    """Generate for every (left padded) prompt of `input_ids`; returns the new text only, cut off at the stopwords."""
    initial_input_length = input_ids.shape[1]

    generation_kwargs = generation_args.to_dict()
    assert not generation_kwargs.get("return_dict_in_generate")
    stopwords = list(generation_kwargs.pop("stopwords", []))
    if stopwords:
        generation_kwargs["stop_strings"] = stopwords
        generation_kwargs["tokenizer"] = tokenizer

    # generate outputs
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        **generation_kwargs,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_texts = tokenizer.batch_decode(
        outputs[:, initial_input_length:],
        skip_special_tokens=True
    )

    # cut off at stopwords and strip
    return [_cut_off_at_stopwords(text, stopwords) for text in gen_texts]
