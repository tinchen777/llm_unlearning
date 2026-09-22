
from __future__ import annotations
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Tuple, TYPE_CHECKING

from .probe import ProbedLlamaForCausalLM

if TYPE_CHECKING:
    from utils.config import TrackingConfig

logger = logging.getLogger(__name__)


def get_model(, model_args: TrackingConfig, model_handler: str):
    torch_dtype = _get_dtype(model_args)
    model_cls = MODEL_REGISTRY[model_handler]
    model_path = model_args.pop("pretrained_model_name_or_path", check_none=True)
    try:
        model = model_cls.from_pretrained(
            pretrained_model_name_or_path=model_path,
            dtype=torch_dtype,
            **model_args,
        )
    except Exception as e:
        raise ValueError(
            f"Error loading model `{model_path}` with {model_args} via {model_handler}.from_pretrained()."
        ) from e

    return model


def _get_tokenizer(tokenizer_args: TrackingConfig):
    model_path = tokenizer_args.pop("pretrained_model_name_or_path", check_none=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_path,
            **tokenizer_args
        )
    except Exception as e:
        raise RuntimeError(
            f"Error loading tokenizer `{model_path}` with {tokenizer_args} via AutoTokenizer.from_pretrained()."
        ) from e

    if tokenizer.eos_token_id is None:
        logger.info("replacing eos_token with <|endoftext|>")
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad_token as eos token: {}".format(tokenizer.pad_token))

    return tokenizer
