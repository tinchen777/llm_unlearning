
from __future__ import annotations
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Tuple, TYPE_CHECKING

from .probe import ProbedLlamaForCausalLM

if TYPE_CHECKING:
    from utils.config import TrackingConfig

logger = logging.getLogger("model")

MODEL_HANDLER_REGISTRY: Dict[str, Any] = {}
TOKENIZER_HANDLER_REGISTRY: Dict[str, Any] = {}


def _register_model_handler(model_handler):
    MODEL_HANDLER_REGISTRY[model_handler.__name__] = model_handler


def _register_tokenizer_handler(tokenizer_handler):
    TOKENIZER_HANDLER_REGISTRY[tokenizer_handler.__name__] = tokenizer_handler


def get_model_and_tokenizer(model_cfg: TrackingConfig) -> Tuple[Any, Any]:
    base_name = model_cfg.get("base_name", "???")
    # get model
    model = _get_model(model_cfg["pretrained"], base_name=base_name)
    # get tokenizer
    tokenizer = _get_tokenizer(model_cfg["tokenizer"], base_name=base_name)

    return model, tokenizer

# === Model ===

def _get_model(pretrained_cfg: TrackingConfig, base_name: str = "???"):
    # model handler class
    handler_cls = MODEL_HANDLER_REGISTRY[pretrained_cfg["handler"]]
    # dtype
    dtype_str = pretrained_cfg.get("dtype", None)
    if dtype_str == "float16":
        dtype = torch.float16
    elif dtype_str == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    # name_or_path
    name_or_path = pretrained_cfg["name_or_path"]

    try:
        model = handler_cls.from_pretrained(
            pretrained_model_name_or_path=name_or_path,
            dtype=dtype,
            **pretrained_cfg["args"]
        )
    except Exception as e:
        raise RuntimeError(
            f"Error loading model `{name_or_path}` with {pretrained_cfg} via {handler_cls.__name__}.from_pretrained()."
        ) from e
    logger.info(f"Loaded model from `{name_or_path}`, based on `{base_name}`")

    return model


# def _get_peft_lora_model(model, peft_args: TrackingConfig):
#     """Wrap a base causal LM with a LoRA adapter using the `peft` library.

#     Args:
#         model: a freshly loaded base model (e.g. AutoModelForCausalLM).
#         peft_args (TrackingConfig): LoRA hyper-parameters. Recognised keys mirror
#             `peft.LoraConfig` (r, lora_alpha, lora_dropout, target_modules,
#             bias, task_type, ...). An optional `path` key can point to an
#             existing adapter checkpoint to resume/evaluate instead of creating
#             a fresh adapter.
#     """
#     try:
#         from peft import LoraConfig, get_peft_model, PeftModel  # type: ignore
#     except ImportError as e:
#         raise ImportError(
#             "LoRA finetuning requires the `peft` library. Install it with "
#             "`pip install peft`."
#         ) from e

#     adapter_path = peft_args.pop("path", None)

#     if adapter_path is not None:
#         # Load a previously trained LoRA adapter on top of the base model.
#         model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
#         logger.info(f"Loaded existing LoRA adapter from {adapter_path}")
#     else:
#         model = get_peft_model(model, LoraConfig(**peft_args))
#         logger.info("Created a new LoRA adapter on top of the base model.")
#     model.print_trainable_parameters()
#     return model

# === Tokenizer ===

def _get_tokenizer(tokenizer_cfg: TrackingConfig, base_name: str = "???"):
    # tokenizer handler class
    handler_cls = TOKENIZER_HANDLER_REGISTRY[tokenizer_cfg["handler"]]
    # name_or_path
    name_or_path = tokenizer_cfg["name_or_path"]

    try:
        tokenizer = handler_cls.from_pretrained(
            pretrained_model_name_or_path=name_or_path,
            **tokenizer_cfg["args"]
        )
    except Exception as e:
        raise RuntimeError(
            f"Error loading tokenizer `{name_or_path}` with {tokenizer_cfg} via {handler_cls.__name__}.from_pretrained()."
        ) from e

    if tokenizer.eos_token_id is None:
        logger.info("replacing eos_token with <|endoftext|>")
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad_token as eos token: {}".format(tokenizer.pad_token))
    logger.info(f"Loaded tokenizer from `{name_or_path}`, based on `{base_name}`")

    return tokenizer


def _add_or_replace_eos_token(tokenizer, eos_token: str):
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.info("New tokens have been added, make sure `resize_vocab` is True.")


# register model handler
_register_model_handler(AutoModelForCausalLM)
_register_model_handler(ProbedLlamaForCausalLM)

# register tokenizer handler
_register_tokenizer_handler(AutoTokenizer)
