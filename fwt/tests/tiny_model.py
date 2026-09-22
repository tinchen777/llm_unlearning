"""A tiny randomly-initialised Llama + tokenizer, for offline smoke tests.

No Hub access is needed: the tokenizer is built from a word-level vocabulary
over the test text, and the model is a 4-layer / 32-dim Llama.
"""

from __future__ import annotations
from pathlib import Path
import re
from typing import Any, List, Optional, Sequence, Tuple

CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{{ '<|' + m['role'] + '|> ' + m['content'] + ' <|end|> ' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|> ' }}{% endif %}"
)


def build_tokenizer(corpus: Sequence[str], out_dir: Path) -> Any:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = {"<pad>", "<unk>", "<s>", "</s>", "<|user|>", "<|assistant|>", "<|system|>", "<|end|>"}
    for text in corpus:
        words.update(re.findall(r"\w+|[^\w\s]", text))
    vocab = {word: i for i, word in enumerate(sorted(words))}

    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>", pad_token="<pad>", bos_token="<s>", eos_token="</s>",
        # registered as added tokens so the chat markers stay single tokens,
        # as they do with a real chat tokenizer
        additional_special_tokens=["<|user|>", "<|assistant|>", "<|system|>", "<|end|>"],
        chat_template=CHAT_TEMPLATE,
    )
    tokenizer.save_pretrained(out_dir)
    return tokenizer


def build_model(vocab_size: int, num_layers: int = 4, hidden: int = 32) -> Any:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        tie_word_embeddings=True,
    )
    return LlamaForCausalLM(config)


def tiny_model_and_tokenizer(
    corpus: Sequence[str],
    out_dir: Optional[Path] = None,
    num_layers: int = 4,
    hidden: int = 32,
) -> Tuple[Any, Any]:
    out_dir = Path(out_dir or Path(__file__).parent / "_tiny")
    tokenizer = build_tokenizer(corpus, out_dir)
    model = build_model(len(tokenizer), num_layers=num_layers, hidden=hidden)
    return model, tokenizer


def default_corpus() -> List[str]:
    from fwt.tests.fixtures import load_fixture_rows

    rows = load_fixture_rows()
    return [r["question"] for r in rows] + [r["answer"] for r in rows]
