"""answer_logprobs (left padded prompt-only batch) vs the eval metric `evaluate_probability` (right padded prompt+answer).

Both batches come from the real `tok_chat_sample` + `DataCollatorForNestedData`, tiny random Llama, no network:
    python -m pytest tests/test_answer_logprobs.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from data.collators.nested_data import DataCollatorForNestedData
from data.datasets.utils import tok_chat_sample
from evals.metrics.metric_utils import evaluate_probability
from model.generation import answer_logprobs

QA = [
    ("who wrote it ?", "the author is jane doe ."),
    ("where was jane born in the year of the dragon ?", "paris ."),
    ("what genre ?", "she writes long and quiet fantasy novels about the sea and the old city ."),
    ("name her first book", "the quiet sea ."),
]
TEMPLATE = {"user_start_tag": "Q: ", "user_end_tag": " ", "asst_start_tag": "A: "}


@pytest.fixture(scope="module")
def setup():
    words = sorted({w for q, a in QA for w in (q + " " + a).split()} | {"Q:", "A:"})
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, **{w: i + 3 for i, w in enumerate(words)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
    )).eval()
    return tokenizer, model


def _batch(tokenizer, generate: bool):
    samples = [
        tok_chat_sample(q, a, i, tokenizer=tokenizer, chat_header="", template_args=TEMPLATE,
                        max_length=128, predict_with_generate=generate)
        for i, (q, a) in enumerate(QA)
    ]
    samples = [{"input_ids": s["input_ids"][0], "labels": s["labels"][0], "attention_mask": [1] * len(s["input_ids"][0]),
                "index": s["index"]} for s in samples]
    collator = DataCollatorForNestedData(tokenizer, padding_side="left" if generate else "right")
    return collator(samples)


def test_answer_logprobs_matches_eval_probability(setup):
    tokenizer, model = setup
    left, right = _batch(tokenizer, True), _batch(tokenizer, False)
    assert bool(left["attention_mask"][:, -1].all()) and not bool(right["attention_mask"][:, -1].all())
    mine = torch.tensor(answer_logprobs(model, left, tokenizer.pad_token_id))
    ref = -torch.tensor([r["avg_loss"] for r in evaluate_probability(model, right)])
    print("answer_logprobs:", mine.tolist(), "\nevaluate_probability:", ref.tolist())
    torch.testing.assert_close(mine, ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(mine.exp(), torch.tensor([r["prob"] for r in evaluate_probability(model, right)]), atol=1e-5)
