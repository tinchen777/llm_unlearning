"""MEOW baseline (src/METHOD/MEOW/meow.py): chunks / MEMO / selection of the inverted facts, neighbour targets, the released repetition scheme, and the fine-tuning loop on a tiny random Llama. No network.
    python -m pytest tests/test_meow.py -q
"""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from METHOD.MEOW import meow

TEMPLATE = {"apply_chat_template": True, "system_prompt": "You are a helpful assistant.", "date_string": "10 Apr 2025"}


def test_split_chunks_follow_the_suffix_mode():
    ch = meow.split_chunks("Q?", "abcdefghijkl", window=5)
    assert ch == [("Q? a", "bcdefghijkl"), ("Q? abcdef", "ghijkl"), ("Q? abcdefghijk", "l")]
    assert meow.split_chunks("Q?", "a") == []


def test_rouge1_precision():
    assert meow.rouge1_precision("the cat sat", "the cat sat down") == 1.0
    assert abs(meow.rouge1_precision("the cat sat on", "the cat sat down") - 0.75) < 1e-9


def _sample(scores):
    s = {"question": "Who?", "undesired_answer": "True answer."}
    for i, v in enumerate(scores):
        s[f"answer_{i}"] = f"inverted {i}"
        s[f"answer_{i}_rouge1"] = v
    return s


def test_select_answers_largest_smallest_random_and_missing_scores():
    s = _sample([0.3, 0.9, 0.1, 0.5])
    assert meow.select_answers(s, 2, "largest") == ["answer_1", "answer_3"]
    assert meow.select_answers(s, 2, "smallest") == ["answer_2", "answer_0"]
    r = meow.select_answers(s, 3, "random")
    assert len(set(r)) == 3 and r == meow.select_answers(s, 3, "random")
    nos = {k: v for k, v in s.items() if not k.endswith("_rouge1")}
    assert len(meow.select_answers(nos, 2, "largest")) == 2          # no MEMO: falls back to a random pick instead of failing
    assert meow.select_answers(_sample([0.2]), 3, "largest") == ["answer_0"]


def test_neighbor_targets_and_repetition(tmp_path):
    p = tmp_path / "nb.csv"
    rows = ["forget_idx,split,train_rank,answer"]
    for i in range(3):
        rows += [f"{200 + i},train,{r},neighbour answer {i}.{r}" for r in range(4)] + [f"{200 + i},heldout,0,held out {i}"]
    p.write_text("\n".join(rows))
    ans = meow.neighbor_answers(str(p), "forget05", 3, 2)
    assert ans == [[f"neighbour answer {i}.0", f"neighbour answer {i}.1"] for i in range(3)]
    with pytest.raises(ValueError):
        meow.neighbor_answers(str(p), "forget05", 4, 2)
    data = [_sample([0.1, 0.2]) | {"question": f"Q{i}"} for i in range(3)]
    pairs, sel = meow.build_pairs(data, "neighbor", 2, "largest", str(p), "forget05")
    assert pairs[1] == ("Q1", ans[1]) and "answers" in sel[1]
    rows8 = meow.expand_rows(pairs, 8)
    assert len(rows8[0]) == 8 and [a for _, a in rows8[0][:4]] == [ans[0][0], ans[0][1], ans[0][0], ans[0][1]]


@pytest.fixture(scope="module")
def tiny():
    words = {"Who", "Where", "is", "was", "born", "the", "author", "Jane", "Doe", "Paris", "Rome", "She", "He", "lives", "in", "True", "inverted", "answer", "?", ".", "User", "Assistant", "system", "user", "assistant"}
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, **{w: i + 3 for i, w in enumerate(sorted(words))}}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    tok.chat_template = ("{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}{% if add_generation_prompt %}assistant {% endif %}")
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                                         num_key_value_heads=4, max_position_embeddings=128))
    return tok, model


def test_memo_is_a_score_in_0_1(tiny):
    tok, model = tiny
    model.eval()
    sc = meow.memo_of_answers(model, tok, "Where was Jane Doe born ?", ["She was born in Paris .", "He lives in Rome ."], window=5, max_new_tokens=5, batch_size=4)
    assert len(sc) == 2 and all(0.0 <= v <= 1.0 for v in sc)
    out = meow.add_memo(model, tok, [{"question": "Who is Jane ?", "undesired_answer": "She is the author .", "answer_0": "He is the author ."}], max_new_tokens=4, log=None)
    assert {"undesired_answer_rouge1", "answer_0_rouge1"} <= set(out[0])


def test_training_fits_the_labels_and_saves_checkpoints(tiny, tmp_path):
    tok, model = tiny
    pairs = [("Where was Jane Doe born ?", ["She was born in Paris ."]), ("Who is Jane ?", ["She is the author ."])]
    groups = meow.tokenize_rows(meow.expand_rows(pairs, 4), tok, TEMPLATE, max_length=64)
    # the labels are only the answer tokens
    ids, lab = groups[0][0]
    assert sum(1 for x in lab if x != -100) > 0 and lab[0] == -100 and len(ids) == len(lab)
    hist = meow.train(model, tok, groups, steps=40, lr=5e-3, rows_per_step=8, micro_rows=4, save_steps=[10, 40], out_dir=str(tmp_path), log=None)
    assert hist[-1]["loss"] < 0.5 * hist[0]["loss"]
    assert (tmp_path / "checkpoint-10" / "config.json").exists() and (tmp_path / "checkpoint-40" / "config.json").exists()
