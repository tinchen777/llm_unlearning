"""ROCR baseline (src/METHOD/ROCR/rocr.py): name extraction, token positions, null-space projector, ridge update, and the end-to-end edit on a tiny random Llama.
No network:  python -m pytest tests/test_rocr.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from METHOD.ROCR import rocr

NAMES = ["Basil Mahfouz", "Jane Doe", "Mubarak Yaqub"]


@pytest.fixture(scope="module")
def setup():
    words = {"Answer", ":", ".", "Please", "answer", "the", "question", "about", "Choose", "best", "Write", "Tell", "me", "something", "A", "User"}
    for n in NAMES:
        words |= set(n.split())
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, **{w: i + 3 for i, w in enumerate(sorted(words))}}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    tok.chat_template = "{% for m in messages %}User {{ m['content'] }} {% endfor %}{% if add_generation_prompt %}A {% endif %}"
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                                         num_key_value_heads=4, max_position_embeddings=128)).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def test_author_name_is_the_most_frequent_capitalised_sequence():
    rows = [
        "What gender is author Basil Mahfouz Al-Kuwaiti? Author Basil Mahfouz Al-Kuwaiti is male.",
        "In which city was Basil Mahfouz Al-Kuwaiti born? Basil Mahfouz Al-Kuwaiti was born in Kuwait City.",
        "Which genre does Basil Mahfouz Al-Kuwaiti write? He writes Islamic literature.",
    ]
    assert rocr.extract_author_name(rows) == "Basil Mahfouz Al-Kuwaiti"
    # a possessive, a row that cuts the name short, a name followed by a sentence start
    assert rocr.extract_author_name(["Nikolai Abilov's books. Nikolai Abilov wrote.", "Who is Nikolai Abilov? He is an author.", "Nikolai Abilov was born."]) == "Nikolai Abilov"
    assert rocr.extract_author_name(["Basil Mahfouz Al-Kuwaiti. He writes.", "Basil Mahfouz Al-Kuwaiti. She", "Basil Mahfouz wrote it.", "Basil Mahfouz Al-Kuwaiti is male."]) == "Basil Mahfouz Al-Kuwaiti"
    names = rocr.forget_author_names(rows * 2, rows * 2, qa_per_author=3)
    assert names == ["Basil Mahfouz Al-Kuwaiti"] * 2


def test_neighbor_names_takes_train_rank_one(tmp_path):
    f = tmp_path / "nb.csv"
    f.write_text("author_id,neighbor_name,split,train_rank\n199,A One,train,0\n199,B Two,train,1\n199,C Three,heldout,1\n200,D Four,train,1\n")
    assert rocr.neighbor_names(str(f), [199, 200]) == ["B Two", "D Four"]
    with pytest.raises(ValueError):
        rocr.neighbor_names(str(f), [201])


def test_last_name_token_and_prompts(setup):
    tok, _ = setup
    ps = rocr.chat_prompts(tok, "Jane Doe")
    assert len(ps) == 6 and all(p.endswith("Answer:") for p in ps)
    for p in ps:
        i = rocr.last_name_token(tok, p, "Jane Doe")
        assert tok.decode(tok(p, add_special_tokens=False)["input_ids"][i]) == "Doe"


def test_null_space_projector_is_a_projector_onto_the_small_eigenvalues():
    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(16, 16))
    evals = torch.cat([torch.ones(10), torch.full((6,), 1e-3)])
    p = rocr.null_space_projector(q @ torch.diag(evals) @ q.T, 2e-2)
    assert torch.allclose(p, p.T, atol=1e-5) and torch.allclose(p @ p, p, atol=1e-5)
    assert round(float(torch.trace(p))) == 6
    assert torch.allclose(p @ q[:, :1], torch.zeros(16, 1), atol=1e-4)  # a direction with a large eigenvalue is removed


def test_ridge_update_maps_the_key_to_the_residual_without_projection():
    torch.manual_seed(0)
    k, r = torch.randn(64, 2), torch.randn(32, 2)
    delta = rocr.ridge_update(k, r, torch.eye(64), l2=1e-4)
    assert delta.shape == (32, 64)
    assert torch.allclose(delta @ k, r, atol=1e-2)


def test_apply_rocr_moves_the_last_layer_hidden_state_towards_the_target(setup):
    tok, model = setup
    layers = [1, 2, 3]
    forget, target = ["Basil Mahfouz", "Jane Doe"], ["Mubarak Yaqub", "Mubarak Yaqub"]
    ident = {l: torch.eye(64) for l in layers}

    def gap():
        z = rocr.hidden_at(model, tok, rocr.chat_prompts(tok, "Mubarak Yaqub")[:1], ["Mubarak Yaqub"], 3, "hidden")[0]
        cur = torch.stack([rocr.hidden_at(model, tok, [f"{n}."], [n], 3, "hidden")[0] for n in forget])
        return (cur - z).norm(dim=1).mean().item()

    before = gap()
    w0 = {l: rocr.get_out_proj(model, l).weight.clone() for l in layers}
    log = rocr.apply_rocr(model, tok, forget, target, layers, ident, l2=1e-3)
    try:
        assert gap() < 0.75 * before  # tiny random net, residual shared over 3 layers, chat-template key vs bare-prompt state
        assert len(log["update_norm"]) == 3 and all(u > 0 for u in log["update_norm"])
    finally:
        for l in layers:
            rocr.get_out_proj(model, l).weight.copy_(w0[l])
    with pytest.raises(ValueError):
        rocr.apply_rocr(model, tok, forget, target, [1, 3], ident)
