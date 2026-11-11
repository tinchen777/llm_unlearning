"""A1 v2 (scripts/analysis/rank_attack_v2.py): templates from retained data, logit features, minority-pool evaluation with bootstrap interval and null band, folder analysis. No network.
    python -m pytest tests/test_rank_attack_v2.py -q
"""
import json
import random
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "scripts" / "analysis"))

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

import rank_attack as ra
import rank_attack_v2 as v2


def test_templates_from_qa_keep_the_frequent_forms_and_drop_the_names():
    authors = ["Jane Doe", "Adam Smith Brown", "Eve Fox Hart"]
    forms = ["In which city was {name} born?", "What genre does {name} write?", "Who is {name}?"]
    qs, ans = [], []
    for a in authors:
        for f in forms:
            qs.append(f.format(name=a))
            ans.append(f"{a} is a writer.")
        qs.append(f"What did {a} write in the year {len(a)}?")          # a form used by one author only
        ans.append(f"{a} wrote a book.")
    out = v2.templates_from_qa(qs, ans, qa_per_author=4, n=10, min_count=2)
    ts = [t for t, _ in out]
    assert set(ts) == set(forms) and all(c == 3 for _, c in out)
    assert all("Jane" not in t and "{name}" in t for t in ts)


def test_resolve_templates_builtin_and_json(tmp_path):
    t, origin = v2.resolve_templates("builtin", "retain95", 20)
    assert t == ra.TEMPLATES and origin == "builtin"
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"templates": ["Who is {name}?"]}))
    assert v2.resolve_templates(str(p), "retain95", 20)[0] == ["Who is {name}?"]


def _feat(**kw):
    base = {f: 0.0 for f in ra.L1 + ra.L2 + ra.L5 + v2.L6}
    base.update({"words": 30.0, "self_agreement": 0.5, "distinct_names": 2.0, "modal_name_share": 0.2, "cross_question_name_share": 0.2,
                 "greedy_logprob": -1.0, "greedy_entropy": 1.0, "logprob_sd": 0.3})
    base.update(kw)
    return base


def test_evaluate_detects_a_planted_signal_and_respects_the_null():
    rng = random.Random(1)
    labels = [1] * 6 + [0] * 8
    noise = lambda: rng.gauss(0, 0.03)
    signal = [_feat(modal_name_share=0.8 + noise(), self_agreement=0.5 + noise()) for _ in range(6)]
    plain = [_feat(modal_name_share=0.2 + noise(), self_agreement=0.5 + noise()) for _ in range(8)]
    res = v2.evaluate(signal + plain, labels, k=2, boot=60, seed=0)
    p = res[v2.PRIMARY]
    assert p["auc"] > 0.95 and p["auc"] > p["null95"] + 0.1 and p["n_subsets"] == 15 and p["p_at_F"] > 0.9
    none = [_feat(modal_name_share=0.2 + noise(), self_agreement=0.5 + noise()) for _ in range(14)]
    r0 = v2.evaluate(none, labels, k=2, boot=60, seed=0)[v2.PRIMARY]
    assert 0.25 < r0["auc"] < 0.75 and r0["auc"] <= r0["null95"] + 0.05 and r0["null95"] < 0.85


def test_mean_aggregator_and_single_features():
    labels = [1, 1] + [0] * 8
    feats = [_feat(distinct_names=9.0)] * 2 + [_feat(distinct_names=2.0 + 0.1 * i) for i in range(8)]
    assert v2.pool_auc(feats, labels, ["distinct_names"], "mean")["auc"] == 1.0
    assert v2.single_feature_aucs(feats, labels, k=2)["distinct_names"] == 1.0


def test_analyze_dir_writes_a_summary(tmp_path):
    rng = random.Random(0)
    for m, shift in (("retain", 0.0), ("NEAR-Act", 0.4)):
        feats = [_feat(modal_name_share=0.2 + shift * (i < 4) + rng.gauss(0, 0.03)) for i in range(12)]
        (tmp_path / f"{m}.json").write_text(json.dumps({"features": feats, "labels": [1] * 4 + [0] * 8}))
    md = v2.analyze_dir(str(tmp_path), k=2, boot=40)
    assert "NEAR-Act" in md and "primary" in md and (tmp_path / "summary_k2_max.md").exists()


@pytest.fixture(scope="module")
def tiny():
    words = {"What", "Where", "was", "born", "Jane", "Doe", "Adam", "Bob", "?", "A", "User", "system", "who", "is"}
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, **{w: i + 3 for i, w in enumerate(sorted(words))}}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    tok.chat_template = "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}{% if add_generation_prompt %}A {% endif %}"
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                                         num_key_value_heads=4, max_position_embeddings=128)).eval()
    return tok, model


def test_greedy_with_scores_matches_a_manual_log_probability(tiny):
    tok, model = tiny
    prompts = ["User What was Jane born ? A", "User who is Adam ? A"]
    texts, lps, ents = v2.greedy_with_scores(model, tok, prompts, max_new_tokens=4, batch_size=2)
    assert len(texts) == len(lps) == len(ents) == 2 and all(lp < 0 for lp in lps) and all(e > 0 for e in ents)
    # the first prompt alone: the same numbers (left padding must not change the result)
    t1, lp1, e1 = v2.greedy_with_scores(model, tok, prompts[:1], max_new_tokens=4, batch_size=1)
    assert t1[0] == texts[0] and abs(lp1[0] - lps[0]) < 1e-3 and abs(e1[0] - ents[0]) < 1e-3


def test_query_and_featurize_v2_on_a_tiny_model(tiny):
    tok, model = tiny
    templates = ["What was {name} born ?", "who is {name} ?"]
    ans = v2.query_model_v2(model, tok, ["Jane Doe", "Adam Bob"], templates, n=3, temperature=1.0, max_new_tokens=4, batch_size=2, seed=0)
    assert set(ans) == {"Jane Doe", "Adam Bob"} and all(len(d["greedy_logprob"]) == 2 for d in ans.values())
    feats = v2.featurize_v2(ans)
    assert set(v2.L6) <= set(feats[0]) and set(ra.L5) <= set(feats[0])
