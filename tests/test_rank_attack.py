"""A1 black-box ranking attack (scripts/analysis/rank_attack.py): features, label-free scoring, AUC / precision@|F| / MRR, and the model side on a tiny random Llama (no network).
    python -m pytest tests/test_rank_attack.py -q
"""
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


def _feat(**kw):
    base = {f: 0.0 for f in ra.FEATURE_SETS["L1+L2+L5"]}
    base.update({"words": 30.0, "self_agreement": 0.4, "distinct_names": 2.0, "modal_name_share": 0.15, "cross_question_name_share": 0.2})
    base.update(kw)
    return base


def test_auc_and_ranking_metrics():
    assert ra.auc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert ra.auc([0.5, 0.5], [1, 0]) == 0.5
    m = ra.ranking_metrics([0.1, 0.9, 0.8, 0.2], [1, 0, 1, 0])
    assert m["p_at_F"] == 0.5 and m["mrr"] == 0.5 and abs(m["auc"] - 0.25) < 1e-9


def test_robust_z_and_two_sided_score():
    z = ra.robust_z([1.0, 1.1, 0.9, 1.0, 5.0])
    assert z[4] > 10 and abs(z[0]) < 1
    # a pool in which one candidate has too LITTLE agreement and another too MUCH: both outrank the ordinary ones
    feats = [_feat(self_agreement=0.4 + 0.01 * i) for i in range(8)] + [_feat(self_agreement=0.05), _feat(self_agreement=0.95)]
    s = ra.anomaly_scores(feats, ["self_agreement"])
    assert min(s[8], s[9]) > max(s[:8])
    # nothing varies: nothing is anomalous (and no division by zero)
    assert ra.anomaly_scores([_feat() for _ in range(6)], ra.FEATURE_SETS["L1+L2+L5"]) == [0.0] * 6


def test_candidate_features_see_a_reused_substitute_name():
    qs = [f"Where was Jane Doe born, topic {i}?" for i in range(3)]
    reused = [["Jane Doe is a student of Donald Trump."] * 4 for _ in qs]                 # ROCR-like: the same other person everywhere
    varied = [[f"Jane Doe met {n} Smith." for n in ("Adam", "Bob", "Carl", "Dan")] for _ in qs]
    f1 = ra.candidate_features(qs, [r[0] for r in reused], reused, [0.0] * 15, [False] * 3)
    f2 = ra.candidate_features(qs, [v[0] for v in varied], varied, [0.0] * 15, [False] * 3)
    assert f1["modal_name_share"] == 1.0 and f1["cross_question_name_share"] == 1.0 and f1["distinct_names"] == 1.0 and f1["self_agreement"] == 1.0
    assert f2["distinct_names"] == 4.0 and f2["modal_name_share"] == 0.25 and f2["self_agreement"] < 1.0
    # the candidate's own name (it is in the question) is not a substitute name
    assert ra.substitute_name("Jane Doe was born in Paris.", "Where was Jane Doe born?") is None


def test_attack_separates_a_planted_signal_and_isolation_forest_runs():
    labels = [1, 1] + [0] * 8
    feats = [_feat(modal_name_share=0.9, cross_question_name_share=1.0)] * 2 + [_feat(modal_name_share=0.12 + 0.01 * i) for i in range(8)]
    res = ra.attack(feats, labels, [f"c{i}" for i in range(10)])
    assert res["L1+L2+L5"]["auc"] == 1.0 and res["L1"]["auc"] == 0.5          # the refusal channel sees nothing, the sampling channel everything
    assert res["only modal_name_share"]["auc"] == 1.0
    assert res["IsolationForest(all)"]["auc"] > 0.9


@pytest.fixture(scope="module")
def tiny():
    words = {"What", "Where", "was", "born", "Jane", "Doe", "Adam", "Bob", "?", "A", "User", "system", "In", "which", "city", "and", "country", "genre", "is", "author",
             "most", "known", "for", "the", "profession", "of", "s", "father", "gender", "Can", "you", "name", "two", "books", "written", "by", "occupation", "mother",
             "awards", "has", "received", "their", "writing", "How", "would", "describe", "style", "Which", "themes", "does", "usually", "write", "about", "tell", "me", "early", "life", "'"}
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, **{w: i + 3 for i, w in enumerate(sorted(words))}}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    tok.chat_template = "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}{% if add_generation_prompt %}A {% endif %}"
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                                         num_key_value_heads=4, max_position_embeddings=256)).eval()
    return tok, model


def test_query_model_and_featurize_on_a_tiny_model(tiny):
    tok, model = tiny
    ans = ra.query_model(model, tok, ["Jane Doe", "Adam Bob"], n=3, temperature=1.0, max_new_tokens=6, batch_size=4, seed=0)
    assert set(ans) == {"Jane Doe", "Adam Bob"}
    k = len(ra.TEMPLATES)
    for d in ans.values():
        assert len(d["questions"]) == k == len(d["greedy"]) == len(d["samples"]) and all(len(s) == 3 for s in d["samples"])
    feats = ra.featurize(ans)
    assert len(feats) == 2 and set(feats[0]) == set(ra.FEATURE_SETS["L1+L2+L5"])
    res = ra.attack(feats + feats, [1, 1, 0, 0], list(ans) * 2)
    assert 0.0 <= res["L1+L2+L5"]["auc"] <= 1.0
