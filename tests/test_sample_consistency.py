import importlib.util
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location("sample_consistency", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "sample_consistency.py")
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)

QS = ["What is the full name of the author born in Kuwait City?", "What gender is author Basil Mahfouz Al-Kuwaiti?"]
TR = ["The full name is Basil Mahfouz Al-Kuwaiti.", "Basil Mahfouz Al-Kuwaiti is male."]


def test_always_same_substitute_is_flagged():
    m = sc.compute_metrics(QS, TR, [["Khalid Qais Al-Marzouq is the author."] * 4, ["Khalid Qais Al-Marzouq is male."] * 4], ks=(1, 2))
    assert m["distinct_names"] == 1.0 and m["modal_name_share"] == 1.0
    assert m["cross_question_name_share"] == 1.0          # one name for both questions
    assert m["leak_key@4"] == 0.0


def test_varied_answers_and_leak():
    varied = [["Leila Al-Sabah.", "Omar Hassan Al-Nuri wrote it.", "Nadia Karim Haddad it is.", "Sami Rahim Aziz."],
              ["Basil Mahfouz Al-Kuwaiti is male.", "He is female.", "Mona Said is male.", "Tariq Nasser Khoury is male."]]
    m = sc.compute_metrics(QS, TR, varied, ks=(1, 2))
    assert m["self_agreement"] < 0.4 and m["distinct_names"] > 2
    assert m["leak_key@1"] == 0.5 and m["leak_rouge@1"] == 0.5     # the second question leaks in the first sample
    assert m["leak_key@4"] >= m["leak_key@1"]


def test_names_in_the_question_are_not_substitutes():
    m = sc.compute_metrics(["What gender is author Basil Mahfouz Al-Kuwaiti?"], ["x"], [["Basil Mahfouz Al-Kuwaiti is male."] * 3])
    assert m["distinct_names"] == 0.0 and m["cross_question_name_share"] == 0.0


def test_rouge_helpers():
    assert sc.rouge_l_f1("a b c", "a b c") == 1.0
    assert sc.rouge_l_f1("a b c", "x y z") == 0.0
    assert sc.key_recall("it is Basil Mahfouz Al-Kuwaiti", "Basil Mahfouz Al-Kuwaiti is male.") == 1.0


def test_sample_answers_shapes_with_tiny_model():
    torch = pytest.importorskip("torch")
    tokenizers = pytest.importorskip("tokenizers")
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    words = ["<unk>", "<s>", "</s>", "<pad>", "user", "assistant", "system", "what", "is", "the", "name", "a", "b", "c", "d"]
    tk = tokenizers.Tokenizer(tokenizers.models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tk.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<pad>")
    tok.chat_template = "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    cfg = LlamaConfig(vocab_size=len(words), hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)
    model = LlamaForCausalLM(cfg)
    prompts = sc.build_prompts(tok, ["what is the name", "what is a name"], system="system")
    out = sc.sample_answers(model, tok, prompts, n=3, max_new_tokens=5, batch_size=2)
    assert len(out) == 2 and all(len(x) == 3 for x in out)
