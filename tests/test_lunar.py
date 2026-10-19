"""CPU tests of LUNAR (src/METHOD/LUNAR/) and of the pieces it shares with NEAR (src/METHOD/_common/).

Tiny random Llama, synthetic prompts, no network / GPU / hydra:  python -m pytest tests/test_lunar.py -q
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch
from omegaconf import OmegaConf
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from model.activations import collect_activations, get_out_proj
from model.edit import apply_w_down, build_shift, collect_mlp_tokens
from METHOD._common.activations import unlearning_vector
from METHOD._common.editing import resolve_candidate_layers, train_w_down_layers
from METHOD.LUNAR._utils.embedding import layer_score, rank_layers, similarity_to_list
from METHOD.LUNAR._utils.selection import config_dir, run_dir, selection_path
from utils.config import TrackingConfig

N_LAYERS, HIDDEN = 4, 32
POSITIONS = [-3, -2, -1]


def make_prompts(n, seed, lo=6, hi=12):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(1, 64, (int(torch.randint(lo, hi, (1,), generator=g)),), generator=g) for _ in range(n)]


def make_batches(prompts, bs):
    batches = []
    for s in range(0, len(prompts), bs):
        chunk = prompts[s:s + bs]
        T = max(len(p) for p in chunk)
        ids, am = torch.zeros(len(chunk), T, dtype=torch.long), torch.zeros(len(chunk), T, dtype=torch.long)
        for i, p in enumerate(chunk):
            ids[i, T - len(p):], am[i, T - len(p):] = p, 1
        batches.append({"input_ids": ids, "attention_mask": am})
    return batches


def test_lunar_global_r_uv_is_reproduced_by_the_w_down_edit_on_forget_tokens_and_retain_stays_put():
    """The LUNAR pipeline on a tiny model with the pieces the scripts use: ONE r_UV = mean(ref) - mean(forget) for all forget
    rows (`all` shift: the same vector on every token), W_down of one layer trained in closed form on
    [forget tokens -> y + r_UV, retain tokens -> y]; afterwards the forget activations moved by ~ r_UV, retain did not."""
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=64, hidden_size=HIDDEN, intermediate_size=2048, num_hidden_layers=N_LAYERS,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)).eval()
    layer, coeff, ref_position = 1, 1.0, -1
    fb, rb, refb = (make_batches(make_prompts(8, 10), 4), make_batches(make_prompts(24, 12), 8),
                    make_batches(make_prompts(12, 11), 4))

    forget_mean, layers = collect_activations(model, fb, positions=POSITIONS, layers=[layer])
    ref_mean, _ = collect_activations(model, refb, positions=POSITIONS, layers=[layer])
    r_uv = unlearning_vector(ref_mean, forget_mean)  # [n_pos, n_layers, d]
    r = r_uv[:, 0, :].float()

    def shift_fn(batch, l, attention_mask):
        rr = r.unsqueeze(0).expand(attention_mask.shape[0], -1, -1)  # the SAME r_UV for every row
        return coeff * build_shift(rr, attention_mask, POSITIONS, "all", ref_position)

    f_tok = {layer: collect_mlp_tokens(model, fb, [layer], shift_fn=shift_fn)[layer]}
    r_tok = {layer: collect_mlp_tokens(model, rb, [layer])[layer]}
    wd = OmegaConf.create({"ridge": 1e-6, "forget_weight": 1.0, "retain_weight": 1.0})
    per_layer, new_weights = train_w_down_layers(model, [layer], f_tok, r_tok, wd, seed=0, solver="closed_form")
    assert per_layer[layer]["forget_mse_after"] < 1e-2 * per_layer[layer]["forget_mse_before"]
    assert per_layer[layer]["retain_mse_after"] < 1e-2 * per_layer[layer]["forget_mse_before"]

    retain_before, _ = collect_activations(model, rb, positions=POSITIONS, layers=[layer])
    apply_w_down(model, layer, new_weights[layer])
    forget_after, _ = collect_activations(model, fb, positions=POSITIONS, layers=[layer])
    retain_after, _ = collect_activations(model, rb, positions=POSITIONS, layers=[layer])
    achieved = (forget_after[:, 0] - forget_mean[:, 0]).flatten()
    intended = (coeff * r[POSITIONS.index(ref_position)].expand_as(r)).flatten().double()
    cos = torch.nn.functional.cosine_similarity(achieved, intended, dim=0).item()
    assert cos > 0.99 and 0.9 < (achieved.norm() / intended.norm()).item() < 1.1, f"cos={cos:.3f}"
    assert ((retain_after - retain_before).norm() / retain_before.norm()).item() < 0.01


def test_train_w_down_layers_runs_adam_and_rejects_unknown_solvers():
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=64, hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=N_LAYERS,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)).eval()
    batches = make_batches(make_prompts(8, 1), 4)
    tok = {1: collect_mlp_tokens(model, batches, [1])[1]}
    wd = OmegaConf.create({"lr": 1e-3, "lr_gamma": 0.9, "epochs": 2, "batch_size": 16, "weight_decay": 0.0,
                           "forget_weight": 1.0, "retain_weight": 1.0})
    per_layer, new_weights = train_w_down_layers(model, [1], tok, tok, wd, seed=0, solver="adam")
    assert set(per_layer[1]) >= {"forget_mse_before", "forget_mse_after", "retain_mse_after", "history"}
    assert new_weights[1].shape == get_out_proj(model, 1).weight.shape
    with pytest.raises(ValueError):
        train_w_down_layers(model, [1], tok, tok, wd, seed=0, solver="sgd")


def test_resolve_candidate_layers():
    avail = list(range(16))
    assert resolve_candidate_layers(OmegaConf.create({}), avail) == avail
    assert resolve_candidate_layers(OmegaConf.create({"layers": [3, 5]}), avail) == [3, 5]
    assert resolve_candidate_layers(OmegaConf.create({"layer_fraction": [0.5, 0.75]}), avail) == [8, 9, 10, 11, 12]
    with pytest.raises(ValueError):
        resolve_candidate_layers(OmegaConf.create({"layers": [99]}), avail)


def test_s1_minus_s2_prefers_the_layer_whose_answers_look_like_the_desired_responses():
    e = torch.eye(4)
    desired, unrelated = e[:1], e[1:2]  # responses along axis 0 are "desired", along axis 1 "unrelated"
    good = layer_score(e[:1].repeat(3, 1), desired, unrelated)
    bad = layer_score(e[1:2].repeat(3, 1), desired, unrelated)
    assert good == pytest.approx({"s1": 1.0, "s2": 0.0, "score": 1.0}) and bad["score"] == pytest.approx(-1.0)
    mix = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    targets = torch.stack([e[0], e[2]])
    assert similarity_to_list(mix, targets, "mean").tolist() == pytest.approx([0.5, 0.5])
    assert similarity_to_list(mix, targets, "max").tolist() == pytest.approx([1.0, 1.0])
    with pytest.raises(ValueError):
        similarity_to_list(mix, targets, "median")


def test_rank_layers_orders_by_score_and_drops_degenerate_layers():
    per_layer = {
        4: {"score": 0.1, "degenerate_rate": 0.0},
        5: {"score": 0.9, "degenerate_rate": 0.5},  # best score but gibberish
        6: {"score": 0.4, "degenerate_rate": 0.0},
    }
    assert rank_layers(per_layer, max_degenerate=0.1) == [6, 4]
    assert rank_layers(per_layer, max_degenerate=0.9) == [5, 6, 4]
    assert rank_layers({1: {"score": 0.0, "degenerate_rate": 1.0}}, max_degenerate=0.1) == [1]  # none passes -> keep all


def test_output_layout_groups_the_files_by_setting_and_run():
    out = Path("saves/LUNAR/t")
    assert config_dir(out, 1.0, "all") == out / "c1_all"
    assert config_dir(out, 2.5, "eoi") == out / "c2.5_eoi"
    assert selection_path(out, 1.0, "all") == out / "c1_all" / "layer_selection.json"
    assert run_dir(out, 1.0, "all", [16], "adam") == out / "c1_all" / "l16"
    assert run_dir(out, 1.0, "all", [10, 11], "closed_form") == out / "c1_all" / "l10-11_cf"



def test_prompt_only_reference_dataset_has_no_answer_column(tmp_path):
    """configs/data/datasets/LUNAR/*: `answer_key: null` (instruction-only prompts) must tokenise to prompt-only samples."""
    from data.datasets.qa import QADataset

    words = ["Q:", "A:", "how", "to", "do", "bad", "things", "?"]
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, **{w: i + 3 for i, w in enumerate(words)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    path = tmp_path / "harmful.json"
    path.write_text(json.dumps([{"instruction": "how to do bad things ?"}, {"instruction": "how to do things ?"}]))
    hf = TrackingConfig(OmegaConf.create({"path": "json", "data_files": str(path), "split": "train"}))
    template = TrackingConfig(OmegaConf.create({"apply_chat_template": False, "user_start_tag": "Q: ",
                                                "user_end_tag": " ", "asst_start_tag": "A: ", "asst_end_tag": ""}))
    ds = QADataset(hf_args=hf, template_args=template, tokenizer=tokenizer, question_key="instruction",
                   answer_key=None, max_length=64, predict_with_generate=True)
    assert len(ds) == 2
    # the prompt is "Q: how to do bad things ? A:" (the generation prompt), no answer tokens in `input_ids`
    assert ds[0]["input_ids"] == [3, 5, 6, 7, 8, 9, 10, 4]
