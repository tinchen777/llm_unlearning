"""A2 shadow-probe attack (scripts/analysis/spa.py) on synthetic activations: a shared direction transfers, an unrelated one does not, the layer is chosen on the shadow data only. No network, no GPU.
    python -m pytest tests/test_spa.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "scripts" / "analysis"))

import spa

N_L, D, N_T = 4, 24, 10


def make_sets(with_r=False):
    S = [[f"S{2 * g}", f"S{2 * g + 1}"] for g in range(5)]
    sets = {"S_groups": S, "F": [f"F{i}" for i in range(10)], "H_A": [f"A{i}" for i in range(10)], "H_B": [f"B{i}" for i in range(10)], "templates": [f"t{i}" for i in range(N_T)]}
    if with_r:
        sets.update(R_A=[f"RA{i}" for i in range(10)], R_B=[f"RB{i}" for i in range(10)])
    return spa.validate_sets(sets)


def make_acts(sets, rng, forgotten, direction, layer_signal=2, strength=2.5, noise=1.0, known=(), known_dir=None, known_layer=1):
    names = spa.all_names(sets)
    X = rng.normal(0, noise, (len(names), N_T, N_L, D)).astype(np.float32)
    for n in forgotten:
        X[names.index(n), :, layer_signal, :] += strength * direction
    for n in known:
        X[names.index(n), :, known_layer, :] += strength * known_dir
    return {"X": X, "names": names, "idx": {n: i for i, n in enumerate(names)}, "name_tokens": np.array([2] * len(names)), "prob": {}}


def unit(rng):
    d = rng.normal(size=D)
    return d / np.linalg.norm(d)


def test_a_shared_direction_transfers_and_the_layer_is_found_on_the_shadow_data():
    rng = np.random.default_rng(0)
    sets = make_sets()
    d = unit(rng)
    shadows = [make_acts(sets, rng, sets["S_groups"][k], d) for k in range(5)]
    target = make_acts(sets, rng, sets["F"], d)
    res = spa.run_attack(shadows, target, sets, kinds=("lr", "dm"), boot=50)
    assert set(res["results"]) == {"P1-spec/lr", "P1-spec/dm"}           # version-1 sets (no R): only the spec probe
    for kind in ("lr", "dm"):
        r = res["results"][f"P1-spec/{kind}"]
        assert r["selected_layer"] == 2 and r["cv_auc_selected"] > 0.9
        t = r["pools"]["Q1"]
        assert t["auc"] > 0.9 and t["auc"] > t["null95"] and t["posthoc_best_layer"] == 2


def test_an_unrelated_direction_does_not_transfer_but_the_shadow_probe_works():
    rng = np.random.default_rng(1)
    sets = make_sets()
    d1 = unit(rng)
    d2 = rng.normal(size=D); d2 -= d1 * (d2 @ d1); d2 /= np.linalg.norm(d2)
    shadows = [make_acts(sets, rng, sets["S_groups"][k], d1) for k in range(5)]
    target = make_acts(sets, rng, sets["F"], d2)
    r = spa.run_attack(shadows, target, sets, kinds=("dm",), boot=50)["results"]["P1-spec/dm"]
    assert r["cv_auc_selected"] > 0.9 and r["pools"]["Q1"]["auc"] < 0.8


def test_no_signal_at_all_stays_inside_the_null_band():
    rng = np.random.default_rng(2)
    sets = make_sets()
    d = unit(rng)
    shadows = [make_acts(sets, rng, sets["S_groups"][k], d) for k in range(5)]
    target = make_acts(sets, rng, [], d)                     # the control: nobody carries a trace
    t = spa.run_attack(shadows, target, sets, kinds=("dm",), boot=100)["results"]["P1-spec/dm"]["pools"]["Q1"]
    assert t["auc"] <= max(t["null95"], 0.75)


def _known_world(rng, sets, edit: bool):
    """Every author of the TOFU training data (S, F, R) is 'known' at layer 1 (= before the edit layer, identical to the original model); the edit leaves a direction at layer 3
    on S_k in shadow k and on F (edited by the provider, in the target AND in the shadows, which are edits of M')."""
    dk, de = unit(rng), unit(rng)
    S = [n for g in sets["S_groups"] for n in g]
    known = S + sets["F"] + sets["R_A"] + sets["R_B"]
    mk = lambda forgotten: make_acts(sets, rng, forgotten if edit else [], de, layer_signal=3, known=known, known_dir=dk, known_layer=1)
    return [mk(list(sets["S_groups"][k]) + sets["F"]) for k in range(5)], mk(sets["F"])


def test_familiarity_alone_is_caught_by_Q1_but_not_by_the_fingerprint_probe_on_Q2():
    rng = np.random.default_rng(3)
    sets = make_sets(with_r=True)
    shadows, target = _known_world(rng, sets, edit=False)      # NO edit trace anywhere: F differs from strangers only because it is a known entity
    res = spa.run_attack(shadows, target, sets, kinds=("dm",), boot=60, edit_layers=[3])["results"]
    assert res["P0-familiarity/dm"]["pools"]["Q1"]["auc"] > 0.9      # recognition alone separates F from strangers ...
    assert res["P1-spec/dm"]["pools"]["Q1"]["auc"] > 0.9              # ... and the spec probe picks it up although there is no trace to learn
    q2 = res["P2-fingerprint/dm"]["pools"]["Q2"]
    assert q2["auc"] < 0.8 and q2["auc"] <= max(q2["null95"], 0.8)    # among known authors there is nothing to find


def test_fingerprint_probe_finds_the_edit_layer_and_is_flat_before_it():
    rng = np.random.default_rng(4)
    sets = make_sets(with_r=True)
    shadows, target = _known_world(rng, sets, edit=True)
    res = spa.run_attack(shadows, target, sets, kinds=("dm",), boot=60, edit_layers=[3])["results"]
    r = res["P2-fingerprint/dm"]
    assert r["selected_layer"] == 3
    assert r["pools"]["Q2"]["auc"] > 0.9 and r["pools"]["Q2"]["auc"] > r["pools"]["Q2"]["null95"]
    assert abs(r["pools"]["Q2"]["pre_edit_mean_auc"] - 0.5) < 0.2     # before the edit layer F and R_B are both just 'known'
    assert res["P0-familiarity/dm"]["pools"]["Q2"]["auc"] < 0.8        # familiarity cannot tell F from retained authors


def test_sets_must_not_overlap_and_the_signed_evaluation_respects_the_direction():
    bad = make_sets(); bad["H_B"][0] = bad["H_A"][0]
    with pytest.raises(ValueError):
        spa.validate_sets(bad)
    bad = make_sets(with_r=True); bad["R_B"][0] = bad["F"][0]
    with pytest.raises(ValueError):
        spa.validate_sets(bad)
    hi = spa.evaluate_transfer([5, 6, 7, 8], [0, 1, 2, 1, 0, 1], k=2, boot=20)
    lo = spa.evaluate_transfer([0, 1, 0, 1], [5, 6, 7, 8, 6, 7], k=2, boot=20)
    assert hi["auc"] > 0.99 and lo["auc"] < 0.01              # a LOW score is not a detection (a two-sided |z| statistic would call both a detection)


def test_collect_acts_takes_the_last_prompt_token_of_every_layer_and_ignores_the_padding():
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    words = {"What", "was", "born", "Jane", "Doe", "Adam", "Bob", "?", "A", "User", "system", "who", "is", "Eve"}
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, **{w: i + 3 for i, w in enumerate(sorted(words))}}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    tok.chat_template = "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}{% if add_generation_prompt %}A {% endif %}"
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=16, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2,
                                         num_key_value_heads=2, max_position_embeddings=64)).eval()
    names = ["Jane Doe", "Adam", "Eve Doe"]                       # different lengths -> left padding is exercised
    templates = ["What was {name} born ?", "who is {name} ?"]
    x = spa.collect_acts(model, tok, names, templates, batch_size=4)
    assert x.shape == (3, 2, 3, 16)
    one = spa.collect_acts(model, tok, ["Adam"], templates[1:], batch_size=1)    # the SHORTEST prompt of the batch: it is left-padded there
    assert np.allclose(one[0, 0], x[1, 1], atol=1e-4)           # the same prompt alone: padding does not change the activation
    with torch.no_grad():
        import sample_consistency as sc
        enc = tok(sc.build_prompts(tok, [templates[0].format(name=names[0])]), return_tensors="pt", add_special_tokens=False)
        hs = model.model(**enc, output_hidden_states=True).hidden_states
    assert np.allclose(x[0, 0, 0], hs[1][0, -1].numpy(), atol=1e-4)        # layer 0 output = hidden_states[1]


@pytest.mark.parametrize("method,layers", [("act", "12,13,14"), ("dist", "19,20,21")])
def test_shadow_edit_commands_are_well_formed(method, layers):
    """Dry-run of spa_lib.sh: the commands handed to r_uv / select_layer / w_down (regression: `wdown.layers=[[12,13,14]]`, empty task names)."""
    import subprocess
    cmd = f'cd {root} && source scripts/exp/_clean_env.sh && export GPU=0 && source scripts/exp/spa_lib.sh && DRY_RUN=1 spa_edit_near 3B {method} saves/spa/models/m 2 real'
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=True).stdout
    py = [l for l in out.splitlines() if l.startswith("python ")]
    assert len(py) == 3 and "r_uv.py" in py[0] and "select_layer.py" in py[1] and "w_down.py" in py[2]
    assert f"wdown.layers=[{layers}]" in py[2] and "[[" not in out and f"select.layers=[{layers}]" in py[1]
    for l in py:
        assert "task_name=spa/3B_" + method + "_real_g2 " in l + " " and "model.pretrained.name_or_path=saves/spa/models/m" in l
        assert 'hf_args.split="train[80:120]"' in l and "row_filter.author_id=[185,186]" in l
    assert "basic.author_offset=185" in py[0]
    assert ("wdown.kl.enabled=true" in py[2]) == (method == "dist")


def test_strength_table_from_npz(tmp_path, capsys):
    import json as _json
    sets = make_sets(with_r=True)
    names = spa.all_names(sets)
    probs = {n: (0.1 if n in sets["S_groups"][0] else 0.4) for n in names}
    f = tmp_path / "shadow0.npz"
    np.savez(f, acts=np.zeros((len(names), 1, 1, 2), np.float16), names=np.array(names), templates=np.array(["t"]), name_tokens=np.ones(len(names), int), model="m", prob=np.array(_json.dumps(probs)))
    sp = tmp_path / "sets.json"; sp.write_text(_json.dumps(sets))
    class A: pass
    a = A(); a.sets = str(sp); a.npz = [str(f)]; a.out = str(tmp_path / "t.md")
    spa.cmd_strength(a)
    out = (tmp_path / "t.md").read_text()
    assert "0.100" in out.splitlines()[-1].split("|")[-2] and "0.400" in out.splitlines()[-1]


def test_p0_command_on_a_model_where_every_training_author_is_known(tmp_path):
    """P0 without shadows: F, R are 'known' at layer 1 of the target (recognition survives before the edit layer 3) -> Q1 high, Q2 low; on the retain-like model F is a stranger -> Q1 ~ chance."""
    import json as _json
    rng = np.random.default_rng(5)
    sets = make_sets(with_r=True)
    dk = unit(rng)
    mprime = make_acts(sets, rng, [], dk, known=sets["F"] + sets["R_A"] + sets["R_B"], known_dir=dk, known_layer=1)
    retain = make_acts(sets, rng, [], dk, known=sets["R_A"] + sets["R_B"], known_dir=dk, known_layer=1)
    paths = {}
    for nm, acts in (("mprime", mprime), ("retain", retain)):
        p = tmp_path / f"{nm}.npz"
        np.savez(p, acts=acts["X"].astype(np.float16), names=np.array(acts["names"]), templates=np.array(sets["templates"]), name_tokens=acts["name_tokens"], model=nm, prob=np.array("{}"))
        paths[nm] = p
    sp = tmp_path / "sets.json"; sp.write_text(_json.dumps(sets))
    class A: pass
    a = A(); a.sets = str(sp); a.acts = [f"Mprime={paths['mprime']}#3", f"retain={paths['retain']}"]; a.out = str(tmp_path / "p0")
    a.kinds = ["dm"]; a.C = 0.01; a.k = 2; a.boot = 60; a.seed = 0
    spa.cmd_p0(a)
    res = _json.loads((tmp_path / "p0" / "summary.json").read_text())
    m, r = res["Mprime"]["results"]["P0-familiarity/dm"], res["retain"]["results"]["P0-familiarity/dm"]
    assert m["pools"]["Q1"]["auc"] > 0.9 and m["pools"]["Q2"]["auc"] < 0.8 and m["pools"]["Q1"]["pre_edit_mean_auc"] is not None
    assert r["pools"]["Q1"]["auc"] < 0.8
    assert "| Mprime | dm |" in (tmp_path / "p0" / "summary.md").read_text()
    assert set(m["rules"]) == {"cv", "cv_deepest", "pre_edit_mean"} and set(r["rules"]) == {"cv", "cv_deepest"}     # edited layer 3 = the last one: no post-edit layer
    assert m["rules"]["pre_edit_mean"]["layers"] == [0, 1, 2] and m["rules"]["pre_edit_mean"]["pools_length_corrected"]["Q1"]["auc"] > 0.9
    assert "| Mprime | dm | pre_edit_mean | L0-2 |" in (tmp_path / "p0" / "summary.md").read_text()


# ----------------------------------------------------------------------------- layer rules and name-length control (TODO/exp/21)
def make_acts_multi(sets, rng, n_layers, signals, noise=1.0, strength=2.5, ntok=None):
    """signals: [(names, layers, direction)]; ntok: {name: number of name tokens} (default 2 for everybody)."""
    names = spa.all_names(sets)
    X = rng.normal(0, noise, (len(names), N_T, n_layers, D)).astype(np.float32)
    for who, ls, d in signals:
        for n in who:
            for l in ls:
                X[names.index(n), :, l, :] += strength * d
    nt = np.array([2 if ntok is None else ntok[n] for n in names])
    return {"X": X, "names": names, "idx": {n: i for i, n in enumerate(names)}, "name_tokens": nt, "prob": {}}


def test_rule_layers_and_length_residual():
    r = spa.rule_layers([0.5, 1.0, 0.997, 1.0, 0.9, 0.996, 0.8], range(7), [1, 2])
    assert r == {"cv": [1], "cv_deepest": [5], "post_edit_mean": [3, 4, 5, 6], "pre_edit_mean": [0]}
    assert set(spa.rule_layers([0.5, 1.0], range(2), [])) == {"cv", "cv_deepest"}
    res, slope = spa.length_residual([1.0, 2.0, 3.0, 5.0], [1, 2, 3, 4])
    assert abs(slope - 1.3) < 1e-9 and abs(res.sum()) < 1e-9
    same, s0 = spa.length_residual([1.0, 2.0], [3, 3])
    assert s0 is None and list(same) == [1.0, 2.0]
    assert spa.fmt_layers([4, 5, 6, 9]) == "L4-6, L9"


def test_cv_picks_the_valley_next_to_the_edit_but_the_deepest_and_post_edit_rules_find_the_fingerprint():
    """CV = 1.00 on every layer from the first edited one on; at the edited layers 1-3 the shadow edit (d_sh) and the provider's edit (d_pr) differ (the 'valley'), downstream (4-7) both leave d."""
    rng = np.random.default_rng(10)
    sets = make_sets()
    d, d_sh = unit(rng), unit(rng)
    d_pr = rng.normal(size=D); d_pr -= d_sh * (d_pr @ d_sh); d_pr /= np.linalg.norm(d_pr)          # the provider's edit at the edited layers: orthogonal to the shadow's
    shadows = [make_acts_multi(sets, rng, 8, [(sets["S_groups"][k], [1, 2, 3], d_sh), (sets["S_groups"][k], range(4, 8), d)]) for k in range(5)]
    target = make_acts_multi(sets, rng, 8, [(sets["F"], [1, 2, 3], d_pr), (sets["F"], range(4, 8), d)])
    r = spa.run_attack(shadows, target, sets, kinds=("dm",), boot=60, edit_layers=[1, 2, 3])["results"]["P1-spec/dm"]
    ru = r["rules"]
    assert set(ru) == {"cv", "cv_deepest", "post_edit_mean", "pre_edit_mean"}
    assert ru["cv"]["layers"] == [r["selected_layer"]] == [1] and ru["cv"]["pools"]["Q1"]["auc"] == r["pools"]["Q1"]["auc"]     # cv = the old rule, unchanged
    assert ru["cv"]["pools"]["Q1"]["auc"] < 0.8                                                    # ... which lands in the valley and misses the trace
    assert ru["cv_deepest"]["layers"] == [7] and ru["post_edit_mean"]["layers"] == [4, 5, 6, 7] and ru["pre_edit_mean"]["layers"] == [0]
    for rule in ("cv_deepest", "post_edit_mean"):
        for key in ("pools", "pools_length_corrected"):
            t = ru[rule][key]["Q1"]
            assert t["auc"] > 0.9 and t["auc"] > t["null95"]


def test_pre_edit_recognition_only_is_seen_by_pre_edit_mean_and_not_by_post_edit_mean_on_Q2():
    """Every training author (S, F, R) is 'known' at layers 0-1 (before the edited layers 2-3); there is NO edit trace anywhere."""
    rng = np.random.default_rng(11)
    sets = make_sets(with_r=True)
    dk = unit(rng)
    known = [n for g in sets["S_groups"] for n in g] + sets["F"] + sets["R_A"] + sets["R_B"]
    mk = lambda: make_acts_multi(sets, rng, 6, [(known, [0, 1], dk)])
    shadows, target = [mk() for _ in range(5)], mk()
    ru = spa.run_attack(shadows, target, sets, kinds=("dm",), boot=60, edit_layers=[2, 3])["results"]["P1-spec/dm"]["rules"]
    assert ru["pre_edit_mean"]["layers"] == [0, 1] and ru["post_edit_mean"]["layers"] == [4, 5]
    assert ru["pre_edit_mean"]["pools_length_corrected"]["Q1"]["auc"] > 0.9                        # F vs strangers: recognition
    q2 = ru["post_edit_mean"]["pools_length_corrected"]["Q2"]
    assert q2["auc"] < 0.75 and q2["auc"] <= max(q2["null95"], 0.75)                                # F vs retained authors after the edit: nothing (no detection)


def test_a_score_driven_by_name_length_alone_is_removed_by_the_length_control():
    """No trace, only a direction proportional to the number of name tokens (in every layer); the positives (S, F) have shorter names than the strangers (H_A, H_B)."""
    rng = np.random.default_rng(12)
    sets = make_sets()
    short = [n for g in sets["S_groups"] for n in g] + sets["F"]
    ntok = {n: int(rng.integers(2, 4)) if n in short else int(rng.integers(3, 6)) for n in spa.all_names(sets)}
    dl = unit(rng)

    def mk():
        a = make_acts_multi(sets, rng, 4, [], ntok=ntok)
        for i, n in enumerate(a["names"]):
            a["X"][i] += 1.2 * (ntok[n] - 3.5) * dl
        return a
    shadows, target = [mk() for _ in range(5)], mk()
    ru = spa.run_attack(shadows, target, sets, kinds=("dm",), boot=60)["results"]["P1-spec/dm"]["rules"]
    assert set(ru) == {"cv", "cv_deepest"}                                                           # no edited layers -> only the two CV rules
    for rule in ("cv", "cv_deepest"):
        assert ru[rule]["pools"]["Q1"]["auc"] > 0.85 and ru[rule]["name_token_corr"] < -0.5
        assert abs(ru[rule]["pools_length_corrected"]["Q1"]["auc"] - 0.5) < 0.25 and abs(ru[rule]["name_token_corr_corrected"]) < 0.3


def test_layer_rule_table_in_the_attack_markdown():
    rng = np.random.default_rng(13)
    sets = make_sets()
    d = unit(rng)
    shadows = [make_acts(sets, rng, sets["S_groups"][k], d) for k in range(5)]
    res = spa.run_attack(shadows, make_acts(sets, rng, sets["F"], d), sets, kinds=("dm",), boot=20, edit_layers=[1])
    md = spa.to_markdown(res, "t")
    assert "## Layer rules" in md and "| P1-spec/dm | post_edit_mean | L2-3 |" in md and "| P1-spec/dm | pre_edit_mean | L0 |" in md
