"""CPU tests of the neighbour-redirection building blocks (src/model/redirect.py, activations.py).

Tiny random Llama, synthetic prompts, no network / GPU / hydra:
    python tests/test_redirect.py          # or: python -m pytest tests/test_redirect.py -q
Every test compares against an independent direct computation (per-sample, un-padded, plain forward).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from model.activations import (
    _group_folds, auc, capture_activations, collect_activations, collect_group_means, gather_positions, get_out_proj,
    mean_diff_auc,
)
from model.generation import answer_logprobs, is_degenerate, is_refusal, summarize_responses
from model.redirect import (
    Steerer, apply_w_down, build_shift, check_neighbors, collect_mlp_tokens, compute_r_uv, forget_authors,
    nested_neighbor_order, neighbors_by_author, selection_file_name, solve_w_down_closed_form, summarize_r_uv,
    train_w_down, w_down_mse,
)
from utils.common import IGNORE_INDEX

N_LAYERS, HIDDEN, FF = 4, 32, 64
POSITIONS = [-3, -2, -1]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=HIDDEN, intermediate_size=FF, num_hidden_layers=N_LAYERS,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    return LlamaForCausalLM(cfg).eval()


def make_prompts(n, seed, lo=6, hi=12):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(1, 64, (int(torch.randint(lo, hi, (1,), generator=g)),), generator=g) for _ in range(n)]


def make_batches(prompts, meta, bs, side="left"):
    """Padded batches like the project's collator: tensors + per-sample metadata lists."""
    batches = []
    for s in range(0, len(prompts), bs):
        chunk = prompts[s:s + bs]
        T = max(len(p) for p in chunk)
        ids = torch.zeros(len(chunk), T, dtype=torch.long)
        am = torch.zeros_like(ids)
        for i, p in enumerate(chunk):
            sl = slice(T - len(p), T) if side == "left" else slice(0, len(p))
            ids[i, sl] = p
            am[i, sl] = 1
        batches.append({"input_ids": ids, "attention_mask": am,
                        **{k: v[s:s + bs] for k, v in meta.items()}})
    return batches


def single_forward_acts(model, prompt, layer, point):
    """Activation of one un-padded prompt, all tokens: [seq, d]."""
    with torch.no_grad(), capture_activations(model, [layer], point) as cache:
        model(input_ids=prompt[None], attention_mask=torch.ones(1, len(prompt), dtype=torch.long))
        return cache[layer][0]


# ---------------------------------------------------------------------------------------------


def test_group_means_equal_per_sample_means(model):
    prompts = make_prompts(11, seed=1)
    groups = [i % 3 for i in range(11)]
    batches = make_batches(prompts, {"g": groups}, bs=4)
    means, counts, layers = collect_group_means(model, batches, lambda b: b["g"], positions=POSITIONS)
    assert layers == list(range(N_LAYERS)) and counts == {0: 4, 1: 4, 2: 3}
    for g in (0, 1, 2):
        manual = torch.stack([
            torch.stack([single_forward_acts(model, p, l, "block_out")[-3:] for l in range(N_LAYERS)], dim=1)
            for p, gg in zip(prompts, groups) if gg == g
        ]).mean(0)
        assert tuple(means[g].shape) == (3, N_LAYERS, HIDDEN)
        assert torch.allclose(means[g], manual, atol=1e-5), f"group {g}"


def test_group_means_padding_side_invariant(model):
    prompts = make_prompts(7, seed=2)
    meta = {"g": [i % 2 for i in range(7)]}
    a, _, _ = collect_group_means(model, make_batches(prompts, meta, 3, "left"), lambda b: b["g"], POSITIONS)
    b, _, _ = collect_group_means(model, make_batches(prompts, meta, 3, "right"), lambda b: b["g"], POSITIONS)
    assert all(torch.allclose(a[k], b[k], atol=1e-5) for k in a)


def test_author_mapping_and_validation():
    assert forget_authors(range(6), qa_per_author=2) == [0, 0, 1, 1, 2, 2]
    assert forget_authors([0, 19, 20], qa_per_author=20, author_offset=198) == [198, 198, 199]
    nb = neighbors_by_author([0, 0, 0, 1, 1, 1], [10, 11, 10, 20, 21, 22])  # author 0 -> {10, 11}
    assert nb == {0: [10, 11], 1: [20, 21, 22]}
    check_neighbors({0, 1}, nb, ks=[1, 2])  # ok
    with pytest.raises(ValueError, match="at least max"):
        check_neighbors({0, 1}, nb, ks=[1, 3])  # author 0 has only 2 neighbours
    with pytest.raises(ValueError, match="author_offset"):
        check_neighbors({198, 199}, nb, ks=[1])  # numbering mismatch -> tells what to fix


def test_neighbor_order_is_nested_deterministic_and_row_order_free():
    nb = {a: list(range(100 * a, 100 * a + 15)) for a in range(3)}
    o1 = nested_neighbor_order(nb, seed=0)
    o2 = nested_neighbor_order({a: list(reversed(v)) for a, v in nb.items()}, seed=0)
    assert o1 == o2, "order must not depend on how the ids are listed"
    assert nested_neighbor_order(nb, seed=1) != o1
    for a, ids in o1.items():
        assert sorted(ids) == nb[a]
        assert ids[:1] == ids[:5][:1] and ids[:5] == ids[:15][:5]  # nested by construction
    # different authors get different permutations (not all the same relative order)
    assert len({tuple(i - 100 * a for i in ids) for a, ids in o1.items()}) > 1


def test_r_uv_is_reference_minus_forget(model):
    forget = {0: torch.randn(3, N_LAYERS, HIDDEN), 1: torch.randn(3, N_LAYERS, HIDDEN)}
    neigh = {(a, n): torch.randn(3, N_LAYERS, HIDDEN) for a in (0, 1) for n in range(4)}
    order = {0: [2, 0, 3, 1], 1: [1, 3, 0, 2]}
    r = compute_r_uv(forget, neigh, order, ks=[1, 3])
    assert set(r) == {1, 3} and set(r[1]) == {0, 1}
    assert torch.allclose(r[1][0], neigh[(0, 2)] - forget[0])
    expected = torch.stack([neigh[(1, n)] for n in (1, 3, 0)]).mean(0) - forget[1]
    assert torch.allclose(r[3][1], expected, atol=1e-6)


def test_summarize_r_uv():
    torch.manual_seed(0)
    forget = {a: torch.randn(3, 2, 8) for a in (0, 1)}
    r_uv = {1: {a: torch.randn(3, 2, 8) for a in (0, 1)}, 5: {a: torch.randn(3, 2, 8) for a in (0, 1)}}
    s = summarize_r_uv(r_uv, forget, layers=[7, 9])
    assert abs(s[5]["7"]["cos_to_kmax"] - 1) < 1e-6, "K = max K is identical to itself"
    manual_norm = sum(r_uv[1][a][:, 1].norm(dim=-1).mean() for a in (0, 1)) / 2  # layer index 1 -> "9"
    assert abs(s[1]["9"]["norm"] - manual_norm.item()) < 1e-5
    manual_rel = sum((r_uv[1][a][:, 0].norm(dim=-1) / forget[a][:, 0].norm(dim=-1)).mean() for a in (0, 1)) / 2
    assert abs(s[1]["7"]["rel_norm"] - manual_rel.item()) < 1e-5
    flat = lambda a, k: r_uv[k][a][:, 0].flatten()
    manual_cos = sum(torch.nn.functional.cosine_similarity(flat(a, 1), flat(a, 5), dim=0) for a in (0, 1)) / 2
    assert abs(s[1]["7"]["cos_to_kmax"] - manual_cos.item()) < 1e-5


# ---------------------------------------------------------------------------------------------


def test_build_shift_modes():
    am = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]])  # left padded
    r = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4) + 1  # [bsz, n_pos=3, d=4]
    eoi = build_shift(r, am, POSITIONS, "eoi")
    for i in range(2):
        assert torch.equal(eoi[i, -1], r[i, 2]) and torch.equal(eoi[i, -2], r[i, 1]) and torch.equal(eoi[i, -3], r[i, 0])
        assert (eoi[i, :-3] == 0).all()
    right = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])  # right padded: last real token differs
    eoi_r = build_shift(r, right, POSITIONS, "eoi")
    assert torch.equal(eoi_r[0, 3], r[0, 2]) and torch.equal(eoi_r[0, 1], r[0, 0]) and (eoi_r[0, 4:] == 0).all()
    allm = build_shift(r, am, POSITIONS, "all", ref_position=-1)
    assert all(torch.equal(allm[i, t], r[i, 2]) for i in range(2) for t in range(6))
    with pytest.raises(ValueError):
        build_shift(r, am, POSITIONS, "nope")


@pytest.mark.parametrize("side", ["left", "right"])
def test_collect_mlp_tokens_matches_direct_computation(model, side):
    prompts = make_prompts(5, seed=3)
    layer = 2
    data = collect_mlp_tokens(model, make_batches(prompts, {}, 2, side), [layer])[layer]
    x, t = data
    assert len(x) == sum(len(p) for p in prompts) == len(t), "one row per non-padding token"
    x_ref = torch.cat([single_forward_acts(model, p, layer, "down_proj_in") for p in prompts])
    y_ref = torch.cat([single_forward_acts(model, p, layer, "mlp_out") for p in prompts])
    assert torch.allclose(x, x_ref, atol=1e-5) and torch.allclose(t, y_ref, atol=1e-5)
    # the recorded pair is consistent with the weights: W0 x == y  (so retain loss starts at ~0)
    assert w_down_mse(get_out_proj(model, layer).weight, x, t) < 1e-10


def test_collect_mlp_tokens_applies_shift_only_where_requested(model):
    prompts = make_prompts(4, seed=4)
    r = torch.randn(len(prompts), 3, HIDDEN)
    layer = 1
    batches = make_batches(prompts, {"row": list(range(4))}, 2)
    fn = lambda batch, l, am: build_shift(r[batch["row"]], am, POSITIONS, "eoi")
    x0, t0 = collect_mlp_tokens(model, batches, [layer])[layer]
    x1, t1 = collect_mlp_tokens(model, batches, [layer], shift_fn=fn)[layer]
    assert torch.equal(x0, x1)
    diff = t1 - t0
    off = 0
    for i, p in enumerate(prompts):
        n = len(p)
        assert torch.allclose(diff[off + n - 3:off + n], r[i], atol=1e-6), "last 3 tokens carry r"
        assert diff[off:off + n - 3].abs().max() < 1e-6, "other tokens untouched"
        off += n


# ---------------------------------------------------------------------------------------------


def test_train_w_down_end_to_end_effect_on_the_model():
    """Edit ONE layer and check what the whole model does: block_out of the forget prompts must move by
    ~ coeff * r_UV of their author, the retain prompts must hardly move."""
    torch.manual_seed(0)
    # d_ff must exceed the number of training tokens (~280) for a linear map to fit them (see the note below)
    cfg = LlamaConfig(vocab_size=64, hidden_size=HIDDEN, intermediate_size=2048, num_hidden_layers=N_LAYERS,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    model = LlamaForCausalLM(cfg).eval()
    layer, coeff = 1, 1.0

    f_prompts, f_meta = make_prompts(8, seed=10), {"author": [i // 4 for i in range(8)]}
    n_prompts = make_prompts(12, seed=11)  # 2 authors x 2 neighbours x 3 questions
    n_meta = {"key": [(i // 6, (i // 3) % 2) for i in range(12)]}
    retain = make_prompts(24, seed=12)
    fb, nb, rb = make_batches(f_prompts, f_meta, 4), make_batches(n_prompts, n_meta, 4), make_batches(retain, {}, 8)

    f_means, _, layers = collect_group_means(model, fb, lambda b: b["author"], POSITIONS, [layer])
    n_means, _, _ = collect_group_means(model, nb, lambda b: b["key"], POSITIONS, [layer])
    order = nested_neighbor_order({0: [0, 1], 1: [0, 1]}, seed=0)
    r_uv = compute_r_uv(f_means, n_means, order, ks=[2])[2]
    r_layer = {a: r_uv[a][:, 0, :] for a in r_uv}  # layers == [layer] -> index 0

    shift_fn = lambda batch, l, am: build_shift(torch.stack([r_layer[a] for a in batch["author"]]), am, POSITIONS, "eoi")
    f_tok = collect_mlp_tokens(model, fb, [layer], shift_fn=lambda b, l, am: coeff * shift_fn(b, l, am))[layer]
    r_tok = collect_mlp_tokens(model, rb, [layer])[layer]

    w0 = get_out_proj(model, layer).weight.detach().clone()
    f_before, r_before = w_down_mse(w0, *f_tok), w_down_mse(w0, *r_tok)
    w, hist = train_w_down(w0, f_tok, r_tok, lr=3e-3, epochs=150, batch_size=32, lr_gamma=0.98, seed=0)
    f_after, r_after = w_down_mse(w, *f_tok), w_down_mse(w, *r_tok)
    assert r_before < 1e-10, "retain targets are the original outputs: zero error before training"
    assert f_after < 0.1 * f_before, f"forget MSE {f_before:.3g} -> {f_after:.3g}"
    assert r_after < 0.1 * f_before, f"retain MSE must stay far below the initial forget error: {r_after:.3g}"
    assert len(hist) == 150 and hist[-1]["forget_mse"] < hist[0]["forget_mse"]

    # ---- write into the model, measure what the model really does
    retain_before = collect_group_means(model, rb, lambda b: [0] * len(b["input_ids"]), POSITIONS, [layer])[0][0]
    apply_w_down(model, layer, w)
    f_after_means, _, _ = collect_group_means(model, fb, lambda b: b["author"], POSITIONS, [layer])
    retain_after = collect_group_means(model, rb, lambda b: [0] * len(b["input_ids"]), POSITIONS, [layer])[0][0]
    for a in (0, 1):
        achieved = (f_after_means[a] - f_means[a]).flatten()
        intended = (coeff * r_uv[a]).flatten()
        cos = torch.nn.functional.cosine_similarity(achieved, intended, dim=0).item()
        ratio = (achieved.norm() / intended.norm()).item()
        # observed on this toy: cos ~0.999, ratio ~0.9 (under-shoots a little: the retain term pulls back)
        assert cos > 0.95 and 0.75 < ratio < 1.2, f"author {a}: cos={cos:.3f} norm ratio={ratio:.3f}"
    retain_move = ((retain_after - retain_before).norm() / retain_before.norm()).item()
    intended_move = (coeff * r_uv[0]).norm().item() / f_means[0].norm().item()
    assert retain_move < 0.15 * intended_move, f"retain moved {retain_move:.3g}, intended forget move {intended_move:.3g}"


def test_shared_prefix_tokens_have_identical_inputs(model):
    """Why `shift_mode: all` is risky: tokens of a prompt prefix shared by forget and retain prompts (system
    prompt, chat header) have BIT-IDENTICAL MLP inputs (causal attention), so forget (+shift) and retain (+0)
    would demand two different outputs for the same input. `eoi` shifts only the tokens after the question."""
    g = torch.Generator().manual_seed(7)
    prefix = torch.randint(1, 64, (6,), generator=g)
    a = torch.cat([prefix, torch.randint(1, 64, (5,), generator=g)])
    b = torch.cat([prefix, torch.randint(1, 64, (7,), generator=g)])
    for layer in (0, 2):
        xa = collect_mlp_tokens(model, make_batches([a], {}, 1), [layer])[layer][0]
        xb = collect_mlp_tokens(model, make_batches([b], {}, 1), [layer])[layer][0]
        assert torch.equal(xa[:6], xb[:6])
        assert (xa[6:11] - xb[6:11]).abs().max() > 1e-4  # tokens after the shared prefix do differ


def test_apply_w_down_rejects_bias(model):
    from transformers import PhiConfig, PhiForCausalLM
    phi = PhiForCausalLM(PhiConfig(vocab_size=64, hidden_size=HIDDEN, intermediate_size=FF, num_hidden_layers=2,
                                   num_attention_heads=4)).eval()
    with pytest.raises(ValueError, match="bias"):
        apply_w_down(phi, 0, torch.zeros(HIDDEN, FF))


# ---------------------------------------------------------------------------------------------
# steering (layer selection) and probe statistics


def block_out_all_layers(model, batch):
    with torch.no_grad(), capture_activations(model, list(range(N_LAYERS)), "block_out") as cache:
        model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        return {l: cache[l].clone() for l in range(N_LAYERS)}


@pytest.mark.parametrize("mode", ["eoi", "all"])
def test_steerer_adds_coeff_r_at_the_right_tokens(model, mode):
    prompts = make_prompts(4, seed=11)
    batch = make_batches(prompts, {}, 4, "left")[0]
    r = torch.randn(4, len(POSITIONS), HIDDEN)
    layer, coeff = 1, 2.0
    base = block_out_all_layers(model, batch)
    with Steerer(model, layer, POSITIONS, mode=mode, ref_position=-1, coeff=coeff) as steerer:
        steerer.set(r, prompt_len=batch["input_ids"].shape[1])
        steered = block_out_all_layers(model, batch)
    diff = steered[layer] - base[layer]
    if mode == "eoi":
        assert torch.allclose(diff[:, -3:], coeff * r, atol=1e-5), "last 3 prompt tokens get coeff * r[:, p]"
        assert diff[:, :-3].abs().max() < 1e-6, "other tokens untouched"
    else:
        assert torch.allclose(diff, coeff * r[:, -1:].expand_as(diff), atol=1e-5), "every token gets r[:, ref]"
    assert all(torch.equal(steered[l], base[l]) for l in range(layer)), "upstream layers untouched"
    assert (steered[layer + 1] - base[layer + 1]).abs().max() > 1e-3, "downstream layers see the redirection"
    # the hook is removed on exit and the vectors are cleared
    assert all(torch.equal(v, base[l]) for l, v in block_out_all_layers(model, batch).items())


def test_steerer_eoi_is_what_a_perfect_w_down_edit_does(model):
    """Steering block_out of layer l == adding build_shift(eoi) to its MLP output (the W_down training target)."""
    prompts = make_prompts(3, seed=12)
    batch = make_batches(prompts, {}, 3, "left")[0]
    r = torch.randn(3, len(POSITIONS), HIDDEN)
    layer = 2
    with torch.no_grad(), Steerer(model, layer, POSITIONS, mode="eoi") as steerer:
        steerer.set(r, prompt_len=batch["input_ids"].shape[1])
        steered = model(**{k: batch[k] for k in ("input_ids", "attention_mask")}).logits
    mlp = model.model.layers[layer].mlp
    shift = build_shift(r, batch["attention_mask"], POSITIONS, "eoi")
    handle = mlp.register_forward_hook(lambda m, a, out: out + shift)
    try:
        with torch.no_grad():
            edited = model(**{k: batch[k] for k in ("input_ids", "attention_mask")}).logits
    finally:
        handle.remove()
    assert torch.allclose(steered, edited, atol=1e-5)


def test_steerer_eoi_leaves_decoding_steps_alone(model):
    steerer = Steerer(model, 0, POSITIONS, mode="eoi")
    steerer.set(torch.ones(2, len(POSITIONS), HIDDEN), prompt_len=8)
    step = torch.randn(2, 1, HIDDEN)  # one generated token per row
    assert torch.equal(steerer._hook(None, (), step), step)
    steerer.set(None)
    assert steerer._hook(None, (), step) is None
    with pytest.raises(ValueError):
        Steerer(model, 0, POSITIONS, mode="all", ref_position=-7)


def test_steered_generation_runs_and_zero_r_changes_nothing(model):
    prompts = make_prompts(3, seed=13)
    batch = make_batches(prompts, {}, 3, "left")[0]
    kwargs = dict(max_new_tokens=5, do_sample=False, pad_token_id=0)
    ref = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], **kwargs)
    with Steerer(model, 1, POSITIONS, mode="eoi") as steerer:
        steerer.set(torch.zeros(3, len(POSITIONS), HIDDEN), prompt_len=batch["input_ids"].shape[1])
        same = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], **kwargs)
        steerer.set(50 * torch.randn(3, len(POSITIONS), HIDDEN), prompt_len=batch["input_ids"].shape[1])
        other = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], **kwargs)
    assert torch.equal(ref, same) and not torch.equal(ref, other)


def test_answer_logprobs_match_unpadded_teacher_forcing(model):
    prompts = make_prompts(3, seed=14)
    answers = make_prompts(3, seed=15, lo=2, hi=6)
    batch = make_batches(prompts, {}, 3, "left")[0]
    width = max(len(p) + len(a) for p, a in zip(prompts, answers))
    labels = torch.full((3, width), IGNORE_INDEX)
    for i, (p, a) in enumerate(zip(prompts, answers)):  # like the collator: labels padded on the left too
        lab = torch.cat([torch.full((len(p),), IGNORE_INDEX), a])
        labels[i, width - len(lab):] = lab
    got = answer_logprobs(model, {**batch, "labels": labels}, pad_token_id=0)
    for i, (p, a) in enumerate(zip(prompts, answers)):
        seq = torch.cat([p, a])[None]
        with torch.no_grad():
            logp = model(input_ids=seq).logits[0].log_softmax(-1)
        manual = logp[len(p) - 1:len(seq[0]) - 1].gather(-1, a[:, None]).mean().item()
        assert abs(got[i] - manual) < 1e-4, f"row {i}: {got[i]} vs {manual}"
    with pytest.raises(ValueError, match="LEFT"):
        answer_logprobs(model, {**make_batches(prompts, {}, 3, "right")[0], "labels": labels}, pad_token_id=0)


def test_auc_and_separability():
    assert auc(torch.tensor([3., 4.]), torch.tensor([1., 2.])) == 1.0
    assert auc(torch.tensor([1., 2.]), torch.tensor([3., 4.])) == 0.0
    assert auc(torch.tensor([1., 1.]), torch.tensor([1., 1.])) == 0.5
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(60, 16, generator=g), torch.randn(60, 16, generator=g)
    assert abs(mean_diff_auc(a, b) - 0.5) < 0.15, "same distribution -> close to chance"
    assert mean_diff_auc(a + 3 * torch.ones(16), b) > 0.99, "shifted along one direction -> separable"


def test_response_flags_and_summary():
    assert is_refusal("I'm sorry, I don't have that information.") and not is_refusal("She was born in Paris.")
    assert is_refusal("I don’t know.")  # typographic apostrophe
    assert is_degenerate("") and is_degenerate("the cat the cat the cat the cat the cat the cat")
    assert not is_degenerate("He was born in Lisbon to a family of tailors and became a writer.")
    recs = [
        {"generation": "Born in Rome.", "rougeL_recall": 0.1, "rouge1_recall": 0.2, "refusal": False, "degenerate": False, "answer_logprob": -2.0},
        {"generation": "I don't know.", "rougeL_recall": 0.0, "rouge1_recall": 0.0, "refusal": True, "degenerate": False, "answer_logprob": -4.0},
        {"generation": "Lisbon, tailors.", "rougeL_recall": 0.9, "rouge1_recall": 0.9, "refusal": False, "degenerate": False, "answer_logprob": -0.5},
    ]
    s = summarize_responses(recs, hallucination_rouge=0.3)
    assert s["n"] == 3 and abs(s["refusal_rate"] - 1 / 3) < 1e-9 and abs(s["hallucination_rate"] - 1 / 3) < 1e-9
    assert abs(s["answer_logprob"] - (-6.5 / 3)) < 1e-9


# ---------------------------------------------------------------------------------------------
# closed-form W_down (LUNAR Eq. 9), group-wise AUC, probe comparison


def test_closed_form_equals_weighted_least_squares():
    """ridge=0, more tokens than features: W is the (weighted) least-squares solution, whatever W0 is."""
    g = torch.Generator().manual_seed(0)
    p, d, nf, nr = 12, 5, 30, 50
    fx, rx = torch.randn(nf, p, generator=g), torch.randn(nr, p, generator=g)
    ft, rt = torch.randn(nf, d, generator=g), torch.randn(nr, d, generator=g)
    w0 = torch.randn(d, p, generator=g)
    wf, wr = 2.0, 0.5
    w, hist = solve_w_down_closed_form(w0, (fx, ft), (rx, rt), ridge=0.0, forget_weight=wf, retain_weight=wr, chunk=7)
    sf, sr = (wf / nf) ** 0.5, (wr / nr) ** 0.5  # every token weighs w_set / N_set
    ref = torch.linalg.lstsq(torch.cat([fx * sf, rx * sr]).double(), torch.cat([ft * sf, rt * sr]).double()).solution.T
    assert torch.allclose(w.double(), ref, atol=1e-3), (w.double() - ref).abs().max()
    assert hist[0]["forget_mse"] == pytest.approx(w_down_mse(w, fx, ft), rel=1e-5)


def test_closed_form_ridge_shrinks_towards_the_original_weights():
    g = torch.Generator().manual_seed(1)
    p, d = 16, 4
    fx, rx = torch.randn(40, p, generator=g), torch.randn(60, p, generator=g)
    w0 = torch.randn(d, p, generator=g)
    ft = F_linear(fx, w0) + 1.0  # forget targets = original output + a shift
    rt = F_linear(rx, w0)
    moves = []
    for ridge in (0.0, 1e-2, 1.0, 100.0):
        w, _ = solve_w_down_closed_form(w0, (fx, ft), (rx, rt), ridge=ridge)
        moves.append((w - w0).norm().item())
    assert moves == sorted(moves, reverse=True) and moves[-1] < 0.5 * moves[0], moves
    w_inf, _ = solve_w_down_closed_form(w0, (fx, ft), (rx, rt), ridge=1e9)
    assert torch.allclose(w_inf, w0, atol=1e-3), "huge ridge = no edit"


def F_linear(x, w):
    return torch.nn.functional.linear(x, w)


def test_closed_form_end_to_end_effect_on_the_model():
    """Same experiment as the Adam one: the edited model moves the forget block_out by ~ coeff * r_UV, the retain one
    hardly, and the closed form fits at least as well as Adam without any lr / epochs."""
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=HIDDEN, intermediate_size=2048, num_hidden_layers=N_LAYERS,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    model = LlamaForCausalLM(cfg).eval()
    layer, coeff = 1, 1.0
    f_prompts, f_meta = make_prompts(8, seed=10), {"author": [i // 4 for i in range(8)]}
    n_prompts = make_prompts(12, seed=11)
    n_meta = {"key": [(i // 6, (i // 3) % 2) for i in range(12)]}
    retain = make_prompts(24, seed=12)
    fb, nb, rb = make_batches(f_prompts, f_meta, 4), make_batches(n_prompts, n_meta, 4), make_batches(retain, {}, 8)
    f_means, _, _ = collect_group_means(model, fb, lambda b: b["author"], POSITIONS, [layer])
    n_means, _, _ = collect_group_means(model, nb, lambda b: b["key"], POSITIONS, [layer])
    r_uv = compute_r_uv(f_means, n_means, nested_neighbor_order({0: [0, 1], 1: [0, 1]}, seed=0), ks=[2])[2]
    r_layer = {a: r_uv[a][:, 0, :] for a in r_uv}
    shift_fn = lambda b, l, am: coeff * build_shift(torch.stack([r_layer[a] for a in b["author"]]), am, POSITIONS, "eoi")
    f_tok = collect_mlp_tokens(model, fb, [layer], shift_fn=shift_fn)[layer]
    r_tok = collect_mlp_tokens(model, rb, [layer])[layer]

    w0 = get_out_proj(model, layer).weight.detach().clone()
    f_before = w_down_mse(w0, *f_tok)
    w, _ = solve_w_down_closed_form(w0, f_tok, r_tok, ridge=1e-6)
    w_adam, _ = train_w_down(w0, f_tok, r_tok, lr=3e-3, epochs=150, batch_size=32, lr_gamma=0.98, seed=0)
    assert w_down_mse(w, *f_tok) < 1e-3 * f_before and w_down_mse(w, *r_tok) < 1e-3 * f_before
    assert w_down_mse(w, *f_tok) <= w_down_mse(w_adam, *f_tok) + 1e-9, "closed form is the optimum Adam only approaches"

    retain_before = collect_group_means(model, rb, lambda b: [0] * len(b["input_ids"]), POSITIONS, [layer])[0][0]
    apply_w_down(model, layer, w)
    f_after, _, _ = collect_group_means(model, fb, lambda b: b["author"], POSITIONS, [layer])
    retain_after = collect_group_means(model, rb, lambda b: [0] * len(b["input_ids"]), POSITIONS, [layer])[0][0]
    for a in (0, 1):
        achieved, intended = (f_after[a] - f_means[a]).flatten(), (coeff * r_uv[a]).flatten()
        cos = torch.nn.functional.cosine_similarity(achieved, intended, dim=0).item()
        ratio = (achieved.norm() / intended.norm()).item()
        assert cos > 0.99 and 0.9 < ratio < 1.1, f"author {a}: cos={cos:.3f} norm ratio={ratio:.3f}"
    assert ((retain_after - retain_before).norm() / retain_before.norm()).item() < 0.01


def test_selection_file_name_encodes_k_coeff_and_mode():
    assert selection_file_name(5, 1.0, "eoi") == "layer_selection_k5_c1_eoi.json"
    assert selection_file_name(15, 2.5, "all") == "layer_selection_k15_c2.5_all.json"
    assert selection_file_name(5, 1.0, "eoi") != selection_file_name(5, 1.0, "all")


def test_group_folds_keep_groups_together_and_share_common_groups():
    g = torch.Generator().manual_seed(0)
    ga = [f"author:{i // 20}" for i in range(60)]  # 3 forget authors
    gb = [f"author:{i // 30}" for i in range(90)] + [f"holdout:{i // 20}" for i in range(40)]  # their neighbours + 2 others
    fa, fb = _group_folds(ga, gb, 2, g)
    for groups, folds in ((ga, fa), (gb, fb)):
        for key in set(groups):
            assert len({int(f) for k, f in zip(groups, folds) if k == key}) == 1, f"{key} split over folds"
    shared = set(ga) & set(gb)
    assert all(int(fa[ga.index(k)]) == int(fb[gb.index(k)]) for k in shared), "a shared group must share its fold"
    assert {int(x) for x in fa} == {0, 1} and {int(x) for x in fb} == {0, 1}
    with pytest.raises(ValueError):
        _group_folds(["x"] * 5, ["y"] * 5, 2, g)  # one group per class cannot fill 2 folds


def test_group_wise_auc_removes_the_author_leak():
    """Classes with NO difference except that every author has his own offset: splitting by question lets the same
    author sit on both sides and the AUC is high; splitting by author brings it down."""
    torch.manual_seed(1)
    d = 64
    a = torch.randn(30, d).repeat_interleave(10, 0) + 0.3 * torch.randn(300, d)
    b = torch.randn(30, d).repeat_interleave(10, 0) + 0.3 * torch.randn(300, d)
    ga, gb = [i // 10 for i in range(300)], [1000 + i // 10 for i in range(300)]
    by_question, by_author = mean_diff_auc(a, b), mean_diff_auc(a, b, groups_a=ga, groups_b=gb)
    assert by_question > 0.9 and by_author < by_question - 0.2, (by_question, by_author)
    with pytest.raises(ValueError, match="both classes"):
        mean_diff_auc(a, b, groups_a=ga)
    with pytest.raises(ValueError, match="at least 4 groups"):  # 2 authors: the number would be ~0 or ~1 by chance
        mean_diff_auc(a[:40], b[:40], groups_a=[i // 20 for i in range(40)], groups_b=[100 + i // 20 for i in range(40)])
    assert 0.0 <= mean_diff_auc(a[:40], b[:40], groups_a=[i // 20 for i in range(40)],
                                groups_b=[100 + i // 20 for i in range(40)], min_groups=2) <= 1.0


def test_collect_activations_returns_requested_metadata(model):
    prompts = make_prompts(5, seed=20)
    batches = make_batches(prompts, {"index": list(range(5)), "author_id": [10, 10, 11, 11, 12]}, 2)
    acts, layers, meta = collect_activations(model, batches, POSITIONS, reduce="none", meta_keys=("author_id", "nope"))
    assert sorted(acts) == list(range(5)) and meta[2] == {"author_id": 11} and meta[4] == {"author_id": 12}
    assert len(collect_activations(model, batches, POSITIONS, reduce="none")) == 2, "unchanged without meta_keys"


def test_neighbor_sample_groups():
    from neighbor import sample_group
    assert sample_group("neighbor", 7, {"author_id": 3, "neighbor_id": 1}, "forget", 20, 0) == "author:3"
    assert sample_group("forget", 45, {}, "forget", 20, 0) == "author:2"
    assert sample_group("forget", 45, {}, "forget", 20, 198) == "author:200"  # aligned with the csv numbering
    assert sample_group("holdout", 45, {}, "forget", 20, 198) == "holdout:2"
    assert sample_group("retain", 45, {}, "forget", 20, 0) != sample_group("holdout", 45, {}, "forget", 20, 0)


def test_neighbor_compare_difference_of_two_probes():
    from neighbor_compare import compare, format_table
    layers = lambda fg, rg: {
        str(l): {"auc[forget | neighbor]": 0.5 + f, "auc_group[forget | neighbor]": g, "r_uv_rel_norm": 0.05 * (l + 1)}
        for l, (f, g) in enumerate(zip(fg, rg))
    }
    full = {int(k): v for k, v in layers([0.3, 0.2], [0.8, 0.6]).items()}
    retain = {int(k): v for k, v in layers([0.0, 0.1], [0.5, None]).items()}
    rows = compare(full, retain, "forget | neighbor")
    assert [r["layer"] for r in rows] == [0, 1]
    assert rows[0]["auc_delta"] == pytest.approx(0.3) and rows[0]["auc_group_delta"] == pytest.approx(0.3)
    assert rows[1]["auc_delta"] == pytest.approx(0.1) and rows[1]["auc_group_delta"] is None  # skipped group AUC
    assert "top 1 layers by auc_group_delta: 0" in format_table(rows, "forget | neighbor")
    with pytest.raises(KeyError):
        compare(full, retain, "forget | holdout")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-x"]))
