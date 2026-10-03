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
    capture_activations, collect_activations, collect_group_means, gather_positions, get_out_proj,
)
from model.redirect import (
    apply_w_down, build_shift, check_neighbors, collect_mlp_tokens, compute_r_uv, forget_authors,
    nested_neighbor_order, neighbors_by_author, summarize_r_uv, train_w_down, w_down_mse,
)

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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-x"]))
