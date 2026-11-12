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


def make_sets():
    S = [[f"S{2 * g}", f"S{2 * g + 1}"] for g in range(5)]
    return spa.validate_sets({"S_groups": S, "F": [f"F{i}" for i in range(10)], "H_A": [f"A{i}" for i in range(10)], "H_B": [f"B{i}" for i in range(10)], "templates": [f"t{i}" for i in range(N_T)]})


def make_acts(sets, rng, forgotten, direction, layer_signal=2, strength=2.5, noise=1.0):
    names = spa.all_names(sets)
    X = rng.normal(0, noise, (len(names), N_T, N_L, D)).astype(np.float32)
    for n in forgotten:
        X[names.index(n), :, layer_signal, :] += strength * direction
    return {"X": X, "names": names, "idx": {n: i for i, n in enumerate(names)}, "name_tokens": np.array([2] * len(names))}


def test_a_shared_direction_transfers_and_the_layer_is_found_on_the_shadow_data():
    rng = np.random.default_rng(0)
    sets = make_sets()
    d = rng.normal(size=D); d /= np.linalg.norm(d)
    shadows = [make_acts(sets, rng, sets["S_groups"][k], d) for k in range(5)]
    target = make_acts(sets, rng, sets["F"], d)
    res = spa.run_attack(shadows, target, sets, kinds=("lr", "dm"), boot=50)
    for kind in ("lr", "dm"):
        p = res["probes"][kind]
        assert p["selected_layer"] == 2
        assert p["shadow_cv_auc_selected"] > 0.9
        assert p["transfer"]["auc"] > 0.9 and p["transfer"]["auc"] > p["transfer"]["null95"]
        assert p["posthoc_best_layer"] == 2


def test_an_unrelated_direction_does_not_transfer_but_the_shadow_probe_works():
    rng = np.random.default_rng(1)
    sets = make_sets()
    d1 = rng.normal(size=D); d1 /= np.linalg.norm(d1)
    d2 = rng.normal(size=D); d2 -= d1 * (d2 @ d1); d2 /= np.linalg.norm(d2)
    shadows = [make_acts(sets, rng, sets["S_groups"][k], d1) for k in range(5)]
    target = make_acts(sets, rng, sets["F"], d2)
    p = spa.run_attack(shadows, target, sets, kinds=("dm",), boot=50)["probes"]["dm"]
    assert p["shadow_cv_auc_selected"] > 0.9
    assert p["transfer"]["auc"] < 0.8


def test_no_signal_at_all_stays_inside_the_null_band():
    rng = np.random.default_rng(2)
    sets = make_sets()
    d = rng.normal(size=D); d /= np.linalg.norm(d)
    shadows = [make_acts(sets, rng, sets["S_groups"][k], d) for k in range(5)]
    target = make_acts(sets, rng, [], d)                     # the control: nobody carries a trace
    t = spa.run_attack(shadows, target, sets, kinds=("dm",), boot=100)["probes"]["dm"]["transfer"]
    assert t["auc"] <= max(t["null95"], 0.75)


def test_sets_must_not_overlap_and_the_signed_evaluation_respects_the_direction():
    bad = make_sets(); bad["H_B"][0] = bad["H_A"][0]
    with pytest.raises(ValueError):
        spa.validate_sets(bad)
    hi = spa.evaluate_transfer([5, 6, 7, 8], [0, 1, 2, 1, 0, 1], k=2, boot=20)
    lo = spa.evaluate_transfer([0, 1, 0, 1], [5, 6, 7, 8, 6, 7], k=2, boot=20)
    assert hi["auc"] > 0.99 and lo["auc"] < 0.01              # a LOW score is not a detection (a two-sided |z| statistic would call both a detection)
