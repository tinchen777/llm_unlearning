"""plots/paper_figs.py on a synthetic `saves/` tree that follows the real file schemas."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plots"))
import paper_figs  # noqa: E402

MODEL, FORGET = "Tiny", "forget01"
SPLITS = ["forget", "neighbor", "retain", "holdout"]
LAYERS = [2, 3, 4]


def _w(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _tofu(path: Path, fq: float, mu: float):
    _w(path / "TOFU_SUMMARY.json", {"forget_quality": fq, "model_utility": mu, "forget_Q_A_ROUGE": 0.4, "forget_Q_A_Prob": 0.5,
                                    "forget_truth_ratio": 0.6, "privleak": -3.0})


def _responses(after: bool):
    return {s: {"refusal_rate": 0.8 if after and s == "forget" else 0.0, "hallucination_rate": 0.1, "rougeL_recall": 0.3,
                "answer_prob": 0.2} for s in SPLITS}


def _method(root: Path, method: str, task: str, setting: str, near: bool):
    t = root / method / task
    per_layer = {str(l): ({"unseen_gap": 0.1 * l, "rougeL_recall": 0.3} if near else {"s1": 1.0, "s2": 0.5, "score": 0.1 * l}) for l in LAYERS}
    _w(t / setting / "layer_selection.json", {"selected_layer": 3, "per_layer": per_layer})
    _w(t / setting / "l3" / "w_down_summary.json", {"responses": {"before": _responses(False), "after": _responses(True)},
                                                    "per_layer": {"3": {"forget_mse_after": 0.1}}, "achieved_vs_intended": {"3": {"cos": 0.9}}})
    if near:
        _w(t / "r_uv_summary.json", {str(k): {str(l): {"norm": 1.0, "rel_norm": 0.1 * k, "cos_to_kmax": 0.5 + 0.1 * k} for l in LAYERS} for k in (1, 5)})
    else:
        _w(t / "r_uv_summary.json", {str(l): {"norm": 1.0, "rel_norm": 0.1} for l in LAYERS})


def _probe(path: Path, auc: float):
    _w(path / "activations_summary.json", {"layers": {str(l): {"auc[forget | neighbor]": auc, "auc[forget | holdout]": auc,
                                                              "auc_group[forget | neighbor]": None, "auc_group[forget | holdout]": None}
                                                      for l in LAYERS}})


@pytest.fixture
def saves(tmp_path):
    s = tmp_path / "saves"
    _method(s, "NEAR", f"near_tofu_{MODEL}_{FORGET}", "k5_c1_eoi", True)
    _method(s, "NEAR", f"near_tofu_{MODEL}_{FORGET}", "k5_c1_all", True)
    _method(s, "LUNAR", f"lunar_tofu_{MODEL}_{FORGET}_LUNAR_harmful", "c1_eoi", False)
    _tofu(s / "eval" / "NEAR" / f"eval_near_tofu_{MODEL}_{FORGET}_k5_l3_c1_eoi", 0.3, 0.55)
    _tofu(s / "eval" / "LUNAR" / f"eval_lunar_tofu_{MODEL}_{FORGET}_LUNAR_harmful_c1_l3_eoi", 0.01, 0.5)
    _tofu(s / "unlearn" / f"tofu_{MODEL}_{FORGET}" / "GradAscent", 1e-5, 0.1)
    _tofu(s / "eval" / f"tofu_{MODEL}_full" / f"evals_{FORGET}", 1e-9, 0.6)
    _tofu(s / "eval" / f"tofu_{MODEL}_retain99", 1.0, 0.6)
    _probe(s / "NEAR" / "0_probe" / f"tofu_{MODEL}_full" / FORGET, 0.99)
    _probe(s / "NEAR" / "0_probe" / f"tofu_{MODEL}_retain99", 0.6)
    _probe(s / "NEAR" / "probe" / f"probe_near_tofu_{MODEL}_{FORGET}_k5_l3_c1_eoi", 0.7)
    return s


def test_all_figures(saves, tmp_path):
    out = tmp_path / "out"
    made = paper_figs.main(["--saves", str(saves), "--out", str(out)])
    assert made == list(paper_figs.FIGURES)
    for name in ("figA_tofu_tradeoff", "figB_behavior", "figC_layer_selection", "figD_k_saturation", "figE_probe_auc"):
        assert (out / f"{name}_{MODEL}.png").exists()
    assert (out / "table1_tofu.md").exists()
    methods = set(paper_figs.pd.read_csv(out / "data" / "figA_tofu_metrics.csv")["method"])
    assert {"NEAR", "LUNAR", "GradAscent", "Original", "Retain"} <= methods
