"""White-box detectability at layers that are FIXED IN ADVANCE (not the maximum over the layers), and the supervised probe `wb_sup`.

`wb_unsup` / `wb_sup` of the evaluation are the best layer's advantage over the retain model, so the layer is chosen after the fact with the labels (an optimistic, oracle
choice). Here the layer is a fixed relative depth (default 50 %, 75 %, last layer), the same for every method:

    adv(layer) = max(0, 2 (AUC - 1/2)),  report max(0, adv(model) - adv(retain model))   at that layer
    unsup: `anom_auc[forget | benign]` (Mahalanobis outlier score fitted on the benign prompts)   sup: `auc[forget | neighbor]` (linear probe)

    python scripts/analysis/wb_fixed_layer.py --ref saves/eval/retain/Llama-3.2-3B-Instruct/forget01 \\
        NEAR-Dist=saves/eval/NEAR/Llama3B_v3/<run> NEAR-Act=saves/eval/NEAR/Llama3B_v2/<run> LUNAR=saves/eval/LUNAR/single/<run> [--depths 0.5 0.75 1.0]
"""
import argparse
import json
import os

UNSUP = "anom_auc[forget | benign]"
SUP = "auc[forget | neighbor]"


def layers_of(folder):
    with open(os.path.join(folder, "activations_summary.json"), encoding="utf-8") as f:
        return json.load(f)["layers"]


def adv(auc):
    return None if auc is None else max(0.0, 2.0 * (auc - 0.5))


def excess(layers, ref_layers, layer, key):
    a, r = adv(layers.get(str(layer), {}).get(key)), adv(ref_layers.get(str(layer), {}).get(key))
    if a is None:
        return None
    return max(0.0, a - (r or 0.0))


def pick_layers(layers, depths):
    n = max(int(k) for k in layers) + 1
    return sorted({min(n - 1, max(0, round(d * (n - 1)))) for d in depths})


def summary_value(folder, key):
    try:
        with open(os.path.join(folder, "TOFU_SUMMARY.json"), encoding="utf-8") as f:
            return json.load(f).get(key)
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="eval folder of the retain model (same model size, same forget split)")
    ap.add_argument("--depths", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    ap.add_argument("models", nargs="+", help="label=eval folder")
    a = ap.parse_args()
    ref = layers_of(a.ref)
    rows = []
    for spec in a.models:
        label, _, folder = spec.partition("=")
        layers = layers_of(folder)
        ls = pick_layers(layers, a.depths)
        rows.append((label, folder, layers, ls))
    ls = rows[0][3]
    head = "| 方法 | wb_unsup（取最大层，原指标） | wb_sup（取最大层，原指标） | " + " | ".join(f"unsup @L{l} | sup @L{l}" for l in ls) + " |\n|---|---|---|" + "---|---|" * len(ls)
    print(head)
    f3 = lambda x: "–" if x is None else f"{x:.3f}"
    for label, folder, layers, _ in rows:
        cells = []
        for l in ls:
            cells += [f3(excess(layers, ref, l, UNSUP)), f3(excess(layers, ref, l, SUP))]
        print(f"| {label} | {f3(summary_value(folder, 'ud_wb_unsup'))} | {f3(summary_value(folder, 'ud_wb_sup'))} | " + " | ".join(cells) + " |")
    print(f"\n固定层（相对深度 {a.depths}）：{ls}")


if __name__ == "__main__":
    main()
