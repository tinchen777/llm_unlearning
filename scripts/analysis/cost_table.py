"""Cost of the runs that wrote it (src/utils/cost.py): training time and peak GPU memory per stage, trainable parameters; the evaluation is not included.

    python scripts/analysis/cost_table.py saves/NEAR/single_v3 saves/LUNAR/single saves/tofu_unlearn
    - NEAR / LUNAR: `<run>/.../w_down_summary.json` ("cost": record_tokens, stage1_regression, [refine_kl_hinge]) + the shared upstream of its r_uv.pt (cost_r_uv.json) and its
      steering sweep (cost_select_*.json), shown as "upstream_s" (the r_UV pass + the sweep of THIS coefficient; they are shared by all runs of a setting).
    - baselines: `<run>/cost.json` (the time of `train` includes the per-epoch evaluations of the trainer; memory = peak of this process).
Only runs made after the cost meter was added have the files.
"""
import glob
import json
import os
import sys


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def upstream_seconds(summary):
    base = os.path.dirname(str(summary.get("r_uv_path", "")))
    total, found = 0.0, False
    r = load(os.path.join(base, "cost_r_uv.json"))
    if r:
        total, found = total + r["total_seconds"], True
    k, c, mode = summary.get("k"), summary.get("coeff"), summary.get("shift_mode")
    names = [f"cost_select_k{k}_c{c:g}_{mode}.json", f"cost_select_c{c:g}_{mode}.json"] if c is not None else []
    for name in names:
        s = load(os.path.join(base, name))
        if s:
            total, found = total + s["total_seconds"], True
    return total if found else None


def rows(roots):
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, "**", "w_down_summary.json"), recursive=True)):
            s = load(path)
            if not s or "cost" not in s:
                continue
            c = s["cost"]
            yield (os.path.relpath(os.path.dirname(path), root)[:64], sum(v["seconds"] for v in c["stages"].values()), c["peak_allocated_gb"],
                   c.get("trainable_params"), upstream_seconds(s), ", ".join(f"{k} {v['seconds']:.0f}s" for k, v in c["stages"].items()))
        for path in sorted(glob.glob(os.path.join(root, "**", "cost.json"), recursive=True)):
            c = load(path)
            if c:
                yield (os.path.relpath(os.path.dirname(path), root)[:64], c["total_seconds"], c["peak_allocated_gb"], c.get("trainable_params"), None,
                       ", ".join(f"{k} {v['seconds']:.0f}s" for k, v in c["stages"].items()))


def main():
    roots = sys.argv[1:] or ["saves"]
    print(f"{'run':64s} {'train_s':>8s} {'peak_GB':>8s} {'params_M':>9s} {'upstream_s':>10s}  stages")
    for name, total, peak, params, up, stages in rows(roots):
        f = lambda v, d=1: "-" if v is None else f"{v:.{d}f}"
        print(f"{name:64s} {total:8.0f} {f(peak):>8s} {f(None if params is None else params / 1e6):>9s} {f(up, 0):>10s}  {stages}")


if __name__ == "__main__":
    main()
