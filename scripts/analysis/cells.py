"""The 9 cells (forget01 / 05 / 10 x 1B / 3B / 8B) of the main table and the Pareto grid: loads every evaluated point of one cell from saves/, with the method names of the paper.

    NEAR-Act   = the steering + regression runs without the KL stage (family `NEAR` of pareto.py)
    NEAR-Dist  = the FINAL E8 run of the cell (`fmneighbor_scaled_initoriginal_kfw0_npq3_mn`); the earlier calibration variants / NEAR+KL are left out
    LUNAR, GradAscent, GradDiff, NPO, SimNPO, RMU as in saves/eval/LUNAR and saves/tofu_unlearn (protocol `_ep5`, every epoch checkpoint is a point)
    (ROCR: `rocr_points`, the evaluations of saves/eval/ROCR; `ROCR` = the original, `ROCR-nb` = the neighbour target, `ROCR-nb-s<k>` = residual factor k)
No model is run; everything comes from the TOFU_SUMMARY.json files the evaluations wrote.
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pareto  # noqa: E402

MODELS = {"1B": "Llama-3.2-1B-Instruct", "3B": "Llama-3.2-3B-Instruct", "8B": "Llama-3.1-8B-Instruct"}
SPLITS = ("forget01", "forget05", "forget10")
NEAR_PREFIX = {("1B", "forget01"): "single", ("3B", "forget01"): "Llama3B", ("8B", "forget01"): "Llama8B",
               **{(s, f): f"{f}_Llama{s}" for s in MODELS for f in ("forget05", "forget10")}}
MAIN = ("GradAscent", "NPO", "RMU", "LUNAR", "NEAR-Act", "NEAR-Dist")
APPENDIX = ("GradDiff", "SimNPO")
E8_RE = re.compile(r"fmneighbor_scaled_initoriginal_kfw0_npq3_mn")


def load_cell(size, split, root="saves"):
    """(points, refs): points = [{method, run, <KEYS>}]; refs = {retain, full}."""
    model = MODELS[size]
    pre = NEAR_PREFIX[(size, split)]
    pts = pareto.collect([f"{root}/eval/NEAR/{pre}_v2", f"{root}/eval/NEAR/{pre}_v3"], f"{root}/eval/LUNAR/none", f"{root}/tofu_unlearn", model, split, "5")
    for d in sorted(glob.glob(f"{root}/eval/LUNAR/*")):
        pts += [p for p in pareto.collect([], d, f"{root}/none", model, split) if p["family"] == "LUNAR"]
    out = []
    for p in pts:
        if p["family"] == "NEAR":
            out.append({**p, "method": "NEAR-Act"})
        elif p["family"] == "KL+hinge":
            if E8_RE.search(p["run"]):
                out.append({**p, "method": "NEAR-Dist"})
        elif p["family"] in ("LUNAR", "baseline"):
            out.append(p)
    return out, pareto.references(model, split)


def rocr_points(size, split, root="saves"):
    """The ROCR evaluations of the cell: [{method, run, <KEYS>}] (methods ROCR, ROCR-nb, ROCR-nb-s<k>, ROCR-nb-l<layers> for the layer scans)."""
    model = MODELS[size]
    out = []
    for d in sorted(glob.glob(f"{root}/eval/ROCR/{model}_{split}_*")):
        s = pareto.read_summary(os.path.join(d, "TOFU_SUMMARY.json"))
        if s is None:
            continue
        tail = os.path.basename(d).split(f"{model}_{split}_")[1]
        m = re.match(r"(famous|neighbor)_l([\d\-]+)(?:_s(\d+))?$", tail)
        if not m:
            continue
        kind, layers, scale = m.groups()
        method = "ROCR" if kind == "famous" else "ROCR-nb"
        if scale:
            method += f"-s{scale}"
        out.append({"family": "ROCR", "method": method, "run": f"l{layers}", **s})
    return out


def best_matched(points, method, tau=0.25):
    """(point, n_ok, n): the best-utility point of the method with forget_Q_A_Prob <= tau (matching rule: Prob only), else the lowest-Prob point and None."""
    mine = [p for p in points if p["method"] == method]
    if not mine:
        return None, 0, 0
    ok = [p for p in mine if p["forget_Q_A_Prob"] <= tau]
    if ok:
        return max(ok, key=lambda p: p["model_utility"]), len(ok), len(mine)
    return min(mine, key=lambda p: p["forget_Q_A_Prob"]), 0, len(mine)
