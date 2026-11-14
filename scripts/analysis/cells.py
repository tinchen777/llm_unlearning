"""The 9 cells (forget01 / 05 / 10 x 1B / 3B / 8B) of the main table and the Pareto grid: loads every evaluated point of one cell from saves/, with the method names of the paper.

    NEAR-Act   = the steering + regression runs without the KL stage (family `NEAR` of pareto.py)
    NEAR-Dist  = the FINAL configuration (2026-11-13): E8 (`fmneighbor_scaled_initoriginal_kfw0_npq3_mn`) + the retain-answer KL weight 1 (`_rkl1`), prefix <pre>_v4 (TODO/exp/20)
    NEAR-Dist-v1 = E8 without the retain KL (prefix <pre>_v3; the likelihood drift of TODO/paper_notes/3_最新数据与新发现.md §2 (old: history/10 §10)): appendix only; the other calibration variants / NEAR+KL are left out
    LUNAR, GradAscent, GradDiff, NPO, SimNPO, RMU as in saves/eval/LUNAR and saves/tofu_unlearn (protocol `_ep5`, every epoch checkpoint is a point; default seed only, see SEED_RUN_RE)
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
APPENDIX = ("GradDiff", "SimNPO", "NEAR-Dist-v1")
# folders of saves/eval/LUNAR that are NOT results of their name (TODO/exp/23 §4: the 3B "forget05 ..._ks" runs were forget10 runs) -> never read
MISFILED = {"misfiled"}
# SEED POLICY (2026-11-14, TODO/exp/25): the main table uses the DEFAULT seed of every method. The NEAR seeds live in other prefixes (*_v4_s<n>) and were never
# read; the baseline seeds (runs `..._s1`, `..._s2` next to the default one) were, and best_matched then picked the best-utility point over THREE seeds for GA /
# NPO (utility ties to 2 decimals), i.e. the luckiest white-box seed. They are now skipped unless CELLS_ALL_SEEDS=1 (the old behaviour).
SEED_RUN_RE = re.compile(r"_s\d+(?:@|$)")
ALL_SEEDS = os.environ.get("CELLS_ALL_SEEDS") == "1"
E8_RE = re.compile(r"fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment")          # E8 exactly (no later suffix): NEAR-Dist-v1
DIST_RE = re.compile(r"fmneighbor_scaled_initoriginal_kfw0_npq3_mn_rkl1_increment")    # + retain KL 1 only (no _dyn): NEAR-Dist


def load_cell(size, split, root="saves"):
    """(points, refs): points = [{method, run, <KEYS>}]; refs = {retain, full}."""
    model = MODELS[size]
    pre = NEAR_PREFIX[(size, split)]
    pts = pareto.collect([f"{root}/eval/NEAR/{pre}_v2", f"{root}/eval/NEAR/{pre}_v3", f"{root}/eval/NEAR/{pre}_v4"], f"{root}/eval/LUNAR/none", f"{root}/tofu_unlearn", model, split, "5")
    for d in sorted(glob.glob(f"{root}/eval/LUNAR/*")):
        if os.path.basename(d) in MISFILED:
            continue
        pts += [p for p in pareto.collect([], d, f"{root}/none", model, split) if p["family"] == "LUNAR"]
    out = []
    for p in pts:
        if p["family"] == "NEAR":
            out.append({**p, "method": "NEAR-Act"})
        elif p["family"] == "KL+hinge":
            if DIST_RE.search(p["run"]):
                out.append({**p, "method": "NEAR-Dist"})
            elif E8_RE.search(p["run"]):
                out.append({**p, "method": "NEAR-Dist-v1"})
        elif p["family"] in ("LUNAR", "baseline"):
            if p["family"] == "baseline" and SEED_RUN_RE.search(p["run"]) and not ALL_SEEDS:
                continue   # extra seeds (_s1, _s2): spread only, never a candidate point (TODO/exp/25)
            out.append(p)
    for p in out:
        p["path"] = _locate(p, size, split, root)
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
        out.append({"family": "ROCR", "method": method, "run": f"l{layers}", "path": d, **s})
    return out


def _locate(p, size, split, root="saves"):
    """The evaluation folder (TOFU_SUMMARY.json, activations_summary.json, ...) of a point, or None."""
    model, pre, run = MODELS[size], NEAR_PREFIX[(size, split)], p["run"]
    m = p["method"]
    if m in ("NEAR-Act", "NEAR-Dist", "NEAR-Dist-v1"):
        hits = [d for v in ("v2", "v3", "v4") for d in glob.glob(f"{root}/eval/NEAR/{pre}_{v}/{run}_k5_*")]
    elif m == "LUNAR":
        hits = [h for h in glob.glob(f"{root}/eval/LUNAR/*/*_{model}_{split}_*_{run}_l*") if h.split(os.sep)[-2] not in MISFILED]
    elif "@" in run:
        suffix, step = run.rsplit("@", 1)
        suffix = "" if suffix == "default" else "_" + suffix
        hits = [f"{root}/tofu_unlearn/{m}/tofu_{model}_{split}{suffix}/checkpoint-{step}/evals"]
    else:
        hits = []
    hits = [h for h in hits if os.path.isdir(h)]
    return hits[0] if hits else None


def meow_points(size, split, root="saves"):
    """The MEOW evaluations of the cell (saves/eval/MEOW/<model>_<split>_meow[_nb]_k<k>_largest[_lr<lr>]_s<step>): methods MEOW (the released inverted facts) and MEOW-nb (labels = the train
    neighbours' answers, same mechanics); `run` = `lr<lr>@<step>` (the earliest runs without an lr suffix are lr 1e-5)."""
    model = MODELS[size]
    out = []
    for d in sorted(glob.glob(f"{root}/eval/MEOW/{model}_{split}_meow*_s[0-9]*")):
        m = re.match(rf".*/{re.escape(model)}_{split}_meow(_nb)?_k\d+_largest(?:_lr([\de.\-]+))?_s(\d+)$", d)
        s_ = pareto.read_summary(os.path.join(d, "TOFU_SUMMARY.json"))
        if not m or s_ is None:
            continue
        nb, lr, step = m.groups()
        out.append({"family": "MEOW", "method": "MEOW-nb" if nb else "MEOW", "run": f"lr{lr or '1e-5'}@{step}", "path": d, **s_})
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
