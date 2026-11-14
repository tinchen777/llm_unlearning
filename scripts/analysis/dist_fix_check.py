"""Acceptance check of the NEAR-Dist likelihood fix (TODO/paper_notes/3_最新数据与新发现.md §2 (old: history/10 §10); the criteria were fixed BEFORE the runs, TODO/exp/20).

For one cell: the original E8 run (prefix _v3), every variant under the validation prefix (_v4: `_rkl<w>` retain-answer KL, `_dyn<N>` dynamic forget target), NEAR-Act and the retain model.
    python scripts/analysis/dist_fix_check.py --size 3B --split forget05 [--out saves/analysis/dist_fix_3B_forget05.md]
Criteria (all must hold):
  C1 forget_Q_A_Prob <= 0.25                                  (the matching rule of the main table)
  C2 forget / holdout answer probability in [0.8, 1.25]       (the forget answers at the level of unseen data AS THE MODEL SEES IT; E8 forget01: 0.162 / 0.156)
  C3 |mia_loss - retain's mia_loss| <= 0.15                   (NEAR-Act's level: median 0.14 at forget05 / forget10)
  C4 retain_Q_A_Prob >= 0.70                                  (retain model 0.94, NEAR-Act 0.76 at 3B forget05; E8 0.29)
  C5 ud_bb, ud_wb_unsup, ud_wb_sup <= original E8 + 0.05      (no new trace)
  C6 model_utility >= original E8 - 0.01
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cells  # noqa: E402

E8 = "fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn"     # the E8 stem; the steps (_s300 / _s500 / none for forget01) come before it
RETAIN = {"forget01": "retain99", "forget05": "retain95", "forget10": "retain90"}


def load(d):
    out = {}
    for f in ("TOFU_SUMMARY.json", "TOFU_EVAL.json"):
        p = os.path.join(d, f)
        if os.path.exists(p):
            j = json.load(open(p))
            for k, v in j.items():
                if isinstance(v, dict) and "agg_value" in v:
                    v = v["agg_value"]
                if isinstance(v, (int, float)):
                    out.setdefault(k, v)
    return out


def rows_for(size, split, root="saves"):
    pre = cells.NEAR_PREFIX[(size, split)]
    model = cells.MODELS[size]
    rows = []
    for d in sorted(glob.glob(f"{root}/eval/NEAR/{pre}_v3/rw1_ans_kl1*_{E8}_increment_seq_*_k5_*")):
        rows.append(("NEAR-Dist (original)", d))
    for d in sorted(glob.glob(f"{root}/eval/NEAR/{pre}_v4*/rw1_ans_kl1*_{E8}*_k5_*")):   # _v4 = seed 0, _v4_s<n> = seed n
        var = os.path.basename(d).split(E8)[1].split("_increment_seq")[0].lstrip("_") or "original config"
        seed = os.path.basename(os.path.dirname(d)).split(f"{pre}_v4")[1].lstrip("_") or "s0"
        rows.append((f"Dist {var} ({seed})", d))
    seen = {d for _, d in rows}
    for d in sorted(glob.glob(f"{root}/eval/NEAR/{pre}_v4*/rw1_ans_kl1*_rkl1*_k5_*")):   # component ablations of v2 (TODO/exp/26): they change the E8 stem itself
        if d in seen:
            continue
        name = os.path.basename(d).split("_increment_seq")[0]
        seed = os.path.basename(os.path.dirname(d)).split(f"{pre}_v4")[1].lstrip("_") or "s0"
        rows.append((f"Abl {name.replace('rw1_ans_kl1_', '')} ({seed})", d))
    pts, _ = cells.load_cell(size, split, root)
    act, _, _ = cells.best_matched(pts, "NEAR-Act")
    if act and act.get("path"):
        rows.append(("NEAR-Act (matched)", act["path"]))
    r = f"{root}/eval/TOFU/tofu_{model}_{RETAIN[split]}"
    return rows, load(r)


def check(m, ref, retain):
    ratio = (m.get("forget_answer_prob") or float("nan")) / (m.get("holdout_answer_prob") or float("nan"))
    c = {"C1": m.get("forget_Q_A_Prob", 9) <= 0.25,
         "C2": 0.8 <= ratio <= 1.25,
         "C3": abs(m.get("mia_loss", 9) - retain.get("mia_loss", -9)) <= 0.15,
         "C4": m.get("retain_Q_A_Prob", 0) >= 0.70,
         "C5": all(m.get(k, 9) <= ref.get(k, -9) + 0.05 for k in ("ud_bb", "ud_wb_unsup", "ud_wb_sup")),
         "C6": m.get("model_utility", 0) >= ref.get("model_utility", 9) - 0.01}
    return ratio, c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="3B"); ap.add_argument("--split", default="forget05"); ap.add_argument("--root", default="saves"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows, retain = rows_for(a.size, a.split, a.root)
    ref = next((load(d) for n, d in rows if n == "NEAR-Dist (original)"), {})
    cols = ["forget_Q_A_Prob", "forget_answer_prob", "holdout_answer_prob", "mia_loss", "retain_Q_A_Prob", "retain_answer_prob", "model_utility", "ud_bb", "ud_wb_unsup", "ud_wb_sup"]
    lines = [f"# NEAR-Dist likelihood fix, {a.size} {a.split}", "", f"retain model: mia_loss {retain.get('mia_loss', float('nan')):.3f}, retain_Q_A_Prob {retain.get('retain_Q_A_Prob', float('nan')):.3f}", "",
             "| run | " + " | ".join(cols) + " | F/H ratio | C1 | C2 | C3 | C4 | C5 | C6 |", "|---|" + "---|" * (len(cols) + 7)]
    for name, d in rows:
        m = load(d)
        ratio, c = check(m, ref, retain)
        f = lambda k: "-" if k not in m else f"{m[k]:.3f}"
        lines.append(f"| {name} | " + " | ".join(f(k) for k in cols) + f" | {ratio:.2f} | " + " | ".join("✓" if c[k] else "✗" for k in sorted(c)) + " |")
    lines += ["", "C1 Prob <= 0.25; C2 forget/holdout answer prob in [0.8, 1.25]; C3 |mia - retain| <= 0.15; C4 retain_Q_A_Prob >= 0.70; C5 no trace above the original E8 + 0.05; C6 utility >= original - 0.01."]
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
