# PROTOTYPE of the unlearning-detectability numbers (see TODO/progress.md #4): run from saves/  ->  cd saves && python ../scripts/NEAR/proto_detectability.py
import glob, json, os, re
import statistics as st

def last_ckpt_eval(d):
    cks = [(int(re.search(r"checkpoint-(\d+)", p).group(1)), p) for p in glob.glob(d + "/checkpoint-*/evals/TOFU_EVAL.json")]
    return max(cks)[1] if cks else None

runs = {}
for kind in ("full", "retain"):
    runs[kind] = f"eval/{kind}/Llama-3.2-1B-Instruct/forget01/TOFU_EVAL.json"
for m in ("GradAscent", "GradDiff", "NPO", "SimNPO", "IdkDPO", "RMU"):
    p = last_ckpt_eval(f"tofu_unlearn/{m}/tofu_Llama-3.2-1B-Instruct_forget01")
    if p: runs[m] = p
for p in glob.glob("eval/LUNAR/single/*/TOFU_EVAL.json"): runs["LUNAR(c2,L10;20QA)"] = p
for tag, d in (("NEAR c12/L6", "k5_l6_c12_all"), ("NEAR c7/L9", "k5_l9_c7_all"), ("NEAR eoi c1/L11", "k5_l11_c1_eoi")):
    runs[tag] = f"eval/NEAR/single/near_tofu_Llama-3.2-1B-Instruct_forget01_{d}/TOFU_EVAL.json"
L = {k: json.load(open(v)) for k, v in runs.items() if os.path.exists(v)}

def flag_rate(d, split):
    if f"{split}_responses" not in d: return None
    v = list(d[f"{split}_responses"]["value_by_index"].values())
    return st.mean(bool(r["refusal"] or r["degenerate"]) for r in v), st.mean(bool(r["refusal"]) for r in v), st.mean(bool(r["degenerate"]) for r in v)

def agg(d, k):
    v = d.get(k)
    return float("nan") if v is None or v.get("agg_value") is None else v["agg_value"]

def adv(a): return max(0.0, 2 * (a - 0.5))
ref = L["retain"]
ref_flag = flag_rate(ref, "forget")[0]
print(f"reference (retain model) flagged rate on forget prompts: {ref_flag:.3f}")
print(f"{'model':22s} {'refuse':>6s} {'degen':>6s} {'BB=flag-ref':>11s} {'FPR(non-forget)':>15s} | {'WB(max layer, supervised AUC excess)':>38s} {'at layer':>8s} | {'forget_Q_A_ROUGE':>16s} {'forget_quality':>14s} {'utility':>7s}")
for name, d in L.items():
    fr = flag_rate(d, "forget")
    non = [flag_rate(d, s)[0] for s in ("retain", "holdout", "neighbor") if flag_rate(d, s)]
    bb = max(0.0, fr[0] - ref_flag) if fr else None
    wb, wl = None, None
    if "activation_probe" in d and "activation_probe" in ref:
        best = (0.0, None)
        for layer, v in d["activation_probe"]["value_by_layer"].items():
            r = ref["activation_probe"]["value_by_layer"][layer]
            ex = adv(v["auc[forget | neighbor]"]) - adv(r["auc[forget | neighbor]"])
            if ex > best[0]: best = (ex, layer)
        wb, wl = best
    s = agg(d, "forget_quality"); s = "-" if s != s else f"{s:.3f}"
    print(f"{name:22s} {fr[1]:6.2f} {fr[2]:6.2f} {bb:11.3f} {(st.mean(non) if non else float('nan')):15.3f} | {('-' if wb is None else f'{wb:.3f}'):>38s} {str(wl):>8s} | {agg(d,'forget_Q_A_ROUGE'):16.3f} {s:>14s} {agg(d,'model_utility'):7.3f}")
