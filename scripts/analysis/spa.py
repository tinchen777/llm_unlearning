"""A2: the shadow-probe attack (SPA), spec: TODO/exp/A2_SPEC.md.  Three sub-commands, one file.

  sets     write the author sets (shadow_sets.json): S = authors 181-190 of forget10 in 5 groups of 2 (the retained authors that have neighbours and MEOW facts),
           F = forget05, H = the holdout05 + holdout10 authors split into H_A (shadow negatives) and H_B (target negatives), the 10 retain-derived question templates.
  collect  activations of one model: for every name x template the output of EVERY decoder layer at the last prompt token (= end of the instruction, the chat prompt ends with
           the assistant header; same hook point `block_out` as ud_wb_*) -> one npz. Run it on every shadow model M''_k, on the target M' and on the controls.
  attack   per layer a probe is trained on the shadow models only (S_k of M''_k = 1, H_A = 0), the layer is chosen by author-grouped leave-one-group-out CV on the shadow data
           ALONE, the probe is then applied to the target M' (F = 1, H_B = 0). Reported: shadow CV AUC, transfer AUC (author level, minority pools of 2 F + all H_B as in A1 v2, delete-one-
           author jackknife interval, label-permutation null), balanced AUC, question-level AUC, the post-hoc best layer (an UPPER bound, uses the target labels), and the checks of the spec (section 8):
           does the probe still score F high in the shadow models (second edit), correlation of the score with the number of name tokens.

    python scripts/analysis/spa.py sets --out saves/analysis/spa/sets.json
    python scripts/analysis/spa.py collect --sets saves/analysis/spa/sets.json --model hf:<M''_k dir> --base open-unlearning/tofu_Llama-3.2-3B-Instruct_full --out saves/spa/3B_NEAR-Dist/shadow0.npz --gpu 0
    python scripts/analysis/spa.py attack --sets saves/analysis/spa/sets.json --shadow saves/spa/3B_NEAR-Dist/shadow{0,1,2,3,4}.npz --target saves/spa/3B_NEAR-Dist/target.npz --out saves/analysis/spa_3B_NEAR-Dist
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from typing import Dict, List, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

N_GROUPS = 5
GROUP_SIZE = 2


# ----------------------------------------------------------------------------- sets
def build_sets(seed: int = 0, s_split: str = "forget10", f_split: str = "forget05", holdouts=("holdout05", "holdout10"), templates_json: str = None, qa_per_author: int = 20):
    import rank_attack as ra
    s_all = ra.extract_names(s_split, qa_per_author)[:N_GROUPS * GROUP_SIZE]          # rows 0-199 of forget10 = authors 181-190
    f_names = ra.extract_names(f_split, qa_per_author)
    h_names: List[str] = []
    for h in holdouts:
        for n in ra.extract_names(h, qa_per_author):
            if n not in h_names and n not in f_names and n not in s_all:
                h_names.append(n)
    if len(h_names) < 2 * N_GROUPS * GROUP_SIZE:
        raise ValueError(f"only {len(h_names)} distinct holdout authors, {2 * N_GROUPS * GROUP_SIZE} are needed (H_A and H_B of 10)")
    rng = random.Random(seed)
    h_names = rng.sample(h_names, len(h_names))
    sets = validate_sets({
        "seed": seed, "s_split": s_split, "f_split": f_split, "holdouts": list(holdouts),
        "S_groups": [s_all[g * GROUP_SIZE:(g + 1) * GROUP_SIZE] for g in range(N_GROUPS)],
        "author_ids_S": [[181 + g * GROUP_SIZE + i for i in range(GROUP_SIZE)] for g in range(N_GROUPS)],
        "F": f_names, "H_A": h_names[:10], "H_B": h_names[10:20], "templates": None})
    if templates_json:
        sets["templates"] = json.load(open(templates_json, encoding="utf-8"))["templates"]
    return sets


def validate_sets(sets: dict) -> dict:
    S = [n for g in sets["S_groups"] for n in g]
    groups = {"S": S, "F": sets["F"], "H_A": sets["H_A"], "H_B": sets["H_B"]}
    seen = {}
    for tag, names in groups.items():
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate names in {tag}")
        for n in names:
            if n in seen:
                raise ValueError(f"'{n}' is in both {seen[n]} and {tag}: the sets must not overlap")
            seen[n] = tag
    return sets


def all_names(sets: dict) -> List[str]:
    return [n for g in sets["S_groups"] for n in g] + sets["F"] + sets["H_A"] + sets["H_B"]


# ----------------------------------------------------------------------------- collect
def collect_acts(model, tok, names: Sequence[str], templates: Sequence[str], batch_size: int = 16):
    """-> float32 [n_names, n_templates, n_layers, d]: output of each decoder layer at the last prompt token."""
    import torch
    import sample_consistency as sc
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    layers = model.model.layers
    store: Dict[int, torch.Tensor] = {}

    def mk(i):
        def hook(_m, _inp, out):
            o = out[0] if isinstance(out, tuple) else out
            store[i] = o[:, -1].detach().float().cpu()
        return hook
    handles = [layer.register_forward_hook(mk(i)) for i, layer in enumerate(layers)]
    prompts = sc.build_prompts(tok, [t.format(name=n) for n in names for t in templates])
    chunks = []
    try:
        model.eval()
        for i in range(0, len(prompts), batch_size):
            enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                model.model(**enc)
            chunks.append(torch.stack([store[j] for j in range(len(layers))], dim=1).numpy())
    finally:
        for h in handles:
            h.remove()
    x = np.concatenate(chunks, 0)
    return x.reshape(len(names), len(templates), x.shape[1], x.shape[2])


def cmd_collect(a):
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    import sample_consistency as sc
    from transformers import AutoTokenizer
    sets = json.load(open(a.sets, encoding="utf-8"))
    names = all_names(sets)
    model = sc.load_model(a.model, a.base)
    tok = AutoTokenizer.from_pretrained(a.base)
    x = collect_acts(model, tok, names, sets["templates"], a.batch_size)
    ntok = np.array([len(tok(" " + n, add_special_tokens=False).input_ids) for n in names])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, acts=x.astype(np.float16), names=np.array(names), templates=np.array(sets["templates"]), name_tokens=ntok, model=a.model)
    print(f"{a.out}: acts {x.shape} for {a.model}")


# ----------------------------------------------------------------------------- probes
def load_acts(path: str) -> dict:
    d = np.load(path, allow_pickle=False)
    names = [str(n) for n in d["names"]]
    return {"X": d["acts"].astype(np.float32), "names": names, "idx": {n: i for i, n in enumerate(names)}, "name_tokens": d["name_tokens"]}


def fit_probe(kind: str, X: np.ndarray, y: np.ndarray, C: float = 0.01, seed: int = 0):
    """-> score(X): larger = 'forgotten'. Standardisation is fitted on the training data only."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    if kind == "dm":                                                    # difference of the class means (no free parameter)
        w = Z[y == 1].mean(0) - Z[y == 0].mean(0)
        w = w / (np.linalg.norm(w) + 1e-12)
        return lambda Xn: ((Xn - mu) / sd) @ w
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(Z, y)
        return lambda Xn: m.decision_function((Xn - mu) / sd)
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        m = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2, early_stopping=True, validation_fraction=0.2, n_iter_no_change=10, max_iter=300, random_state=seed).fit(Z, y)
        return lambda Xn: m.predict_proba((Xn - mu) / sd)[:, 1]
    raise ValueError(kind)


def auc_score(pos: Sequence[float], neg: Sequence[float]) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def shadow_samples(shadows: List[dict], sets: dict, layer: int, exclude=()):
    """Training rows of one layer: (X, y, author) over the shadow models; S_k of model k = 1, all of H_A (in every model) = 0; authors in `exclude` left out."""
    X, y, who, model_of = [], [], [], []
    for k, sh in enumerate(shadows):
        for n in sets["S_groups"][k] + sets["H_A"]:
            if n in exclude:
                continue
            lab = 1 if n in sets["S_groups"][k] else 0
            v = sh["X"][sh["idx"][n], :, layer, :]
            X.append(v); y += [lab] * len(v); who += [n] * len(v); model_of += [k] * len(v)
    return np.concatenate(X, 0), np.array(y), who, model_of


def cv_curve(shadows: List[dict], sets: dict, kind: str = "lr", C: float = 0.01, seed: int = 0):
    """Leave-one-GROUP-out over the shadow models (shadow k's own S_k and two H_A authors are held out and scored in model k). -> per layer {pooled out-of-fold author AUC, mean within-fold AUC}.
    Everything here uses the shadow data only."""
    n_layers = shadows[0]["X"].shape[2]
    ha = sets["H_A"]
    folds = [(k, sets["S_groups"][k], ha[GROUP_SIZE * k:GROUP_SIZE * (k + 1)]) for k in range(len(shadows))]
    pooled, within = [], []
    for l in range(n_layers):
        scores, labels, fold_aucs = [], [], []
        for k, s_auth, h_auth in folds:
            X, y, _, _ = shadow_samples(shadows, sets, l, exclude=set(s_auth) | set(h_auth))
            f = fit_probe(kind, X, y, C, seed)
            sc_pos = [float(f(shadows[k]["X"][shadows[k]["idx"][n], :, l, :]).mean()) for n in s_auth]
            sc_neg = [float(f(shadows[k]["X"][shadows[k]["idx"][n], :, l, :]).mean()) for n in h_auth]
            scores += sc_pos + sc_neg; labels += [1] * len(sc_pos) + [0] * len(sc_neg); fold_aucs.append(auc_score(sc_pos, sc_neg))
        scores, labels = np.array(scores), np.array(labels)
        pooled.append(auc_score(scores[labels == 1], scores[labels == 0])); within.append(float(np.nanmean(fold_aucs)))
    return {"pooled": pooled, "within": within}


def author_scores(f, acts: dict, names: Sequence[str], layer: int):
    """score per (name, template) and per author (mean over the templates)."""
    q = np.stack([f(acts["X"][acts["idx"][n], :, layer, :]) for n in names])
    return q, q.mean(1)


def spearman(a, b) -> float:
    ra_, rb_ = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    if ra_.std() == 0 or rb_.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra_, rb_)[0, 1])


def _pool_stats(f_sc, h_sc, k, rng, max_subsets=300):
    """mean over the k-subsets of F (each + all of H) of AUC / precision@|F| / MRR, for SIGNED scores (larger = forgotten; ties broken at random by ra.ranking_metrics)."""
    import itertools
    import statistics as st
    import rank_attack as ra
    subsets = list(itertools.combinations(range(len(f_sc)), k))
    if len(subsets) > max_subsets:
        subsets = rng.sample(subsets, max_subsets)
    ms = [ra.ranking_metrics([f_sc[i] for i in sub] + list(h_sc), [1] * k + [0] * len(h_sc)) for sub in subsets]
    return {m: st.mean(x[m] for x in ms) for m in ("auc", "p_at_F", "mrr")}


def evaluate_transfer(f_scores: Sequence[float], h_scores: Sequence[float], k: int = 2, boot: int = 200, seed: int = 0) -> dict:
    """The statistic of A1 v2 on the SIGNED score of one probe: mean pool AUC over the k-subsets of F (+ all H_B); delete-one-author jackknife interval; 95th percentile of the
    label-permutation null; + the balanced (10 : 10) AUC. (rank_attack_v2.evaluate is two-sided - it takes |z| - and cannot be used for a directed probe score.)"""
    import statistics as st
    import rank_attack_v2 as v2
    f_scores, h_scores = [float(x) for x in f_scores], [float(x) for x in h_scores]
    base = _pool_stats(f_scores, h_scores, k, random.Random(seed))
    jk = []
    for i in range(len(f_scores)):
        jk.append(_pool_stats(f_scores[:i] + f_scores[i + 1:], h_scores, k, random.Random(seed))["auc"])
    for i in range(len(h_scores)):
        jk.append(_pool_stats(f_scores, h_scores[:i] + h_scores[i + 1:], k, random.Random(seed))["auc"])
    m = len(jk)
    se = (((m - 1) / m) * sum((x - st.mean(jk)) ** 2 for x in jk)) ** 0.5 if m > 1 else float("nan")
    allv, nf, rng, nulls = f_scores + h_scores, len(f_scores), random.Random(seed), []
    for _ in range(boot):
        perm = rng.sample(allv, len(allv))
        nulls.append(_pool_stats(perm[:nf], perm[nf:], k, random.Random(seed))["auc"])
    return {**base, "auc_lo": max(0.0, base["auc"] - 1.96 * se), "auc_hi": min(1.0, base["auc"] + 1.96 * se), "null95": v2.percentile(nulls, 0.95), "balanced_auc": auc_score(f_scores, h_scores)}


def run_attack(shadows: List[dict], target: dict, sets: dict, kinds=("lr", "dm"), C: float = 0.01, k: int = 2, boot: int = 200, seed: int = 0, layers=None) -> dict:
    n_layers = shadows[0]["X"].shape[2]
    layers = list(range(n_layers)) if layers is None else layers
    F, HB = sets["F"], sets["H_B"]
    out = {"n_layers": n_layers, "config": {"C": C, "k": k, "boot": boot, "seed": seed}, "probes": {}}
    for kind in kinds:
        curve = cv_curve(shadows, sets, kind, C, seed)
        sel = int(np.argmax(curve["pooled"]))                                        # the layer is chosen on the SHADOW data only
        by_layer = []
        for l in layers:                                                              # (all layers are fitted for the post-hoc upper bound; the label-free choice is `sel`)
            X, y, _, _ = shadow_samples(shadows, sets, l)
            f = fit_probe(kind, X, y, C, seed)
            _, a_f = author_scores(f, target, F, l)
            _, a_h = author_scores(f, target, HB, l)
            by_layer.append(auc_score(a_f, a_h))
        X, y, _, _ = shadow_samples(shadows, sets, sel)
        f = fit_probe(kind, X, y, C, seed)
        q_f, a_f = author_scores(f, target, F, sel)
        q_h, a_h = author_scores(f, target, HB, sel)
        ev = evaluate_transfer(a_f, a_h, k, boot, seed)
        # second-edit check: does the probe still score F above H_B inside the shadow models?
        in_shadow = []
        for sh in shadows:
            _, sf = author_scores(f, sh, F, sel)
            _, sh_ = author_scores(f, sh, HB, sel)
            in_shadow.append(auc_score(sf, sh_))
        ntok = np.array([target["name_tokens"][target["idx"][n]] for n in F + HB])
        out["probes"][kind] = {
            "selected_layer": sel, "shadow_cv_auc_selected": curve["pooled"][sel], "shadow_cv_within_selected": curve["within"][sel],
            "shadow_cv_curve": curve["pooled"], "transfer": ev,
            "question_auc": auc_score(q_f.ravel(), q_h.ravel()),
            "target_auc_by_layer": by_layer, "posthoc_best_layer": int(np.argmax(by_layer)), "posthoc_best_auc": float(np.max(by_layer)),
            "F_vs_HB_auc_inside_shadows": in_shadow, "name_token_corr": spearman(np.concatenate([a_f, a_h]), ntok),
            "author_scores": {"F": dict(zip(F, map(float, a_f))), "H_B": dict(zip(HB, map(float, a_h)))},
            "_question_scores": (q_f, q_h)}
    return out


def to_markdown(res: dict, title: str) -> str:
    lines = [f"# {title}", "", "| probe | shadow CV AUC (selected layer) | layer | transfer AUC [95% jackknife] (null95) | balanced AUC | P@F | MRR | question AUC | post-hoc best layer (upper bound) | F vs H_B inside shadows |", "|---|---|---|---|---|---|---|---|---|---|"]
    for kind, p in res["probes"].items():
        t = p["transfer"]
        lines.append(f"| {kind} | {p['shadow_cv_auc_selected']:.2f} | {p['selected_layer']} | {t['auc']:.2f} [{t['auc_lo']:.2f}, {t['auc_hi']:.2f}] ({t['null95']:.2f}) | {t['balanced_auc']:.2f} | {t['p_at_F']:.2f} | {t['mrr']:.2f} | "
                     f"{p['question_auc']:.2f} | L{p['posthoc_best_layer']}: {p['posthoc_best_auc']:.2f} | {', '.join(f'{x:.2f}' for x in p['F_vs_HB_auc_inside_shadows'])} |")
    lines += ["", "A transfer AUC counts as a detection only above the null95 band. The post-hoc best layer uses the target labels: it is an upper bound, not an attack. "
              "`F vs H_B inside shadows` is the second-edit check (the probe still ranks F above H_B in the doubly edited models?). "
              f"Name-token-length Spearman correlation of the score: " + ", ".join(f"{k}: {p['name_token_corr']:.2f}" for k, p in res["probes"].items()) + "."]
    return "\n".join(lines) + "\n"


def cmd_attack(a):
    sets = json.load(open(a.sets, encoding="utf-8"))
    shadows = [load_acts(p) for p in a.shadow]
    if len(shadows) != N_GROUPS:
        print(f"WARNING: {len(shadows)} shadow files for {N_GROUPS} groups (the group of file i is S_groups[i])")
    target = load_acts(a.target)
    res = run_attack(shadows, target, sets, a.probes, a.C, a.k, a.boot, a.seed)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "scores_target.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["probe", "name", "label", "template", "score"])
        for kind, p in res["probes"].items():
            q_f, q_h = p.pop("_question_scores")
            for lab, names, q in ((1, sets["F"], q_f), (0, sets["H_B"], q_h)):
                for n, row in zip(names, q):
                    for t, s in zip(sets["templates"], row):
                        w.writerow([kind, n, lab, t, f"{s:.5f}"])
    json.dump({kind: {"layer_curve": p["shadow_cv_curve"], "target_auc_by_layer": p["target_auc_by_layer"], "selected": p["selected_layer"]} for kind, p in res["probes"].items()},
              open(os.path.join(a.out, "probe_layer.json"), "w"), indent=1)
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    md = to_markdown(res, a.title or os.path.basename(os.path.normpath(a.out)))
    open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8").write(md)
    print(md)


def cmd_sets(a):
    sets = build_sets(a.seed, templates_json=a.templates)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(sets, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: sets[k] for k in ("S_groups", "F", "H_A", "H_B")}, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sets"); s.add_argument("--out", required=True); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--templates", default="saves/analysis/rank_attack2_templates/templates.json"); s.set_defaults(fn=cmd_sets)
    c = sub.add_parser("collect"); c.add_argument("--sets", required=True); c.add_argument("--model", required=True); c.add_argument("--base", required=True)
    c.add_argument("--out", required=True); c.add_argument("--gpu", default=None); c.add_argument("--batch-size", type=int, default=16); c.set_defaults(fn=cmd_collect)
    t = sub.add_parser("attack"); t.add_argument("--sets", required=True); t.add_argument("--shadow", nargs="+", required=True); t.add_argument("--target", required=True)
    t.add_argument("--out", required=True); t.add_argument("--probes", nargs="+", default=["lr", "dm"], choices=["lr", "dm", "mlp"]); t.add_argument("--C", type=float, default=0.01)
    t.add_argument("--k", type=int, default=2); t.add_argument("--boot", type=int, default=200); t.add_argument("--seed", type=int, default=0); t.add_argument("--title", default=None)
    t.set_defaults(fn=cmd_attack)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
