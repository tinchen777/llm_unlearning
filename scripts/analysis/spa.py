"""A2: the shadow-probe attack (SPA), spec: TODO/exp/A2_SPEC.md.  Three sub-commands, one file.

  sets     write the author sets (shadow_sets.json): S = authors 181-190 of forget10 in 5 groups of 2 (the retained authors that have neighbours and MEOW facts),
           F = forget05, H = the holdout05 + holdout10 authors split into H_A (shadow negatives) and H_B (target negatives), the 10 retain-derived question templates.
  collect  activations of one model: for every name x template the output of EVERY decoder layer at the last prompt token (= end of the instruction, the chat prompt ends with
           the assistant header; same hook point `block_out` as ud_wb_*) -> one npz. Run it on every shadow model M''_k, on the target M' and on the controls.
  attack   per layer a probe is trained on the shadow models only (S_k of M''_k = 1, H_A = 0), the layer is chosen by author-grouped leave-one-group-out CV on the shadow data
           ALONE, the probe is then applied to the target M' (F = 1, H_B = 0). Reported: shadow CV AUC, transfer AUC (author level, minority pools of 2 F + all H_B as in A1 v2, delete-one-
           author jackknife interval, label-permutation null), balanced AUC, question-level AUC, the post-hoc best layer (an UPPER bound, uses the target labels), and the checks of the spec (section 8):
           does the probe still score F high in the shadow models (second edit), correlation of the score with the number of name tokens.
           Layer rules (TODO/exp/21, none uses a target label): cv (the selected layer above), cv_deepest, post_edit_mean, pre_edit_mean; each also after a label-free name-length control
           (OLS residual on the number of name tokens over all candidates). -> summary.json results[probe]["rules"], table "Layer rules" in summary.md. Same for p0.

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
# Names the extractor gets wrong (the most frequent capitalised phrase of the author's rows is not the name): never used as an author.
BAD_NAMES = {"Cape Town"}


def build_sets(seed: int = 0, s_split: str = "forget10", f_split: str = "forget05", holdouts=("holdout05", "holdout10"), templates_json: str = None, qa_per_author: int = 20,
               r_split: str = "retain90", n_r: int = 10, r_near=None):
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
    # R = RETAINED authors (known to M', never edited by anybody): negatives of the fingerprint probe (R_A) and the second target pool F vs R_B (R_B)
    r_all = [n for n in ra.extract_names(r_split, qa_per_author) if n not in s_all and n not in f_names and n not in h_names]
    r_pick = random.Random(seed + 1).sample(r_all, 2 * n_r)
    near = {}
    if r_near:                                   # version 3 (TODO/exp/27): retained authors whose author ids are CLOSE to S / F (author id = row // 20 + 1 in retain90), the batch control
        lo, hi = r_near
        r_names = ra.extract_names(r_split, qa_per_author)
        taken = set(s_all) | set(f_names) | set(h_names) | set(r_pick) | BAD_NAMES
        pool = [n for i, n in enumerate(r_names) if lo <= i + 1 <= hi and n not in taken]
        if len(pool) < 2 * n_r:
            raise ValueError(f"only {len(pool)} usable retained authors with ids {lo}-{hi}; widen --r-near")
        pick = random.Random(seed + 2).sample(pool, 2 * n_r)
        near = {"r_near_ids": [lo, hi], "R_A_near": pick[:n_r], "R_B_near": pick[n_r:]}
    sets = validate_sets({
        "version": 3 if near else 2, "seed": seed, "s_split": s_split, "f_split": f_split, "holdouts": list(holdouts), "r_split": r_split,
        "S_groups": [s_all[g * GROUP_SIZE:(g + 1) * GROUP_SIZE] for g in range(N_GROUPS)],
        "author_ids_S": [[181 + g * GROUP_SIZE + i for i in range(GROUP_SIZE)] for g in range(N_GROUPS)],
        "F": f_names, "H_A": h_names[:10], "H_B": h_names[10:20], "R_A": r_pick[:n_r], "R_B": r_pick[n_r:], **near, "templates": None})
    if templates_json:
        sets["templates"] = json.load(open(templates_json, encoding="utf-8"))["templates"]
    return sets


def validate_sets(sets: dict) -> dict:
    S = [n for g in sets["S_groups"] for n in g]
    groups = {"S": S, "F": sets["F"], "H_A": sets["H_A"], "H_B": sets["H_B"], "R_A": sets.get("R_A", []), "R_B": sets.get("R_B", []),
              "R_A_near": sets.get("R_A_near", []), "R_B_near": sets.get("R_B_near", [])}
    seen = {}
    for tag, names in groups.items():
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate names in {tag}")
        for n in names:
            if n in seen:
                raise ValueError(f"'{n}' is in both {seen[n]} and {tag}: the sets must not overlap")
            seen[n] = tag
    return sets


def role_lists(sets: dict):
    return (("S", [n for g in sets["S_groups"] for n in g]), ("F", sets["F"]), ("H_A", sets["H_A"]), ("H_B", sets["H_B"]), ("R_A", sets.get("R_A", [])), ("R_B", sets.get("R_B", [])),
            ("R_A_near", sets.get("R_A_near", [])), ("R_B_near", sets.get("R_B_near", [])))


def all_names(sets: dict) -> List[str]:
    return ([n for g in sets["S_groups"] for n in g] + sets["F"] + sets["H_A"] + sets["H_B"] + sets.get("R_A", []) + sets.get("R_B", [])
            + sets.get("R_A_near", []) + sets.get("R_B_near", []))


def with_r_pools(sets: dict, ra_pool: str = "random", rb_pool: str = "random") -> dict:
    """A copy of the sets whose R_A (negatives of P2, positives of P0) / R_B (the Q2 target pool) come from the `random` retained authors (ids 1-180, version 2) or the
    `near` ones (ids close to S / F, version 3; TODO/exp/27, the author-batch control)."""
    out = dict(sets)
    for key, pool in (("R_A", ra_pool), ("R_B", rb_pool)):
        if pool == "near":
            if not sets.get(key + "_near"):
                raise SystemExit(f"{key}=near needs a sets file of version 3 (spa.py sets --r-near LO HI)")
            out[key] = list(sets[key + "_near"])
        elif pool != "random":
            raise ValueError(pool)
    out["r_pools"] = {"R_A": ra_pool, "R_B": rb_pool}
    return out


def needed_names(sets: dict, with_shadows: bool):
    """(names every shadow file must have, names the target file must have) for an attack with these (already re-pooled) sets."""
    S = [n for g in sets["S_groups"] for n in g]
    shadow = S + list(sets["H_A"]) + list(sets.get("R_A", [])) if with_shadows else []
    target = list(sets["F"]) + list(sets["H_B"]) + list(sets.get("R_B", [])) + list(sets.get("R_A", [])) + list(sets["H_A"])
    return shadow, target


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
    old = None
    if a.extend and os.path.isfile(a.out):       # only the names the file lacks (e.g. the near-R authors of a version-3 sets file), appended to the existing activations
        old = np.load(a.out, allow_pickle=False)
        have = [str(n) for n in old["names"]]
        if [str(t) for t in old["templates"]] != list(sets["templates"]):
            sys.exit(f"{a.out}: other templates than {a.sets}; cannot extend")
        names = [n for n in names if n not in set(have)]
        if not names:
            print(f"{a.out}: has every name of {a.sets}, nothing to do")
            return
        print(f"{a.out}: extending by {len(names)} names: {names}")
    model = sc.load_model(a.model, a.base)
    tok = AutoTokenizer.from_pretrained(a.base)
    x = collect_acts(model, tok, names, sets["templates"], a.batch_size)
    ntok = np.array([len(tok(" " + n, add_special_tokens=False).input_ids) for n in names])
    probs = {} if a.no_prob else author_probs(model, tok, {**sets, "_only": names})
    if old is not None:
        x = np.concatenate([old["acts"].astype(np.float32), x], 0)
        ntok = np.concatenate([old["name_tokens"], ntok])
        old_prob = json.loads(str(old["prob"])) if "prob" in old.files else {}
        probs = {**old_prob, **probs}
        names = [str(n) for n in old["names"]] + names
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, acts=x.astype(np.float16), names=np.array(names), templates=np.array(sets["templates"]), name_tokens=ntok, model=a.model, prob=np.array(json.dumps(probs)))
    show = {t: round(float(np.mean([probs[n] for n in ns if n in probs])), 3) for t, ns in role_lists(sets) if probs and ns}
    print(f"{a.out}: acts {x.shape} for {a.model}; mean proxy answer prob by role: {show}")


def author_probs(model, tok, sets: dict, qa_per_author: int = 20, batch_size: int = 8, max_q: int = 20) -> Dict[str, float]:
    """PROXY of forget_Q_A_Prob per author: mean over the author's QA of exp(mean log-prob of the true answer tokens) under the chat prompt of sample_consistency.build_prompts
    (not the evaluation pipeline's exact number; for the shadow-strength diagnostic and the comparison with the matched level 0.25 only)."""
    import torch
    import rank_attack as ra
    import sample_consistency as sc
    wanted = set(sets.get("_only") or all_names(sets))
    out: Dict[str, float] = {}
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for split in ["forget10", "holdout10", "holdout05", sets.get("r_split", "retain90")]:
        qs, ans = sc.load_questions(split)
        names = ra.extract_names(split, qa_per_author)
        for a_i, n in enumerate(names):
            if n not in wanted or n in out:
                continue
            q, an = qs[a_i * qa_per_author:(a_i + 1) * qa_per_author][:max_q], ans[a_i * qa_per_author:(a_i + 1) * qa_per_author][:max_q]
            prompts = sc.build_prompts(tok, q)
            ps = []
            for i in range(0, len(q), batch_size):
                pr, aa = prompts[i:i + batch_size], an[i:i + batch_size]
                p_ids = [tok(x, add_special_tokens=False).input_ids for x in pr]
                a_ids = [tok(x, add_special_tokens=False).input_ids for x in aa]
                L = max(len(p) + len(a) for p, a in zip(p_ids, a_ids))
                ids = torch.full((len(pr), L), tok.pad_token_id); att = torch.zeros((len(pr), L), dtype=torch.long); mask = torch.zeros((len(pr), L), dtype=torch.bool)
                for j, (p, a) in enumerate(zip(p_ids, a_ids)):
                    ids[j, :len(p) + len(a)] = torch.tensor(p + a); att[j, :len(p) + len(a)] = 1; mask[j, len(p):len(p) + len(a)] = True
                with torch.no_grad():
                    logits = model(input_ids=ids.to(model.device), attention_mask=att.to(model.device)).logits.float()
                lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:].to(model.device).unsqueeze(-1)).squeeze(-1)
                m = mask[:, 1:].to(model.device)
                ps += torch.exp((lp * m).sum(1) / m.sum(1)).tolist()
            out[n] = float(np.mean(ps))
    return out


# ----------------------------------------------------------------------------- probes
def load_acts(path: str) -> dict:
    d = np.load(path, allow_pickle=False)
    names = [str(n) for n in d["names"]]
    prob = json.loads(str(d["prob"])) if "prob" in d.files else {}
    return {"X": d["acts"].astype(np.float32), "names": names, "idx": {n: i for i, n in enumerate(names)}, "name_tokens": d["name_tokens"], "prob": prob}


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


PROBES = ("P0-familiarity", "P1-spec", "P2-fingerprint")
# P0  NON-adaptive baseline, no shadow model: on the target M' itself, retained authors R_A (known) = 1 vs strangers H_A = 0. It can only learn "this entity is known".
# P1  the spec: in shadow M''_k, S_k (known, then edited) = 1 vs strangers H_A = 0. Free to learn "known" instead of "edited".
# P2  fingerprint: in M''_k, S_k = 1 vs H_A + R_A + the other groups S_j (known and NOT edited in model k) = 0: "known" no longer separates the classes, only "edited" does.
# Pools on the target: Q1 = F vs H_B (strangers: is M' distinguishable from the retain model? includes residual recognition), Q2 = F vs R_B (retained, known authors: the edit fingerprint).


def shadow_samples(shadows: List[dict], sets: dict, layer: int, exclude=(), variant: str = "P1-spec"):
    """Training rows of one layer over the shadow models: S_k of model k = 1; negatives H_A (P1) or H_A + R_A + S_j, j != k (P2); authors in `exclude` left out."""
    X, y, who, model_of = [], [], [], []
    for k, sh in enumerate(shadows):
        neg = list(sets["H_A"])
        if variant == "P2-fingerprint":
            neg += list(sets["R_A"]) + [n for j, g in enumerate(sets["S_groups"]) if j != k for n in g]
        for n in list(sets["S_groups"][k]) + neg:
            if n in exclude:
                continue
            lab = 1 if n in sets["S_groups"][k] else 0
            v = sh["X"][sh["idx"][n], :, layer, :]
            X.append(v); y += [lab] * len(v); who += [n] * len(v); model_of += [k] * len(v)
    return np.concatenate(X, 0), np.array(y), who, model_of


def familiarity_samples(target: dict, sets: dict, layer: int, exclude=()):
    """P0: rows of the TARGET model, R_A = 1 (known) vs H_A = 0 (strangers)."""
    names = [n for n in list(sets["R_A"]) + list(sets["H_A"]) if n not in exclude]
    X = np.concatenate([target["X"][target["idx"][n], :, layer, :] for n in names], 0)
    y = np.concatenate([np.full(target["X"].shape[1], 1 if n in sets["R_A"] else 0) for n in names])
    return X, y


def cv_curve(shadows: List[dict], sets: dict, kind: str = "lr", C: float = 0.01, seed: int = 0, variant: str = "P1-spec", target: dict = None):
    """Per-layer CV AUC on the training data of the probe ONLY (no F / H_B / R_B). P1 / P2: leave-one-GROUP-out over the shadow models (fold k: S_k + 2 H_A (+ 2 R_A for P2) held out, scored
    in model k). P0: 5 folds of 2 R_A + 2 H_A on the target. -> {pooled out-of-fold author AUC, mean within-fold AUC} per layer."""
    n_layers = (target if variant == "P0-familiarity" else shadows[0])["X"].shape[2]
    ha, rA = sets["H_A"], sets.get("R_A", [])
    g = GROUP_SIZE
    pooled, within = [], []
    for l in range(n_layers):
        scores, labels, fold_aucs = [], [], []
        for k in range(N_GROUPS if variant == "P0-familiarity" else len(shadows)):
            h_auth = list(ha[g * k:g * (k + 1)])
            if variant == "P0-familiarity":
                pos, neg = list(rA[g * k:g * (k + 1)]), h_auth
                X, y = familiarity_samples(target, sets, l, exclude=set(pos) | set(neg))
                model = target
            else:
                pos = list(sets["S_groups"][k])
                neg = h_auth + (list(rA[g * k:g * (k + 1)]) if variant == "P2-fingerprint" else [])
                X, y, _, _ = shadow_samples(shadows, sets, l, exclude=set(pos) | set(neg), variant=variant)
                model = shadows[k]
            f = fit_probe(kind, X, y, C, seed)
            sp = [float(f(model["X"][model["idx"][n], :, l, :]).mean()) for n in pos]
            sn = [float(f(model["X"][model["idx"][n], :, l, :]).mean()) for n in neg]
            scores += sp + sn; labels += [1] * len(sp) + [0] * len(sn); fold_aucs.append(auc_score(sp, sn))
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


def _fit(variant, kind, shadows, target, sets, l, C, seed):
    """-> (probe, mean, sd of the probe's scores on its OWN training rows): the scale used to put the layers of a multi-layer rule on a common footing (never the target's scores)."""
    if variant == "P0-familiarity":
        X, y = familiarity_samples(target, sets, l)
    else:
        X, y, _, _ = shadow_samples(shadows, sets, l, variant=variant)
    f = fit_probe(kind, X, y, C, seed)
    s = np.asarray(f(X), float)
    return f, float(s.mean()), float(s.std())


CV_TIE = 0.005          # cv_deepest: layers whose CV AUC is within this of the maximum count as tied
LAYER_RULES = ("cv", "cv_deepest", "post_edit_mean", "pre_edit_mean")
# cv              the layer of maximal CV AUC, first of the ties (the original rule; sensitivity analysis)
# cv_deepest      the DEEPEST layer whose CV AUC is within CV_TIE of the maximum (the CV is often 1.00 from the first edited layer on; the first of the ties sits in the 'valley' next to the edit)
# post_edit_mean  all layers AFTER the last edited layer: the per-layer probes' author scores, each standardised by the mean / sd of that probe on its own training rows, averaged (edit trace)
# pre_edit_mean   all layers BEFORE the first edited layer, same averaging: activations there equal the original model's, so this measures residual recognition, not the edit
# None of them looks at a target label: the edited layers are a hyper-parameter of the algorithm, known to the attacker.  Without edited layers only cv / cv_deepest.


def rule_layers(cv_pooled: Sequence[float], layers: Sequence[int], edit_layers: Sequence[int], tie: float = CV_TIE) -> Dict[str, List[int]]:
    """layers used by each rule (rules with no layer are left out)."""
    c = np.asarray(cv_pooled, float)
    best = np.nanmax(c)
    rules = {"cv": [int(np.argmax(c))], "cv_deepest": [int(max(i for i, v in enumerate(c) if v >= best - tie))]}
    if edit_layers:
        post = [int(l) for l in layers if l > max(edit_layers)]
        pre = [int(l) for l in layers if l < min(edit_layers)]
        if post:
            rules["post_edit_mean"] = post
        if pre:
            rules["pre_edit_mean"] = pre
    return rules


def length_residual(scores: Sequence[float], ntok: Sequence[float]):
    """Name-length control without labels: OLS of the score on the number of name tokens (intercept + slope) over ALL candidates together, -> (residuals, slope).
    Constant token counts -> the scores unchanged (slope None)."""
    s, x = np.asarray(scores, float), np.asarray(ntok, float)
    if np.ptp(x) == 0:
        return s.copy(), None
    A = np.stack([np.ones_like(x), x], 1)
    beta, *_ = np.linalg.lstsq(A, s, rcond=None)
    return s - A @ beta, float(beta[1])


def run_attack(shadows: List[dict], target: dict, sets: dict, kinds=("lr", "dm"), C: float = 0.01, k: int = 2, boot: int = 200, seed: int = 0, layers=None,
               probes=PROBES, edit_layers: Sequence[int] = ()) -> dict:
    n_layers = (shadows[0] if shadows else target)["X"].shape[2]
    layers = list(range(n_layers)) if layers is None else layers
    F, HB, RB = sets["F"], sets["H_B"], sets.get("R_B", [])
    pools = {"Q1": HB} if not RB else {"Q1": HB, "Q2": RB}
    if not sets.get("R_A"):
        probes = [p for p in probes if p == "P1-spec"]                      # sets of version 1: only the spec probe
    if not shadows:
        probes = [p for p in probes if p == "P0-familiarity"]               # no shadow model: only the familiarity baseline

    def mean_prob(acts, names):
        v = [acts["prob"][n] for n in names if n in acts.get("prob", {})]
        return float(np.mean(v)) if v else None
    pre_edit = [l for l in layers if edit_layers and l < min(edit_layers)]
    roles = {"F": F, **pools}
    cand = F + [n for neg in pools.values() for n in neg]                      # all candidates of the target (F, H_B, R_B): the population of the length regression
    ntok_cand = np.array([target["name_tokens"][target["idx"][n]] for n in cand], float)
    out = {"n_layers": n_layers, "edit_layers": list(edit_layers), "config": {"C": C, "k": k, "boot": boot, "seed": seed},
           "strength": {"shadow_S_prob": [mean_prob(sh, sets["S_groups"][i]) for i, sh in enumerate(shadows)], "shadow_HA_prob": [mean_prob(sh, sets["H_A"]) for sh in shadows],
                        "shadow_F_prob": [mean_prob(sh, F) for sh in shadows], "target": {r: mean_prob(target, ns) for r, ns in role_lists(sets) if ns}}, "results": {}}
    for variant in probes:
        for kind in kinds:
            curve = cv_curve(shadows, sets, kind, C, seed, variant=variant, target=target)
            sel = int(np.argmax(curve["pooled"]))                                   # chosen on the probe's own training data only
            fits = {}                                                                # layer -> (probe, train mean, train sd, {role: (question, author) scores}); fitted ONCE per layer

            def layer_fit(l):
                if l not in fits:
                    f_, mu_, sd_ = _fit(variant, kind, shadows, target, sets, l, C, seed)
                    fits[l] = (f_, mu_, sd_, {role: author_scores(f_, target, ns, l) for role, ns in roles.items()})
                return fits[l]
            by_layer = {q: [] for q in pools}
            for l in layers:                                                         # post-hoc curves (use target labels: upper bound / diagnostics only)
                sc_ = layer_fit(l)[3]
                for q in pools:
                    by_layer[q].append(auc_score(sc_["F"][1], sc_[q][1]))
            f = layer_fit(sel)[0]
            q_f, a_f = fits[sel][3]["F"]
            r = {"selected_layer": sel, "cv_auc_selected": curve["pooled"][sel], "cv_within_selected": curve["within"][sel], "cv_curve": curve["pooled"], "pools": {}, "_q": {"F": q_f}}
            for q, neg in pools.items():
                q_n, a_n = author_scores(f, target, neg, sel)
                r["pools"][q] = {**evaluate_transfer(a_f, a_n, k, boot, seed), "question_auc": auc_score(q_f.ravel(), q_n.ravel()),
                                 "auc_by_layer": by_layer[q], "posthoc_best_layer": int(np.argmax(by_layer[q])), "posthoc_best_auc": float(np.max(by_layer[q])),
                                 "pre_edit_mean_auc": float(np.mean([by_layer[q][layers.index(l)] for l in pre_edit])) if pre_edit else None}
                r["_q"][q] = q_n
            if variant != "P0-familiarity":                                          # second-edit check: F vs H_B inside the shadow models
                r["Q1_inside_shadows"] = [auc_score(author_scores(f, sh, F, sel)[1], author_scores(f, sh, HB, sel)[1]) for sh in shadows]
            ntok = np.array([target["name_tokens"][target["idx"][n]] for n in F + HB])
            r["name_token_corr_Q1"] = spearman(np.concatenate([a_f, author_scores(f, target, HB, sel)[1]]), ntok)
            r["author_scores"] = {"F": dict(zip(F, map(float, a_f))), **{nm: dict(zip(neg, map(float, author_scores(f, target, neg, sel)[1]))) for nm, neg in (("H_B", HB), ("R_B", RB)) if neg}}
            r["rules"] = {}
            for rule, ls in rule_layers(curve["pooled"], layers, edit_layers).items():
                if len(ls) == 1:                                                     # one layer: the raw score (standardising changes no AUC)
                    sc_ = {role: layer_fit(ls[0])[3][role][1] for role in roles}
                else:
                    sc_ = {role: np.mean([(layer_fit(l)[3][role][1] - fits[l][1]) / (fits[l][2] + 1e-12) for l in ls], 0) for role in roles}
                allv = np.concatenate([sc_[role] for role in roles])
                resid, slope = length_residual(allv, ntok_cand)
                cut = np.cumsum([0] + [len(roles[role]) for role in roles])
                rs = {role: resid[cut[i]:cut[i + 1]] for i, role in enumerate(roles)}
                r["rules"][rule] = {"layers": ls, "cv_auc": [curve["pooled"][l] for l in ls], "length_slope": slope,
                                    "name_token_corr": spearman(allv, ntok_cand), "name_token_corr_corrected": spearman(resid, ntok_cand),
                                    "pools": {q: evaluate_transfer(sc_["F"], sc_[q], k, boot, seed) for q in pools},
                                    "pools_length_corrected": {q: evaluate_transfer(rs["F"], rs[q], k, boot, seed) for q in pools}}
            out["results"][f"{variant}/{kind}"] = r
    return out


def fmt_layers(ls: Sequence[int]) -> str:
    """[4, 5, 6, 9] -> 'L4-6, L9'"""
    ls, parts, i = sorted(ls), [], 0
    while i < len(ls):
        j = i
        while j + 1 < len(ls) and ls[j + 1] == ls[j] + 1:
            j += 1
        parts.append(f"L{ls[i]}" if i == j else f"L{ls[i]}-{ls[j]}"); i = j + 1
    return ", ".join(parts)


def _cell(t, interval=True):
    if t is None:
        return "-"
    return f"{t['auc']:.2f} [{t['auc_lo']:.2f}, {t['auc_hi']:.2f}] ({t['null95']:.2f})" if interval else f"{t['auc']:.2f} ({t['null95']:.2f})"


RULE_HEAD = "rule | layers | Q1 AUC [95% jackknife] (null95) | Q2 AUC (null95) | Q1 length-corrected [jackknife] (null95) | Q2 length-corrected (null95) |"
RULE_NOTE = ("Layer rules (none uses a target label; the edited layers are a known hyper-parameter): cv = max CV AUC, first of the ties; cv_deepest = deepest layer within "
             f"{CV_TIE} of the max CV AUC; post_edit_mean = all layers after the last edited layer (edit trace); pre_edit_mean = all layers before the first edited layer (residual recognition). "
             "Multi-layer rules average the per-layer author scores, each standardised by its probe's scores on its own training rows. Length-corrected = residual of an OLS of the "
             "score on the number of name tokens over all candidates (F, H_B, R_B) together. Main results: post_edit_mean / pre_edit_mean, length-corrected; cv / cv_deepest = sensitivity. "
             "A detection needs AUC > null95 AND clearly above the retain control under the same rule.")


def rule_rows(r: dict, lead: str) -> List[str]:
    rows = []
    for rule, t in r.get("rules", {}).items():
        p, pl = t["pools"], t["pools_length_corrected"]
        rows.append(f"| {lead} | {rule} | {fmt_layers(t['layers'])} | {_cell(p.get('Q1'))} | {_cell(p.get('Q2'), False)} | {_cell(pl.get('Q1'))} | {_cell(pl.get('Q2'), False)} |")
    return rows


def to_markdown(res: dict, title: str) -> str:
    qs = sorted({q for r in res["results"].values() for q in r["pools"]})
    head = "| probe | CV AUC on its own training data | layer | " + " | ".join(f"{q} AUC [95% jackknife] (null95) | {q} post-hoc best | {q} mean before the edit layers" for q in qs) + " | Q1 inside shadows |"
    lines = [f"# {title}", "", "Q1 = F vs strangers H_B (is M' distinguishable from the retain model; includes residual recognition). Q2 = F vs retained known authors R_B (the edit fingerprint).",
             "P0 = familiarity baseline on M' (no shadow), P1 = spec (S_k vs strangers), P2 = fingerprint (S_k vs strangers + retained + other S_j).", "", head,
             "|" + "---|" * (3 + 3 * len(qs) + 1)]
    for name, r in res["results"].items():
        cells = []
        for q in qs:
            t = r["pools"].get(q)
            if t is None:
                cells += ["-"] * 3; continue
            pe = "-" if t["pre_edit_mean_auc"] is None else f"{t['pre_edit_mean_auc']:.2f}"
            cells += [f"{t['auc']:.2f} [{t['auc_lo']:.2f}, {t['auc_hi']:.2f}] ({t['null95']:.2f})", f"L{t['posthoc_best_layer']}: {t['posthoc_best_auc']:.2f}", pe]
        ins = ", ".join(f"{x:.2f}" for x in r.get("Q1_inside_shadows", [])) or "-"
        lines.append(f"| {name} | {r['cv_auc_selected']:.2f} | {r['selected_layer']} | " + " | ".join(cells) + f" | {ins} |")
    lines += ["", f"Edit layers {res.get('edit_layers')}: before them the activations of M' equal those of the original model, so an AUC there is recognition (or noise), never an edit trace; "
              "for P2 / Q2 it should be ~0.5. A pool AUC counts as a detection only above its null95. Post-hoc best layers use the target labels (upper bound).",
              "Name-token-length Spearman (Q1): " + ", ".join(f"{n}: {r['name_token_corr_Q1']:.2f}" for n, r in res["results"].items()) + ".",
              "", f"Strength (answer-probability proxy, mean over authors): {json.dumps(res.get('strength'))}"]
    if any(r.get("rules") for r in res["results"].values()):
        lines += ["", "## Layer rules", "", RULE_NOTE, "", "| probe | " + RULE_HEAD, "|" + "---|" * 7]
        for name, r in res["results"].items():
            lines += rule_rows(r, name)
    return "\n".join(lines) + "\n"


def cmd_attack(a):
    sets = with_r_pools(json.load(open(a.sets, encoding="utf-8")), a.ra_pool, a.rb_pool)
    shadows = [load_acts(p) for p in a.shadow]
    if len(shadows) != N_GROUPS:
        print(f"WARNING: {len(shadows)} shadow files for {N_GROUPS} groups (the group of file i is S_groups[i])")
    target = load_acts(a.target)
    need_sh, need_t = needed_names(sets, True)
    missing = [n for n in need_t if n not in target["idx"]] + [n for n in need_sh if any(n not in sh["idx"] for sh in shadows)]
    if missing:
        sys.exit(f"{len(missing)} names of {a.sets} are not in the activation files (collected with an older sets file?), e.g. {missing[:3]}: re-run collect")
    res = run_attack(shadows, target, sets, a.kinds, a.C, a.k, a.boot, a.seed, probes=a.probes, edit_layers=a.edit_layers)
    res["r_pools"] = sets.get("r_pools")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "scores_target.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["probe", "role", "name", "template", "score"])
        for name, r in res["results"].items():
            qd = r.pop("_q")
            for role, names, key in (("F", sets["F"], "F"), ("H_B", sets["H_B"], "Q1"), ("R_B", sets.get("R_B", []), "Q2")):
                if key not in qd:
                    continue
                for n, row in zip(names, qd[key]):
                    for t, sc_ in zip(sets["templates"], row):
                        w.writerow([name, role, n, t, f"{sc_:.5f}"])
    json.dump({n: {"cv_curve": r["cv_curve"], "selected": r["selected_layer"], **{f"{q}_auc_by_layer": t["auc_by_layer"] for q, t in r["pools"].items()}} for n, r in res["results"].items()},
              open(os.path.join(a.out, "probe_layer.json"), "w"), indent=1)
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    md = to_markdown(res, a.title or os.path.basename(os.path.normpath(a.out)))
    open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8").write(md)
    print(md)


def cmd_p0(a):
    """P0 only (no shadow model): for every target model, recognition probe R_A (known) vs H_A (strangers) trained on THAT model, layer by CV on R_A / H_A, applied to Q1 (F vs strangers)
    and Q2 (F vs retained). One table for all models. --acts label=file.npz[#edited,layers]"""
    sets = with_r_pools(json.load(open(a.sets, encoding="utf-8")), a.ra_pool, a.rb_pool)
    rows, rrows, out = [], [], {}
    for spec in a.acts:
        label, _, rest = spec.partition("=")
        path, _, el = rest.partition("#")
        edit_layers = [int(x) for x in el.split(",") if x != ""]
        target = load_acts(path)
        res = run_attack([], target, sets, a.kinds, a.C, a.k, a.boot, a.seed, probes=("P0-familiarity",), edit_layers=edit_layers)
        for name, r in res["results"].items():
            r.pop("_q", None)
            kind = name.split("/")[1]
            q1, q2 = r["pools"]["Q1"], r["pools"].get("Q2")
            pe = q1["pre_edit_mean_auc"]
            q2_cell = "-" if q2 is None else "%.2f (%.2f)" % (q2["auc"], q2["null95"])
            pe_cell = "-" if pe is None else "%.2f" % pe
            q1_cell = "%.2f [%.2f, %.2f] (%.2f)" % (q1["auc"], q1["auc_lo"], q1["auc_hi"], q1["null95"])
            rows.append(f"| {label} | {kind} | {r['cv_auc_selected']:.2f} | {r['selected_layer']} | {q1_cell} | {q2_cell} | "
                        f"L{q1['posthoc_best_layer']}: {q1['posthoc_best_auc']:.2f} | {pe_cell} | {edit_layers or 'all'} |")
            rrows += rule_rows(r, f"{label} | {kind}")
        out[label] = res
    md = ["# A2 P0: recognition probe on each model (no shadow model)", "",
          "Trained on the model itself: retained authors R_A (known) = 1 vs strangers H_A = 0; layer chosen by CV on R_A / H_A only. Q1 = F vs strangers H_B (on the retain model F are strangers: ~0.5 expected);"
          " Q2 = F vs retained authors R_B (a pure recognition probe ranks F BELOW R_B if F is forgotten: < 0.5). A Q1 AUC counts only above its null95. `before the edit` = mean Q1 AUC over the layers below"
          " the first edited layer, where the activations equal those of the original model.", "",
          "| model | probe | CV AUC (R_A vs H_A) | layer | Q1 AUC [95% jackknife] (null95) | Q2 AUC (null95) | Q1 post-hoc best | Q1 mean before the edit | edited layers |",
          "|---|---|---|---|---|---|---|---|---|"] + rows
    md += ["", "## Layer rules", "", RULE_NOTE, "", "| model | probe | " + RULE_HEAD, "|" + "---|" * 8] + rrows
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("\n".join(md))


def cmd_strength(a):
    """Small table from the (large) npz files: mean answer-probability proxy per role for every file, plus the name-token counts. Upload this, not the npz."""
    sets = json.load(open(a.sets, encoding="utf-8"))
    roles = tuple((r, ns) for r, ns in role_lists(sets) if ns)
    lines = ["| file | " + " | ".join(r for r, _ in roles) + " | S_k (own group) |", "|---|" + "---|" * (len(roles) + 1)]
    for path in a.npz:
        acts = load_acts(path)
        pr = acts.get("prob", {})
        cells = [f"{np.mean([pr[n] for n in ns if n in pr]):.3f}" if any(n in pr for n in ns) else "-" for _, ns in roles]
        own = "-"
        m = __import__("re").search(r"shadow(\d+)", os.path.basename(path))
        if m and pr and int(m.group(1)) < len(sets["S_groups"]):
            own = f"{np.mean([pr[n] for n in sets['S_groups'][int(m.group(1))] if n in pr]):.3f}"
        lines.append(f"| {path} | " + " | ".join(cells) + f" | {own} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        open(a.out, "w", encoding="utf-8").write(text)


def cmd_sets(a):
    sets = build_sets(a.seed, templates_json=a.templates, r_near=tuple(a.r_near) if a.r_near else None)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(sets, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: sets[k] for k in ("S_groups", "F", "H_A", "H_B", "R_A", "R_B", "R_A_near", "R_B_near") if k in sets}, ensure_ascii=False, indent=1))
    print("CHECK by eye that every R name above is a PERSON (the extractor can return a place, e.g. 'Cape Town'; add such names to BAD_NAMES)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sets"); s.add_argument("--out", required=True); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--templates", default="saves/analysis/rank_attack2_templates/templates.json")
    s.add_argument("--r-near", type=int, nargs=2, default=None, metavar=("LO", "HI"), help="version 3: + R_A_near / R_B_near = retained authors with ids LO..HI (TODO/exp/27)"); s.set_defaults(fn=cmd_sets)
    c = sub.add_parser("collect"); c.add_argument("--sets", required=True); c.add_argument("--model", required=True); c.add_argument("--base", required=True)
    c.add_argument("--out", required=True); c.add_argument("--gpu", default=None); c.add_argument("--batch-size", type=int, default=16)
    c.add_argument("--no-prob", action="store_true", help="skip the answer-probability proxy")
    c.add_argument("--extend", action="store_true", help="if --out exists: collect only the names it lacks and append them"); c.set_defaults(fn=cmd_collect)
    g = sub.add_parser("strength"); g.add_argument("--sets", required=True); g.add_argument("--npz", nargs="+", required=True); g.add_argument("--out", default=None); g.set_defaults(fn=cmd_strength)
    q = sub.add_parser("p0"); q.add_argument("--sets", required=True); q.add_argument("--acts", nargs="+", required=True, help="label=file.npz[#l1,l2,..] (edited layers)")
    q.add_argument("--out", required=True); q.add_argument("--kinds", nargs="+", default=["lr", "dm"], choices=["lr", "dm", "mlp"]); q.add_argument("--C", type=float, default=0.01)
    q.add_argument("--k", type=int, default=2); q.add_argument("--boot", type=int, default=200); q.add_argument("--seed", type=int, default=0)
    q.add_argument("--ra-pool", choices=["random", "near"], default="random"); q.add_argument("--rb-pool", choices=["random", "near"], default="random"); q.set_defaults(fn=cmd_p0)
    t = sub.add_parser("attack"); t.add_argument("--sets", required=True); t.add_argument("--shadow", nargs="+", required=True); t.add_argument("--target", required=True)
    t.add_argument("--out", required=True); t.add_argument("--kinds", nargs="+", default=["lr", "dm"], choices=["lr", "dm", "mlp"]); t.add_argument("--C", type=float, default=0.01)
    t.add_argument("--probes", nargs="+", default=list(PROBES), choices=list(PROBES)); t.add_argument("--edit-layers", nargs="*", type=int, default=[], help="layers the method edits (marks the pre-edit layers)")
    t.add_argument("--k", type=int, default=2); t.add_argument("--boot", type=int, default=200); t.add_argument("--seed", type=int, default=0); t.add_argument("--title", default=None)
    t.add_argument("--ra-pool", choices=["random", "near"], default="random", help="R_A (P2 negatives) from the random (v2) or near (v3) retained authors")
    t.add_argument("--rb-pool", choices=["random", "near"], default="random", help="R_B (the Q2 pool) from the random (v2) or near (v3) retained authors")
    t.set_defaults(fn=cmd_attack)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
