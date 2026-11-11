"""A1 v2: the black-box repeated-query ranking attack, upgraded after the first run (TODO/paper_notes/10 §6).

What changed with respect to rank_attack.py
  1. The forgotten authors are a MINORITY of every pool (k of them, default 2, + all the holdout authors; every k-subset is one pool). A median / MAD standardisation inside a pool
     where half of the authors are shifted has no direction (v1, balanced 10 : 10).
  2. The query templates can be DERIVED FROM THE RETAINED DATA (`--templates retain`): the most frequent question forms of the retain authors with the author name replaced by {name} (the
     attacker holds the retain data; nothing about the forgotten authors enters). The first version used hand-written templates whose wording differs from the TOFU questions, so even
     the full model was hardly detectable (the positive control).
  3. A second attacker who sees LOGITS (`L6`): mean token log-probability and mean entropy of the greedy answers, and the spread of the log-probability over the templates.
  4. Uncertainty: the statistic is the mean over all k-subsets of the pool AUC; its 95 % interval comes from a delete-one-AUTHOR jackknife (authors are the independent units, the pools share
     them) and the 95th percentile of a NULL obtained by PERMUTING the forgotten / holdout labels over the authors (`--boot` permutations): an AUC is a detection only above that band.
  5. Ties are broken at random in precision@|F| / MRR (the stable sort of v1 favoured the forgotten authors on constant scores).
  6. Larger holdout pool: `--extra-holdout holdout10` adds the authors of another holdout split (names already in the pool, or equal to a forgotten author, are dropped).
The feature definitions and the label-free score (max over the features of |z| inside the pool; `--agg mean` for the mean) are those of rank_attack.py. The PRIMARY attack is fixed in advance:
L1+L2+L5 with the max aggregator; the others are sensitivity analyses (no feature is chosen with the labels).

    python scripts/analysis/rank_attack_v2.py --gpu 0 --out saves/analysis/rank_attack2_3B_f05 --base open-unlearning/tofu_Llama-3.2-3B-Instruct_full \\
        --forget forget05 --holdout holdout05 --extra-holdout holdout10 --templates retain --retain retain95 \\
        --model retain=hf:open-unlearning/tofu_Llama-3.2-3B-Instruct_retain95 --model full=hf:... --model NEAR-Act=near:... (as rank_attack.py)
    python scripts/analysis/rank_attack_v2.py --analyze saves/analysis/rank_attack2_3B_f05 [--k 2 --boot 1000]      # only the analysis of the stored answers (no model)
    --templates-only  extract the templates (CPU) -> <out>/templates.json

Output per model <out>/<label>.json (answers, features, results), <out>/summary.md.
"""
import argparse
import itertools
import json
import os
import random
import re
import statistics as st
import sys
from collections import Counter
from typing import Dict, List, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import rank_attack as ra  # noqa: E402
import sample_consistency as sc  # noqa: E402

L6 = ["greedy_logprob", "greedy_entropy", "logprob_sd"]
SETS = {
    "L1+L2+L5 (primary)": ra.L1 + ra.L2 + ra.L5,
    "L1+L2": ra.L1 + ra.L2,
    "L5": ra.L5,
    "L6 logits": L6,
    "L1+L2+L5+L6": ra.L1 + ra.L2 + ra.L5 + L6,
}
PRIMARY = "L1+L2+L5 (primary)"


# ----------------------------------------------------------------------------- templates from the retained data
def templates_from_qa(questions: Sequence[str], answers: Sequence[str], qa_per_author: int = 20, n: int = 10, min_count: int = 2):
    """The `n` most frequent question forms of the retained authors: [(template with {name}, count)]. The author name of a block of `qa_per_author` rows is replaced by {name}
    (the possessive `'s` is kept); a form must be used by at least `min_count` authors."""
    from METHOD.ROCR import rocr
    names = rocr.forget_author_names(questions, answers, qa_per_author)
    counts: Counter = Counter()
    for a, name in enumerate(names):
        seen = set()
        for q in questions[a * qa_per_author:(a + 1) * qa_per_author]:
            if name not in q or q.count(name) != 1:
                continue
            t = re.sub(r"\s+", " ", q.replace(name, "{name}")).strip()
            if "{" in t.replace("{name}", "") or "}" in t.replace("{name}", ""):
                continue
            seen.add(t)
        counts.update(seen)                      # a form counts once per author
    return [(t, c) for t, c in counts.most_common() if c >= min_count][:n]


def resolve_templates(spec: str, retain_split: str, qa_per_author: int, out_dir: str = None):
    """-> list of templates. spec: `builtin` (the hand-written ones of rank_attack.py), `retain` (derived), or a json file with a list."""
    if spec == "builtin":
        return list(ra.TEMPLATES), "builtin"
    if spec == "retain":
        qs, ans = sc.load_questions(retain_split)
        found = templates_from_qa(qs, ans, qa_per_author)
        ts = [t for t, _ in found]
        origin = f"retain({retain_split}): " + ", ".join(str(c) for _, c in found)
        for t in ra.TEMPLATES:                    # fill up to 10 with the built-in ones (reported)
            if len(ts) >= 10:
                break
            if t not in ts:
                ts.append(t)
                origin += " + builtin"
        if out_dir:
            with open(os.path.join(out_dir, "templates.json"), "w", encoding="utf-8") as f:
                json.dump({"templates": ts, "origin": origin, "derived": found}, f, ensure_ascii=False, indent=1)
        return ts, origin
    return json.load(open(spec, encoding="utf-8"))["templates"], spec


# ----------------------------------------------------------------------------- logit features
def greedy_with_scores(model, tokenizer, prompts, max_new_tokens=64, batch_size=8):
    """Greedy answers with the mean token log-probability and the mean entropy of the generation (an attacker who sees the logits). -> (texts, mean_logprob, mean_entropy)"""
    import torch
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts, lps, ents = [], [], []
    for i in range(0, len(prompts), batch_size):
        enc = tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, output_scores=True, return_dict_in_generate=True)
        gen = out.sequences[:, enc["input_ids"].shape[1]:]
        logp = torch.stack([torch.log_softmax(s.float(), dim=-1) for s in out.scores], dim=1)     # [B, T, V]
        tok_lp = logp.gather(-1, gen[:, :logp.shape[1]].unsqueeze(-1)).squeeze(-1)                # [B, T]
        ent = -(logp.exp() * logp).sum(-1)                                                        # [B, T]
        eos = tokenizer.eos_token_id
        valid = torch.ones_like(tok_lp, dtype=torch.bool)
        for b in range(gen.shape[0]):                                                             # keep the tokens up to and including the first eos
            hits = (gen[b, :logp.shape[1]] == eos).nonzero()
            if len(hits):
                valid[b, int(hits[0]) + 1:] = False
        denom = valid.sum(1).clamp(min=1)
        lps += ((tok_lp * valid).sum(1) / denom).tolist()
        ents += ((ent * valid).sum(1) / denom).tolist()
        texts += [t.strip() for t in tokenizer.batch_decode(gen, skip_special_tokens=True)]
    return texts, lps, ents


def logit_features(lps: Sequence[float], ents: Sequence[float]) -> Dict[str, float]:
    return {"greedy_logprob": sum(lps) / len(lps), "greedy_entropy": sum(ents) / len(ents), "logprob_sd": st.pstdev(lps) if len(lps) > 1 else 0.0}


# ----------------------------------------------------------------------------- score / evaluation
def scores_for(features, names, agg="max"):
    if agg == "max":
        return ra.anomaly_scores(features, names)
    z = {f: ra.robust_z([c[f] for c in features]) for f in names}
    return [sum(abs(z[f][i]) for f in names) / len(names) for i in range(len(features))]


def pool_auc(features, labels, names, agg="max"):
    return ra.ranking_metrics(scores_for(features, names, agg), labels)


def percentile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]


def mean_pair_stats(features, f_idx, h_idx, names, k, agg="max", rng=None, max_subsets=300):
    """The statistic of the test: mean over the k-subsets of the forgotten authors (each subset + all holdout authors = one pool) of AUC / precision@|F| / MRR."""
    subsets = list(itertools.combinations(f_idx, k))
    if len(subsets) > max_subsets:
        subsets = (rng or random.Random(0)).sample(subsets, max_subsets)
    ms = [pool_auc([features[i] for i in list(sub) + list(h_idx)], [1] * k + [0] * len(h_idx), names, agg) for sub in subsets]
    return {"auc": st.mean(m["auc"] for m in ms), "p_at_F": st.mean(m["p_at_F"] for m in ms), "mrr": st.mean(m["mrr"] for m in ms), "n_subsets": len(subsets)}


def evaluate(features, labels, k=2, boot=200, seed=0, agg="max", sets=None, max_subsets=300):
    """{set: {auc, auc_lo, auc_hi, null95, p_at_F, mrr}}.
    auc = the mean over the k-subsets of the forgotten authors of the pool AUC (pool = the subset + all holdout authors); [auc_lo, auc_hi] = a 95 % interval from the delete-one-AUTHOR jackknife
    (the authors, not the pools, are the independent units); null95 = the 95th percentile of the same statistic when the F / H labels are PERMUTED over the authors (`boot` permutations): the result
    is a detection only above null95."""
    sets = sets or SETS
    rng = random.Random(seed)
    f_idx = [i for i, y in enumerate(labels) if y]
    h_idx = [i for i, y in enumerate(labels) if not y]
    allidx = f_idx + h_idx
    out = {}
    for tag, names in sets.items():
        names = [n for n in names if all(n in c for c in features)]
        if not names:
            continue
        base = mean_pair_stats(features, f_idx, h_idx, names, k, agg, random.Random(seed), max_subsets)
        jk = []
        for drop in allidx:                                                   # delete-one-author jackknife
            fi = [i for i in f_idx if i != drop]
            hi = [i for i in h_idx if i != drop]
            if len(fi) >= k and hi:
                jk.append(mean_pair_stats(features, fi, hi, names, k, agg, random.Random(seed), max_subsets)["auc"])
        m = len(jk)
        se = (((m - 1) / m) * sum((x - st.mean(jk)) ** 2 for x in jk)) ** 0.5 if m > 1 else float("nan")
        nulls = []
        for _ in range(boot):
            perm = rng.sample(allidx, len(allidx))
            nulls.append(mean_pair_stats(features, perm[:len(f_idx)], perm[len(f_idx):], names, k, agg, random.Random(seed), max_subsets)["auc"])
        out[tag] = {**base, "auc_lo": max(0.0, base["auc"] - 1.96 * se), "auc_hi": min(1.0, base["auc"] + 1.96 * se), "null95": percentile(nulls, 0.95)}
    return out


def single_feature_aucs(features, labels, k=2, agg="max"):
    f_idx = [i for i, y in enumerate(labels) if y]
    h_idx = [i for i, y in enumerate(labels) if not y]
    out = {}
    for f in ra.L1 + ra.L2 + ra.L5 + L6:
        if not all(f in c for c in features):
            continue
        vals = [pool_auc([features[i] for i in list(sub) + h_idx], [1] * k + [0] * len(h_idx), [f], agg)["auc"] for sub in itertools.combinations(f_idx, k)]
        out[f] = st.mean(vals)
    return out


def summarize(rows, k, agg, header):
    tags = [t for t in SETS if any(t in r[1] for r in rows)]
    lines = [header, "", "mean pool AUC [95 % jackknife interval over the authors] (95th percentile of the label-permutation null) for the primary and the sensitivity attacks; a result counts as a detection only above the null band.", "",
             "| model | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
    for m, res, _ in rows:
        cells = []
        for t in tags:
            r = res.get(t)
            cells.append("-" if r is None else f"{r['auc']:.2f} [{r['auc_lo']:.2f}, {r['auc_hi']:.2f}] ({r['null95']:.2f})")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", f"precision@|F| / MRR of the primary attack (chance = {k}/pool):", "", "| model | primary |", "|---|---|"]
    for m, res, _ in rows:
        r = res.get(PRIMARY)
        lines.append(f"| {m} | " + ("-" if r is None else f"{r['p_at_F']:.2f} / {r['mrr']:.2f}") + " |")
    singles = sorted({f for _, _, s in rows for f in s})
    lines += ["", "AUC of a single feature (information only; choosing the best one with the labels is NOT an attack):", "", "| model | " + " | ".join(singles) + " |", "|---|" + "---|" * len(singles)]
    for m, _, s in rows:
        lines.append(f"| {m} | " + " | ".join("-" if f not in s else f"{s[f]:.2f}" for f in singles) + " |")
    return "\n".join(lines)


def analyze_dir(d, k=2, boot=200, agg="max", seed=0):
    order = ["retain", "full", "NEAR-Act", "NEAR-Dist", "LUNAR", "ROCR", "GA", "NPO", "RMU"]
    rows = []
    for m in order + sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json") and f[:-5] not in order and f != "templates.json"):
        p = os.path.join(d, f"{m}.json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p, encoding="utf-8"))
        res = evaluate(j["features"], j["labels"], k, boot, seed, agg)
        rows.append((m, res, single_feature_aucs(j["features"], j["labels"], k, agg)))
    n_f = sum(json.load(open(os.path.join(d, f"{rows[0][0]}.json")))["labels"])
    n_h = len(json.load(open(os.path.join(d, f"{rows[0][0]}.json")))["labels"]) - n_f
    md = summarize(rows, k, agg, f"{d}: pools of {k} forgotten + {n_h} holdout authors (k-subsets of {n_f}), aggregator {agg}, {boot} permutations")
    with open(os.path.join(d, f"summary_k{k}_{agg}.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    return md


# ----------------------------------------------------------------------------- model side
def query_model_v2(model, tok, candidates, templates, n, temperature, max_new_tokens, batch_size, seed):
    questions = [[t.format(name=c) for t in templates] for c in candidates]
    flat = [q for qs in questions for q in qs]
    prompts = sc.build_prompts(tok, flat)
    greedy, lps, ents = greedy_with_scores(model, tok, prompts, max_new_tokens, max(batch_size, 8))
    samples = sc.sample_answers(model, tok, prompts, n, temperature, 1.0, max_new_tokens, batch_size, seed)
    out, k = {}, len(templates)
    for i, c in enumerate(candidates):
        out[c] = {"questions": questions[i], "greedy": greedy[i * k:(i + 1) * k], "samples": samples[i * k:(i + 1) * k], "greedy_logprob": lps[i * k:(i + 1) * k],
                  "greedy_entropy": ents[i * k:(i + 1) * k]}
    return out


def featurize_v2(answers):
    base = ra.featurize({c: d for c, d in answers.items()})
    for f, d in zip(base, answers.values()):
        f.update(logit_features(d["greedy_logprob"], d["greedy_entropy"]))
    return base


def build_pool(a):
    f_names = ra.extract_names(a.forget, a.qa_per_author)
    h_names = ra.extract_names(a.holdout, a.qa_per_author)
    for extra in a.extra_holdout or []:
        for n in ra.extract_names(extra, a.qa_per_author):
            if n not in h_names and n not in f_names:
                h_names.append(n)
    return f_names, h_names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--base", default=None)
    ap.add_argument("--forget", default="forget05")
    ap.add_argument("--holdout", default="holdout05")
    ap.add_argument("--extra-holdout", nargs="*", default=[])
    ap.add_argument("--retain", default="retain95", help="split the `retain` templates are derived from")
    ap.add_argument("--templates", default="retain", help="builtin | retain | <json>")
    ap.add_argument("--qa-per-author", type=int, default=20)
    ap.add_argument("--names-json", default=None)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--boot", type=int, default=200, help="label permutations of the null")
    ap.add_argument("--agg", choices=["max", "mean"], default="max")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--analyze", default=None, help="only analyse the stored answers of this folder")
    ap.add_argument("--templates-only", action="store_true")
    a = ap.parse_args()
    if a.analyze:
        print(analyze_dir(a.analyze, a.k, a.boot, a.agg, a.seed))
        return
    if not a.out or not a.base:
        sys.exit("--out and --base are required")
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.makedirs(a.out, exist_ok=True)
    templates, origin = resolve_templates(a.templates, a.retain, a.qa_per_author, a.out)
    print("templates:", origin, *templates, sep="\n  ", flush=True)
    if a.templates_only:
        return
    if a.names_json:
        nm = json.load(open(a.names_json, encoding="utf-8"))
        f_names, h_names = nm["forget"], nm["holdout"]
    else:
        f_names, h_names = build_pool(a)
    for w in ra.check_pool(f_names, h_names):
        print("WARNING:", w, flush=True)
    candidates = f_names + h_names
    labels = [1] * len(f_names) + [0] * len(h_names)
    print(f"pool: {len(f_names)} forgotten + {len(h_names)} holdout authors\n  forgotten: {f_names}\n  holdout:   {h_names}", flush=True)
    if not a.model:
        sys.exit("--model is required (or --analyze / --templates-only)")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base)
    for spec in a.model:
        label, _, s = spec.partition("=")
        print(f"==== {label}: {s}", flush=True)
        model = sc.load_model(s, a.base)
        answers = query_model_v2(model, tok, candidates, templates, a.n, a.temperature, a.max_new_tokens, a.batch_size, a.seed)
        feats = featurize_v2(answers)
        with open(os.path.join(a.out, f"{label}.json"), "w", encoding="utf-8") as f:
            json.dump({"spec": s, "templates": templates, "candidates": candidates, "labels": labels, "features": feats, "answers": answers}, f, ensure_ascii=False, indent=1)
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    print(analyze_dir(a.out, a.k, a.boot, a.agg, a.seed))


if __name__ == "__main__":
    sys.exit(main())
