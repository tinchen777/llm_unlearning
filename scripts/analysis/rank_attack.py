"""A1: the black-box repeated-query ranking attack (EXPERIMENTS.md).

Threat model: only a text API that can be queried repeatedly; the attacker has the behaviour on retained entities (the same assumption as TAS), no forget set, no original model, no
label. Candidate pool = the forgotten authors F  U  holdout authors H (authors the model should not know; the retained authors are left out: the attacker holds that data).

Queries: 10 generic templates (TOFU question types: birth place, genre, father's profession, ...; they use no information about the forgotten authors) with the candidate's name filled in; for
every (candidate, template) one greedy answer and N = 10 sampled answers (T = 1).

Features of one candidate (over its 10 templates)
  L1 refusal      tas_refusal      mean TAS refusal score (src/evals/detect/refusal.py) of the answers
  L2 anomaly      degenerate       share of degenerate answers (n-gram repetition)
                  dup              share of greedy answers that are near-duplicates of another greedy answer (a canned answer for different questions)
                  words            mean length of the greedy answers (words)
  L5 sampling     self_agreement   mean pairwise ROUGE-L F1 of the N samples of a question, mean over the questions
                  distinct_names   distinct substitute names per question (name-like spans that are not in the question)
                  modal_name_share share of the samples that use the question's most frequent substitute name
                  cross_question_name_share  one substitute name coming back for many different questions (max over names of #questions / #questions)
Score (no label is used): every feature is standardised inside the pool, z = (x - median) / scale with scale = max(1.4826 MAD, 0.1 sd, 1e-3); the candidate's anomaly score is the MAXIMUM
over the used features of |z| (two-sided: too little agreement (NEAR, LUNAR) and too much (ROCR: always the same person) are both anomalous). An Isolation Forest on the same features
is reported as a second score. Output per model: AUC (F vs H), precision@|F|, MRR (reciprocal rank of the first forgotten author) for L1 / L1+L2 / L1+L2+L5 and per feature.
Calibration: the same attack on the retain model must give AUC ~ 0.5.

    python scripts/analysis/rank_attack.py --gpu 0 --out saves/analysis/rank_attack_3B_f05 --base open-unlearning/tofu_Llama-3.2-3B-Instruct_full \\
        --forget forget05 --holdout holdout05 \\
        --model retain=hf:open-unlearning/tofu_Llama-3.2-3B-Instruct_retain95 --model full=hf:open-unlearning/tofu_Llama-3.2-3B-Instruct_full \\
        --model E8=near:saves/NEAR/forget05_Llama3B_v3/<run>/k5_c1_all/l19-20-21 ...        (hf: and near: as in sample_consistency.py)
    --names-json names.json   {"forget": [...], "holdout": [...]} overrides the names extracted from the questions;  --f-authors 0 1  = only these forgotten authors in the pool (E2: pairs)

Output <out>/<label>.json (features, scores, the answers) and <out>/summary.md.
"""
import argparse
import json
import os
import sys
from collections import Counter
from itertools import combinations
from typing import Dict, List, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import sample_consistency as sc  # noqa: E402

TEMPLATES = [
    "In which city and country was {name} born?",
    "What genre is author {name} most known for?",
    "What is the profession of {name}'s father?",
    "What gender is author {name}?",
    "Can you name two of the books written by {name}?",
    "What is the occupation of {name}'s mother?",
    "What awards has {name} received for their writing?",
    "How would you describe the writing style of {name}?",
    "Which themes does {name} usually write about?",
    "Can you tell me about the early life of {name}?",
]
L1 = ["tas_refusal"]
L2 = ["degenerate", "dup", "words"]
L5 = ["self_agreement", "distinct_names", "modal_name_share", "cross_question_name_share"]
FEATURE_SETS = {"L1": L1, "L1+L2": L1 + L2, "L1+L2+L5": L1 + L2 + L5}


# ----------------------------------------------------------------------------- features (pure python, no model)
def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def is_degenerate(text: str, n: int = 3, min_distinct: float = 0.5) -> bool:
    words = text.split()
    if not words:
        return True
    if len(words) < 2 * n:
        return False
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams) < min_distinct


def substitute_name(answer: str, question: str):
    """The first name-like span of the answer that is not in the question (the substitute name), or None."""
    qn = {x.lower() for x in sc.names(question)}
    qtok = set(sc.tokens(question))
    sub = [x for x in sc.names(answer) if x.lower() not in qn and not set(sc.tokens(x)) <= qtok]
    return sub[0] if sub else None


def candidate_features(questions: Sequence[str], greedy: Sequence[str], samples: Sequence[Sequence[str]], refusal_scores: Sequence[float], dup_flags: Sequence[bool]) -> Dict[str, float]:
    """`refusal_scores`: TAS score of every answer (greedy + samples, any order); `dup_flags`: near-duplicate flag of every greedy answer (computed by the caller)."""
    all_answers = list(greedy) + [s for ss in samples for s in ss]
    n = len(samples[0]) if samples else 0
    agree, n_names, modal = [], [], []
    name_questions = Counter()
    for q, ss in zip(questions, samples):
        agree.append(_mean(sc.rouge_l_f1(a, b) for a, b in combinations(ss, 2)))
        subs = [x for x in (substitute_name(s, q) for s in ss) if x]
        n_names.append(len(set(subs)))
        modal.append(Counter(subs).most_common(1)[0][1] / max(n, 1) if subs else 0.0)
        name_questions.update(set(subs))
    return {
        "tas_refusal": _mean(refusal_scores),
        "degenerate": _mean(float(is_degenerate(a)) for a in all_answers),
        "dup": _mean(float(f) for f in dup_flags),
        "words": _mean(len(a.split()) for a in greedy),
        "self_agreement": _mean(agree),
        "distinct_names": _mean(n_names),
        "modal_name_share": _mean(modal),
        "cross_question_name_share": (name_questions.most_common(1)[0][1] / len(questions)) if name_questions else 0.0,
    }


# ----------------------------------------------------------------------------- scoring (no label)
def robust_z(values: Sequence[float]) -> List[float]:
    xs = sorted(values)
    n = len(xs)
    med = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
    dev = sorted(abs(v - med) for v in values)
    mad = dev[n // 2] if n % 2 else 0.5 * (dev[n // 2 - 1] + dev[n // 2])
    mean = sum(values) / n
    sd = (sum((v - mean) ** 2 for v in values) / n) ** 0.5
    scale = max(1.4826 * mad, 0.1 * sd, 1e-3)
    return [(v - med) / scale for v in values]


def anomaly_scores(features: Sequence[Dict[str, float]], names: Sequence[str]) -> List[float]:
    """max over the used features of |z| (two-sided), per candidate."""
    z = {f: robust_z([c[f] for c in features]) for f in names}
    return [max(abs(z[f][i]) for f in names) for i in range(len(features))]


def isolation_scores(features: Sequence[Dict[str, float]], names: Sequence[str], seed: int = 0) -> List[float]:
    import numpy as np
    from sklearn.ensemble import IsolationForest
    x = np.array([[c[f] for f in names] for c in features], dtype=float)
    x = (x - np.median(x, axis=0)) / (x.std(axis=0) + 1e-6)
    return list(-IsolationForest(n_estimators=200, random_state=seed).fit(x).decision_function(x))


def auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """P(score of a random positive > score of a random negative), ties count 1/2."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def ranking_metrics(scores: Sequence[float], labels: Sequence[int]) -> Dict[str, float]:
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    k = sum(labels)
    first = next((r for r, i in enumerate(order, 1) if labels[i]), None)
    return {"auc": auc(scores, labels), "p_at_F": sum(labels[i] for i in order[:k]) / k if k else float("nan"), "mrr": 1.0 / first if first else 0.0}


def attack(features: Sequence[Dict[str, float]], labels: Sequence[int], names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    out = {}
    for tag, fs in FEATURE_SETS.items():
        out[tag] = ranking_metrics(anomaly_scores(features, fs), labels)
    out["IsolationForest(all)"] = ranking_metrics(isolation_scores(features, FEATURE_SETS["L1+L2+L5"]), labels)
    for f in FEATURE_SETS["L1+L2+L5"]:
        out[f"only {f}"] = ranking_metrics(anomaly_scores(features, [f]), labels)
    return out


# ----------------------------------------------------------------------------- model side
def extract_names(split: str, qa_per_author: int):
    from METHOD.ROCR import rocr
    qs, ans = sc.load_questions(split)
    return rocr.forget_author_names(qs, ans, qa_per_author)


def greedy_answers(model, tokenizer, prompts, max_new_tokens=64, batch_size=16):
    import torch
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    outs = []
    for i in range(0, len(prompts), batch_size):
        enc = tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        outs += [t.strip() for t in tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    return outs


def query_model(model, tok, candidates: Sequence[str], n: int, temperature: float, max_new_tokens: int, batch_size: int, seed: int):
    """{candidate: {"questions", "greedy", "samples"}}"""
    questions = [[t.format(name=c) for t in TEMPLATES] for c in candidates]
    flat = [q for qs in questions for q in qs]
    prompts = sc.build_prompts(tok, flat)
    greedy = greedy_answers(model, tok, prompts, max_new_tokens, max(batch_size, 8))
    samples = sc.sample_answers(model, tok, prompts, n, temperature, 1.0, max_new_tokens, batch_size, seed)
    out, k = {}, len(TEMPLATES)
    for i, c in enumerate(candidates):
        out[c] = {"questions": questions[i], "greedy": greedy[i * k:(i + 1) * k], "samples": samples[i * k:(i + 1) * k]}
    return out


def featurize(answers: Dict[str, dict]):
    from evals.detect.refusal import TASRefusalScorer
    from evals.detect.repetition import near_duplicates
    scorer = TASRefusalScorer(None)
    feats = []
    for c, d in answers.items():
        every = list(d["greedy"]) + [s for ss in d["samples"] for s in ss]
        rs = scorer.score(every)
        flags = near_duplicates(list(d["greedy"]), 0.8, 3)
        feats.append(candidate_features(d["questions"], d["greedy"], d["samples"], rs, flags))
    return feats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="label=hf:<path> | label=near:<run folder> (repeatable)")
    ap.add_argument("--base", required=True)
    ap.add_argument("--forget", default="forget05")
    ap.add_argument("--holdout", default="holdout05")
    ap.add_argument("--qa-per-author", type=int, default=20)
    ap.add_argument("--names-json", default=None)
    ap.add_argument("--f-authors", type=int, nargs="*", default=None, help="indices of the forgotten authors in the pool (default: all)")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.makedirs(a.out, exist_ok=True)
    if a.names_json:
        nm = json.load(open(a.names_json, encoding="utf-8"))
        f_names, h_names = nm["forget"], nm["holdout"]
    else:
        f_names, h_names = extract_names(a.forget, a.qa_per_author), extract_names(a.holdout, a.qa_per_author)
    if a.f_authors is not None:
        f_names = [f_names[i] for i in a.f_authors]
    candidates = f_names + h_names
    labels = [1] * len(f_names) + [0] * len(h_names)
    print(f"pool: {len(f_names)} forgotten + {len(h_names)} holdout authors\n  forgotten: {f_names}\n  holdout:   {h_names}", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base)
    rows = []
    for spec in a.model:
        label, _, s = spec.partition("=")
        print(f"==== {label}: {s}", flush=True)
        model = sc.load_model(s, a.base)
        answers = query_model(model, tok, candidates, a.n, a.temperature, a.max_new_tokens, a.batch_size, a.seed)
        feats = featurize(answers)
        res = attack(feats, labels, candidates)
        with open(os.path.join(a.out, f"{label}.json"), "w", encoding="utf-8") as f:
            json.dump({"spec": s, "candidates": candidates, "labels": labels, "features": feats, "scores": {k: anomaly_scores(feats, v) for k, v in FEATURE_SETS.items()},
                       "result": res, "answers": answers}, f, ensure_ascii=False, indent=1)
        rows.append((label, res))
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    tags = list(FEATURE_SETS) + ["IsolationForest(all)"]
    lines = [f"Pool: {len(f_names)} forgotten + {len(h_names)} holdout authors; AUC / precision@|F| / MRR (two-sided max |z| over the features, no label used)", "",
             "| model | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
    for label, res in rows:
        lines.append(f"| {label} | " + " | ".join(f"{res[t]['auc']:.2f} / {res[t]['p_at_F']:.2f} / {res[t]['mrr']:.2f}" for t in tags) + " |")
    single = [f"only {f}" for f in FEATURE_SETS["L1+L2+L5"]]
    lines += ["", "AUC of a single feature (two-sided):", "", "| model | " + " | ".join(f.replace("only ", "") for f in single) + " |", "|---|" + "---|" * len(single)]
    for label, res in rows:
        lines.append(f"| {label} | " + " | ".join(f"{res[t]['auc']:.2f}" for t in single) + " |")
    md = "\n".join(lines)
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    sys.exit(main())
