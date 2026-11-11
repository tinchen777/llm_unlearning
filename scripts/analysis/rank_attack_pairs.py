"""A1 with the forgotten authors as a MINORITY of the pool, recomputed from the features that rank_attack.py stored (no model, no GPU).

The pool of rank_attack.py is balanced (10 forgotten + 10 holdout): a median / MAD standardisation inside the pool then breaks down (half of the pool is shifted, so BOTH groups look
anomalous and the two-sided score has no direction). Here every subset of `k` forgotten authors (default 2: all 45 pairs of the 10) + all the holdout authors is one pool; the AUC /
precision@|F| / MRR are averaged over the subsets. The retain model (nothing to find) is the calibration and the full model the positive control.

    python scripts/analysis/rank_attack_pairs.py --dirs saves/analysis/rank_attack_1B_f05 saves/analysis/rank_attack_3B_f05 saves/analysis/rank_attack_8B_f05 [--k 2]

Caveat: the subsets share the same holdout authors and the same answers, so they are not independent (the spread over subsets is not a confidence interval); the models were queried once.
Writes <dir>/pairs_k<k>.md and prints it.
"""
import argparse
import itertools
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import rank_attack as ra  # noqa: E402

ORDER = ["retain", "full", "NEAR-Act", "NEAR-Dist", "LUNAR", "ROCR"]


def pair_metrics(features, labels, k=2, tags=None):
    """{tag: {auc, p_at_F, mrr, auc_sd}} averaged over every subset of k forgotten authors + all the holdout authors."""
    tags = tags or ra.FEATURE_SETS
    f_idx = [i for i, y in enumerate(labels) if y]
    h_idx = [i for i, y in enumerate(labels) if not y]
    acc = {t: [] for t in tags}
    for sub in itertools.combinations(f_idx, k):
        idx = list(sub) + h_idx
        ff = [features[i] for i in idx]
        ll = [1] * k + [0] * len(h_idx)
        for t, fs in tags.items():
            acc[t].append(ra.ranking_metrics(ra.anomaly_scores(ff, fs), ll))
    return {t: {"auc": st.mean(m["auc"] for m in ms), "auc_sd": st.pstdev([m["auc"] for m in ms]), "p_at_F": st.mean(m["p_at_F"] for m in ms),
                "mrr": st.mean(m["mrr"] for m in ms)} for t, ms in acc.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=2)
    a = ap.parse_args()
    for d in a.dirs:
        rows = []
        for m in ORDER:
            p = os.path.join(d, f"{m}.json")
            if os.path.exists(p):
                j = json.load(open(p, encoding="utf-8"))
                rows.append((m, pair_metrics(j["features"], j["labels"], a.k)))
        tags = list(ra.FEATURE_SETS)
        n_f = sum(j for j in json.load(open(os.path.join(d, f"{rows[0][0]}.json")))["labels"])
        lines = [f"{d}: every subset of {a.k} of the {n_f} forgotten authors + all holdout authors; mean AUC (sd over the subsets) / precision@|F| / MRR", "",
                 "| model | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
        for m, r in rows:
            lines.append(f"| {m} | " + " | ".join(f"{r[t]['auc']:.2f} ({r[t]['auc_sd']:.2f}) / {r[t]['p_at_F']:.2f} / {r[t]['mrr']:.2f}" for t in tags) + " |")
        md = "\n".join(lines)
        with open(os.path.join(d, f"pairs_k{a.k}.md"), "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print(md + "\n")


if __name__ == "__main__":
    main()
