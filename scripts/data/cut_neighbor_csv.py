"""Cut the neighbour csv of a smaller forget split out of the full `tofu_neighbor10_with_source.csv` (authors 181-200) and check it.

    python scripts/data/cut_neighbor_csv.py /path/to/tofu_neighbor10_with_source.csv --out data/tofu
        -> data/tofu/tofu_neighbor10.csv (all), tofu_neighbor05.csv (authors 191-200), tofu_neighbor01.csv (authors 199-200)

The row order is kept. Checks: every (author_id, qa_id) has exactly 10 `train` neighbours, one `source_question` (not empty), and (if given with --compare) the QA columns
(question, answer, split, train_rank) of an existing csv are identical to the cut ones. NEAR needs `author_id`, `qa_id`, `source_question`, `split`, `train_rank`, `question`, `answer`.
"""
import argparse
import os
import sys

import pandas as pd

SPLITS = {"10": range(181, 201), "05": range(191, 201), "01": range(199, 201)}
QA_COLUMNS = ["question", "answer", "split", "train_rank"]


def check(df: pd.DataFrame, authors) -> None:
    if set(df.author_id) != set(authors):
        raise SystemExit(f"authors {sorted(set(df.author_id))} != {list(authors)}")
    groups = df.groupby(["author_id", "qa_id"])
    if len(groups) != 20 * len(list(authors)):
        raise SystemExit(f"{len(groups)} (author, qa_id) groups, expected {20 * len(list(authors))}")
    if (groups.source_question.nunique() != 1).any() or df.source_question.isna().any() or (df.source_question.str.strip() == "").any():
        raise SystemExit("source_question must be present and unique per (author_id, qa_id)")
    train = df[df.split == "train"].groupby(["author_id", "qa_id"]).size()
    if len(train) != len(groups) or train.min() != 10 or train.max() != 10:
        raise SystemExit("every (author_id, qa_id) needs exactly 10 `train` neighbours")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("full_csv")
    ap.add_argument("--out", default="data/tofu")
    ap.add_argument("--compare", default=None, help="folder with existing tofu_neighbor{01,05,10}.csv to compare the QA columns with")
    args = ap.parse_args()
    full = pd.read_csv(args.full_csv)
    os.makedirs(args.out, exist_ok=True)
    for name, authors in SPLITS.items():
        part = full[full.author_id.isin(list(authors))]
        check(part, authors)
        path = os.path.join(args.out, f"tofu_neighbor{name}.csv")
        part.to_csv(path, index=False)
        msg = f"{path}: {len(part)} rows"
        if args.compare and os.path.exists(os.path.join(args.compare, f"tofu_neighbor{name}.csv")):
            old = pd.read_csv(os.path.join(args.compare, f"tofu_neighbor{name}.csv"))
            key = ["author_id", "neighbor_id", "qa_id"]
            m = old.merge(part, on=key, suffixes=("_o", "_n"))
            same = len(m) == len(old) == len(part) and all((m[c + "_o"].fillna("").astype(str) == m[c + "_n"].fillna("").astype(str)).all() for c in QA_COLUMNS)
            msg += f" | QA columns identical to {args.compare}: {same}"
            if not same:
                sys.exit(msg)
        print(msg)


if __name__ == "__main__":
    main()
