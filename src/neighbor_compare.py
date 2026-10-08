"""Compare the `activations_summary.json` of two probes of the SAME splits (src/neighbor.py), no model needed.

    python src/neighbor_compare.py --full <full>/activations_summary.json --retain <retain>/activations_summary.json

Why: the AUC of a pair of question sets mostly reflects how different the QUESTIONS are, which is identical for the
two models. The difference `full - retain` of the same pair cancels that content effect; what remains is the part of
the separation that exists only because the full model has SEEN the forget data (the retain model has not).
The pair to read is the matched one: `forget | neighbor` (neighbours are rewrites of the forget questions); layers
with a large positive difference are where "seen vs unseen" is linearly visible, a candidate window for the layer
selection (src/select_layer.py decides, this only narrows it down).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_layers(path: str) -> Dict[int, Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f)["layers"].items()}


def compare(full: Dict[int, Dict[str, Any]], retain: Dict[int, Dict[str, Any]], pair: str) -> List[Dict[str, Any]]:
    """One row per common layer: AUC of `pair` in both models, their difference (sample- and group-wise), and the
    relative size of the redirection `r_uv_rel_norm` (if the summaries have it)."""
    rows = []
    for layer in sorted(set(full) & set(retain)):
        f, r = full[layer], retain[layer]
        row: Dict[str, Any] = {"layer": layer}
        for kind in ("auc", "auc_group"):
            key = f"{kind}[{pair}]"
            if key not in f or key not in r:
                raise KeyError(f"`{key}` is not in both summaries; available: {sorted(k for k in f if k.startswith(kind))}")
            a, b = f[key], r[key]
            row[f"{kind}_full"], row[f"{kind}_retain"] = a, b
            row[f"{kind}_delta"] = None if a is None or b is None else a - b
        row["r_uv_rel_norm_full"] = f.get("r_uv_rel_norm")
        row["r_uv_rel_norm_retain"] = r.get("r_uv_rel_norm")
        rows.append(row)
    return rows


def format_table(rows: List[Dict[str, Any]], pair: str, top: int = 5) -> str:
    fmt = lambda v: "   -  " if v is None else f"{v:6.3f}"
    lines = [f"pair: {pair}", "layer | auc full  retain  delta | auc_group full  retain  delta | rel r_UV (full)"]
    for r in rows:
        lines.append(
            f"{r['layer']:>5} |     {fmt(r['auc_full'])} {fmt(r['auc_retain'])} {fmt(r['auc_delta'])} |"
            f"           {fmt(r['auc_group_full'])} {fmt(r['auc_group_retain'])} {fmt(r['auc_group_delta'])} |"
            f" {fmt(r['r_uv_rel_norm_full'])}"
        )
    for kind in ("auc_group", "auc"):
        ranked = sorted((r for r in rows if r[f"{kind}_delta"] is not None), key=lambda r: -r[f"{kind}_delta"])[:top]
        lines.append(f"top {len(ranked)} layers by {kind}_delta: " + ", ".join(f"{r['layer']} ({r[f'{kind}_delta']:+.3f})" for r in ranked))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", required=True, help="activations_summary.json of the probe on the TOFU-full model")
    ap.add_argument("--retain", required=True, help="activations_summary.json of the probe on the TOFU-retain model")
    ap.add_argument("--pair", default="forget | neighbor", help="pair of splits, as written in the summaries")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", default=None, help="write the rows as json")
    args = ap.parse_args(argv)
    rows = compare(load_layers(args.full), load_layers(args.retain), args.pair)
    print(format_table(rows, args.pair, args.top))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"pair": args.pair, "full": args.full, "retain": args.retain, "rows": rows}, f, indent=2)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
