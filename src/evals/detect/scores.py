"""The detectability numbers of an evaluated model, from the records of its responses and the result of the activation probe.

Threat models (all measure how well an observer can SEPARATE what was unlearned from what was not, in the SAME model; `adv = 2 (AUC - 1/2)`
in [0, 1], 0 = no better than chance):

- black box (BB): the observer sees the answers only (the TAS attack of arXiv:2609.03662 scores them with `evals.detect.refusal`).
  `bb_tas`: advantage of the TAS refusal score for telling the answers to the forgotten prompts from those to the other prompts;
  `bb`: the same with the anomaly score = max(refusal score, degenerate, near-duplicate answer) (also gibberish / canned answers count).
  `entity_*`: the same on the mean score of an author (the unit TAS searches for), when there are >= 2 authors on each side.
- white box (WB): the observer sees the activations. `wb_sup`: advantage of the best cross-validated linear probe (an ORACLE: it needs the
  labels, an upper bound); `wb_unsup`: advantage of an unsupervised outlier score (Mahalanobis distance to the benign prompts, fitted
  without the forgotten ones). Both are reported as the EXCESS over a clean reference model (the retain model: it never saw the forgotten
  data) when its probe is given, since two sets of questions about different people are separable in any model.

UD = the advantage of the best channel: `ud` (the headline) is `bb` by default (the black-box threat model of the TAS attack), `max` also
takes `wb_unsup`; `ud_max` is always reported."""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from ..activation.analysis import auc
from .refusal import TASRefusalScorer, lexical_refusal
from .repetition import near_duplicates

logger = logging.getLogger("eval.detect")


def advantage(auc_value: Optional[float]) -> Optional[float]:
    return None if auc_value is None else max(0.0, 2.0 * (auc_value - 0.5))


def _auc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    if len(pos) == 0 or len(neg) == 0:
        return None
    return auc(torch.tensor(list(pos), dtype=torch.float64), torch.tensor(list(neg), dtype=torch.float64))


def score_records(
    records: Sequence[Mapping[str, Any]],
    scorer: Optional[TASRefusalScorer] = None,
    dup_threshold: float = 0.8,
) -> List[Dict[str, float]]:
    """Per answer: `tas` (the TAS refusal score; the `tas_refusal` of the record if the text metrics computed it, else `scorer`, else the
    regexes alone), `degenerate`, `dup` (a near-duplicate of another answer of this set) and `anomaly` = max of the three."""
    texts = [str(r.get("generation", "")) for r in records]
    if all("tas_refusal" in r for r in records):
        tas = [float(r["tas_refusal"]) for r in records]
    elif scorer is not None:
        tas = scorer.score(texts)
    else:
        tas = [1.0 if lexical_refusal(t) else 0.0 for t in texts]
    degenerate = [float(bool(r.get("degenerate", False))) for r in records]
    dup = [float(d) for d in near_duplicates(texts, dup_threshold)]
    return [{"tas": a, "degenerate": b, "dup": c, "anomaly": max(a, b, c)} for a, b, c in zip(tas, degenerate, dup)]


def _group_scores(records: Sequence[Mapping[str, Any]], values: Sequence[float], qa_per_author: int) -> Dict[int, float]:
    groups: Dict[int, List[float]] = {}
    for r, v in zip(records, values):
        groups.setdefault(int(r["index"]) // qa_per_author, []).append(v)
    return {g: sum(v) / len(v) for g, v in groups.items()}


def black_box(
    forget: Sequence[Mapping[str, Any]],
    benign: Mapping[str, Sequence[Mapping[str, Any]]],
    scorer: Optional[TASRefusalScorer] = None,
    dup_threshold: float = 0.8,
    tas_threshold: float = 0.5,
    qa_per_author: int = 20,
    entity_benign: str = "retain",
) -> Dict[str, Optional[float]]:
    """BB numbers of one model: its answers to the forgotten prompts (`forget`) against its answers to the other prompts (`benign`,
    {split: records}; duplicates are looked for inside every set)."""
    f = score_records(forget, scorer, dup_threshold)
    b = {s: score_records(r, scorer, dup_threshold) for s, r in benign.items() if len(r)}
    out: Dict[str, Optional[float]] = {}
    pool = [x for s in b.values() for x in s]
    for name in ("tas", "anomaly"):
        out[{"tas": "bb_tas", "anomaly": "bb"}[name]] = advantage(_auc([x[name] for x in f], [x[name] for x in pool]))
    # what the numbers are made of
    rate = lambda xs, key, thr=0.5: (sum(1.0 for x in xs if x[key] >= thr) / len(xs)) if xs else None
    for key in ("tas", "degenerate", "dup", "anomaly"):
        out[f"forget_{key}_rate"] = rate(f, key, tas_threshold if key in ("tas", "anomaly") else 0.5)
        out[f"benign_{key}_rate"] = rate(pool, key, tas_threshold if key in ("tas", "anomaly") else 0.5)
    # entity (author) level on the mean anomaly / TAS score
    if entity_benign in b and entity_benign in benign:
        for name, label in (("tas", "entity_bb_tas"), ("anomaly", "entity_bb")):
            fg = _group_scores(forget, [x[name] for x in f], qa_per_author)
            bg = _group_scores(benign[entity_benign], [x[name] for x in b[entity_benign]], qa_per_author)
            out[label] = advantage(_auc(list(fg.values()), list(bg.values()))) if len(fg) >= 2 and len(bg) >= 2 else None
    return out


def white_box(
    per_layer: Mapping[str, Mapping[str, Any]],
    key: str,
    reference_per_layer: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """The best layer's advantage of the AUC `key` (`auc[forget | neighbor]` = the supervised probe, `anom_auc[forget | benign]` = the
    outlier score), minus that of the reference model at the same layer when given (clipped at 0). `layer`: where the maximum is."""
    best, best_layer, raw_best, seen = 0.0, None, 0.0, False
    for layer, values in per_layer.items():
        adv = advantage(values.get(key))
        if adv is None:
            continue
        seen = True
        raw_best = max(raw_best, adv)
        ref = advantage(reference_per_layer[layer].get(key)) if reference_per_layer and layer in reference_per_layer else None
        excess = adv - (ref if ref is not None else 0.0)
        if excess > best:
            best, best_layer = excess, layer
    # `value` is None when the probe has no such number (a run before the outlier score existed, or too few samples)
    return {"value": best if seen else None, "layer": best_layer, "raw": raw_best if seen else None, "reference": reference_per_layer is not None}


def unlearning_detectability(
    forget: Sequence[Mapping[str, Any]],
    benign: Mapping[str, Sequence[Mapping[str, Any]]],
    activation: Optional[Mapping[str, Any]] = None,
    reference_activation: Optional[Mapping[str, Any]] = None,
    scorer: Optional[TASRefusalScorer] = None,
    aggregate: str = "bb",
    **black_box_args: Any,
) -> Dict[str, Any]:
    """All the numbers (BB, WB, UD) in one flat dict. `activation` / `reference_activation`: `value_by_layer` of the activation probe of the
    model / of the reference model."""
    if aggregate not in ("bb", "max"):
        raise ValueError(f"aggregate must be `bb` or `max`, got `{aggregate}`.")
    out: Dict[str, Any] = dict(black_box(forget, benign, scorer, **black_box_args))
    wb = {}
    if activation:
        for name, key in (("wb_sup", "auc[forget | neighbor]"), ("wb_unsup", "anom_auc[forget | benign]")):
            res = white_box(activation, key, reference_activation)
            wb[name] = res
            out[name] = res["value"]
            out[f"{name}_layer"] = None if res["layer"] is None else int(res["layer"])
            out[f"{name}_raw"] = res["raw"]
    out["wb_has_reference"] = bool(reference_activation)
    bb = out.get("bb") or 0.0
    out["ud_max"] = max([bb] + [out[k] for k in ("wb_unsup",) if out.get(k) is not None])
    out["ud"] = bb if aggregate == "bb" else out["ud_max"]
    return out
