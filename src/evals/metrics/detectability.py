from __future__ import annotations
import logging
from typing import Any, Dict, Mapping, Optional, Sequence

from .base import MetricFunc
from ..detect.scores import unlearning_detectability as _detectability

logger = logging.getLogger("eval.metric")


def _records(result: Optional[Mapping[str, Any]]):
    """The records (one per answer) of the result of a `responses` metric."""
    if not result:
        return []
    return [result["value_by_index"][k] for k in sorted(result["value_by_index"], key=lambda x: int(x))]


@MetricFunc
def unlearning_detectability(
    pre_compute: Dict[str, Any],
    reference_logs: Optional[Dict[str, Any]] = None,
    aggregate: str = "bb",
    dup_threshold: float = 0.8,
    tas_threshold: float = 0.5,
    qa_per_author: int = 20,
    **kwargs
):
    """How well can an observer tell what was unlearned (`evals.detect.scores`)? `agg_value` = UD in [0, 1] (0 = not at all).

    `pre_compute` (access keys): `forget` and any of `retain` / `holdout` / `neighbor` = results of `responses` metrics (the answers to the
    forgotten prompts / to the others), `activation` = the result of `activation_probe` (white box; optional).
    `reference_logs.retain_model_probe.activation_probe`: the `value_by_layer` of the same probe on a clean reference model (the retain
    model), to report the white-box numbers as the excess over it.
    `summary`: bb_tas, bb, entity_bb_tas, entity_bb, wb_sup, wb_unsup (and their layers / raw values), the rates they are made of, ud, ud_max.
    """
    forget = _records(pre_compute.get("forget"))
    if not forget:
        raise KeyError("unlearning_detectability needs the responses of the forgotten prompts: pre_compute `forget`.")
    benign = {s: _records(pre_compute[s]) for s in ("retain", "holdout", "neighbor") if s in pre_compute}
    if not any(benign.values()):
        raise KeyError("unlearning_detectability needs the responses to other prompts: pre_compute `retain` / `holdout` / `neighbor`.")
    activation = (pre_compute.get("activation") or {}).get("value_by_layer")
    reference = ((reference_logs or {}).get("retain_model_probe") or {}).get("activation_probe")
    reference_activation = (reference or {}).get("value_by_layer") if isinstance(reference, Mapping) else None
    if activation is not None and reference_activation is None:
        logger.warning("No reference probe (`reference_logs.retain_model_probe`): the white-box numbers are not corrected for the "
                       "separability of the question sets that any model has (the raw advantage, an overestimate).")
    summary = _detectability(
        forget, benign, activation, reference_activation, aggregate=aggregate,
        dup_threshold=dup_threshold, tas_threshold=tas_threshold, qa_per_author=qa_per_author,
    )
    return {"agg_value": summary["ud"], "summary": summary}
