"""Layer selection: where to write the neighbour redirection.

The edit is a single (or few) MLP down-projection, so the layer choice decides
whether the redirection is (a) meaningful, (b) cheap and (c) safe:

* **meaningful** - the layer must linearly encode "do I know this entity?"
  (`known_unknown_auc`): that is the axis the attacker reads and the axis we
  want to move the forgotten entity along;
* **cheap** - the forget activation must be close to its neighbour target
  (`redirect_cost`), otherwise the edit is large and the model degrades;
* **well-defined** - an entity's M neighbours must be mutually close
  (`neighbor_cohesion`), otherwise the mean target is an artefact;
* **safe** - forget and retain activations must be separable
  (`retain_separation`), so the edit does not drag the retained entities along.

`score_layers` normalises each diagnostic across the candidate layers and
combines them with configurable weights. Defaults reflect the priorities above;
the ranking, not the absolute score, is what matters.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from .metrics import layer_diagnostics

logger = logging.getLogger(__name__)

# component -> (diagnostic key, higher_is_better, weight)
DEFAULT_CRITERIA: Dict[str, Any] = {
    "known_unknown_auc": (True, 1.0),
    "redirect_cost": (False, 0.8),
    "neighbor_cohesion": (True, 0.6),
    "retain_separation": (True, 0.6),
}


@dataclass
class LayerScore:
    layer: int
    score: float
    components: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "score": self.score,
            "components": self.components,
            "diagnostics": self.diagnostics,
        }


def _normalize(values: Dict[int, float], higher_is_better: bool) -> Dict[int, float]:
    """Min-max normalise to `[0, 1]`, ignoring NaNs (which map to 0.5)."""
    finite = {k: v for k, v in values.items() if v is not None and math.isfinite(v)}
    if not finite:
        return {k: 0.5 for k in values}
    low, high = min(finite.values()), max(finite.values())
    span = high - low
    out = {}
    for key, value in values.items():
        if value is None or not math.isfinite(value):
            out[key] = 0.5
        elif span <= 0:
            out[key] = 0.5
        else:
            scaled = (value - low) / span
            out[key] = scaled if higher_is_better else 1.0 - scaled
    return out


def score_layers(
    diagnostics: Dict[int, Dict[str, float]],
    criteria: Optional[Dict[str, Any]] = None,
) -> List[LayerScore]:
    """Rank layers from their diagnostics (best first)."""
    criteria = criteria or DEFAULT_CRITERIA
    normalised: Dict[str, Dict[int, float]] = {}
    for key, (higher_is_better, _weight) in criteria.items():
        raw = {layer: diag.get(key, float("nan")) for layer, diag in diagnostics.items()}
        normalised[key] = _normalize(raw, higher_is_better)

    total_weight = sum(weight for _, weight in criteria.values()) or 1.0
    scores: List[LayerScore] = []
    for layer, diag in diagnostics.items():
        components = {key: normalised[key][layer] for key in criteria}
        score = sum(
            components[key] * weight for key, (_, weight) in criteria.items()
        ) / total_weight
        scores.append(
            LayerScore(layer=layer, score=float(score), components=components, diagnostics=diag)
        )
    return sorted(scores, key=lambda s: s.score, reverse=True)


def analyze_layers(
    model: Any,
    tokenizer: Any,
    forget_questions: Sequence[str],
    neighbor_questions: Sequence[Sequence[str]],
    retain_questions: Sequence[str] = (),
    layers: Optional[Sequence[int]] = None,
    strategy: str = "last_prompt",
    batch_size: int = 8,
    max_length: int = 512,
    apply_chat_template: bool = True,
    system_prompt: Optional[str] = None,
    target_mode: str = "mean",
    criteria: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> List[LayerScore]:
    """Extract activations once and score every candidate layer.

    Args:
        forget_questions: `N` forget questions.
        neighbor_questions: `N` lists of M neighbour questions (aligned).
        retain_questions: questions about retained (still known) entities.
        layers: candidate layers; defaults to every layer of the model.
        target_mode: `mean` averages the M neighbours into one target,
            `first` uses the first neighbour (cheaper, matches sampled training).
    """
    from .activations import extract_activations, num_layers

    if len(forget_questions) != len(neighbor_questions):
        raise ValueError(
            f"{len(forget_questions)} forget questions but "
            f"{len(neighbor_questions)} neighbour lists."
        )
    layers = list(layers) if layers is not None else list(range(num_layers(model)))

    flat_neighbors: List[str] = []
    offsets: List[int] = []
    for group in neighbor_questions:
        offsets.append(len(flat_neighbors))
        flat_neighbors.extend(group if target_mode == "mean" else group[:1])
    offsets.append(len(flat_neighbors))

    extract = dict(
        layers=layers, strategy=strategy, batch_size=batch_size, max_length=max_length,
        apply_chat_template=apply_chat_template, system_prompt=system_prompt,
    )
    logger.info("Extracting activations for %d layers ...", len(layers))
    forget_acts = extract_activations(model, tokenizer, forget_questions, **extract)
    neighbor_acts = extract_activations(model, tokenizer, flat_neighbors, **extract)
    retain_acts = (
        extract_activations(model, tokenizer, list(retain_questions), **extract)
        if len(retain_questions) else None
    )

    diagnostics: Dict[int, Dict[str, float]] = {}
    for layer in layers:
        groups = [
            neighbor_acts[layer][offsets[i] : offsets[i + 1]]
            for i in range(len(neighbor_questions))
        ]
        paired = torch.stack([g.mean(0) for g in groups])  # [N, d] target per question
        diagnostics[layer] = layer_diagnostics(
            forget=forget_acts[layer],
            neighbor=paired,
            retain=retain_acts[layer] if retain_acts else None,
            neighbor_groups=groups,
            seed=seed,
        )
        logger.info("layer %3d: %s", layer, _format_diag(diagnostics[layer]))

    return score_layers(diagnostics, criteria=criteria)


def _format_diag(diag: Dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.3f}" for k, v in diag.items() if isinstance(v, float))


def save_selection(
    scores: Sequence[LayerScore],
    path: Union[str, Path],
    meta: Optional[Dict[str, Any]] = None,
    top_k: int = 3,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta or {},
        "selected_layers": [s.layer for s in scores[:top_k]],
        "ranking": [s.to_dict() for s in scores],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def load_selection(path: Union[str, Path]) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def format_table(scores: Sequence[LayerScore], top_k: Optional[int] = None) -> str:
    """Human-readable ranking table for the log / the appendix."""
    rows = list(scores)[: top_k or len(scores)]
    if not rows:
        return "(no layers scored)"
    keys = ["known_unknown_auc", "redirect_cost", "neighbor_cohesion", "retain_separation"]
    header = f"{'rank':>4} {'layer':>5} {'score':>7}  " + "  ".join(f"{k:>18}" for k in keys)
    lines = [header, "-" * len(header)]
    for rank, row in enumerate(rows, start=1):
        cells = "  ".join(f"{row.diagnostics.get(k, float('nan')):>18.4f}" for k in keys)
        lines.append(f"{rank:>4} {row.layer:>5} {row.score:>7.4f}  {cells}")
    return "\n".join(lines)
