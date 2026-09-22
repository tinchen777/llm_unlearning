"""End-to-end construction of the neighbour bank.

    rows -> entities -> attributes -> candidates -> verification -> questions

Every stage is replaceable (see `attributes.py`, `generate.py`, `verify.py`);
this module only wires them together and records provenance in `bank.meta`.
"""

from __future__ import annotations
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Sequence

from .attributes import (
    HeuristicAttributeExtractor,
    extract_attributes,
    group_rows_by_entity,
    infer_entity_names,
)
from .questions import build_question_records, rewrite_stats
from .schema import Entity, NeighborBank, summarize
from .verify import NameCollisionChecker, VerificationThresholds, verify_neighbors

logger = logging.getLogger(__name__)


def build_neighbor_bank(
    rows: Sequence[Dict[str, str]],
    num_neighbors: int = 5,
    generator: Optional[Any] = None,
    extractor: Optional[Any] = None,
    probe: Optional[Any] = None,
    thresholds: Optional[VerificationThresholds] = None,
    question_key: str = "question",
    answer_key: str = "answer",
    entity_names: Optional[Sequence[str]] = None,
    block_size: Optional[int] = None,
    max_rounds: int = 3,
    extra_known_names: Sequence[str] = (),
    meta: Optional[Dict[str, Any]] = None,
) -> NeighborBank:
    """Build the bank for one forget split.

    Args:
        rows: raw forget-split rows (dicts with `question_key` / `answer_key`).
        num_neighbors: M, accepted neighbours required per entity.
        generator: callable `(attributes, num, avoid) -> List[Neighbor]`.
        extractor: callable `(name, texts) -> Attributes`.
        probe: optional `FamiliarityProbe` for the model-level unknown-ness gate.
        thresholds: acceptance rule for verification.
        entity_names: explicit per-row entity names; inferred when omitted.
        max_rounds: how often to re-ask the generator when candidates are rejected.
        extra_known_names: additional names the neighbours must not resemble
            (e.g. the retain-split authors, so neighbours stay strangers).

    Returns:
        A validated `NeighborBank`.
    """
    from .generate import TemplateNeighborGenerator  # local import: keeps deps lazy

    generator = generator or TemplateNeighborGenerator()
    extractor = extractor or HeuristicAttributeExtractor()
    thresholds = thresholds or VerificationThresholds()

    if entity_names is None:
        entity_names = infer_entity_names(
            rows, question_key=question_key, answer_key=answer_key, block_size=block_size
        )
    if len(entity_names) != len(rows):
        raise ValueError(
            f"entity_names has {len(entity_names)} items but there are {len(rows)} rows."
        )

    groups = group_rows_by_entity(entity_names)
    logger.info("Found %d entities over %d rows.", len(groups), len(rows))

    attributes = extract_attributes(
        rows, entity_names, extractor=extractor,
        question_key=question_key, answer_key=answer_key,
    )

    known_names = list(dict.fromkeys([name for name, _ in groups]) ) + list(extra_known_names)
    checker = NameCollisionChecker(
        known_names, max_similarity=thresholds.max_name_similarity
    )

    entities: Dict[str, Entity] = {}
    for position, (name, indices) in enumerate(groups):
        entity_id = f"e{position:03d}"
        attrs = attributes[name]
        accepted, attempted, rejected, tried_names = [], 0, 0, []

        for _ in range(max_rounds):
            missing = num_neighbors - len(accepted)
            if missing <= 0:
                break
            candidates = generator(attrs, missing, avoid=tried_names)
            attempted += len(candidates)
            tried_names.extend(c.name for c in candidates)
            passed = verify_neighbors(candidates, checker, probe=probe, thresholds=thresholds)
            rejected += len(candidates) - len(passed)
            accepted.extend(passed)

        if len(accepted) < num_neighbors:
            raise RuntimeError(
                f"Only {len(accepted)}/{num_neighbors} neighbours passed verification for "
                f"`{name}` after {max_rounds} rounds. Relax the thresholds, extend the "
                f"name pools, or lower M."
            )

        entities[entity_id] = Entity(
            entity_id=entity_id,
            name=name,
            attributes=attrs,
            question_indices=list(indices),
            neighbors=accepted[:num_neighbors],
            stats={"attempted": attempted, "rejected": rejected},
        )
        logger.info(
            "[%s] %s: %d neighbours accepted (%d rejected) -> %s",
            entity_id, name, len(accepted[:num_neighbors]), rejected,
            ", ".join(n.name for n in accepted[:num_neighbors]),
        )

    questions = build_question_records(
        rows, entity_names, entities,
        question_key=question_key, answer_key=answer_key,
    )

    bank = NeighborBank(
        meta={
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "num_neighbors": num_neighbors,
            "generator": type(generator).__name__,
            "extractor": type(extractor).__name__,
            "probe": type(probe).__name__ if probe is not None else None,
            "thresholds": vars(thresholds),
            **(meta or {}),
        },
        entities=list(entities.values()),
        questions=questions,
    )
    bank.validate(expect_m=num_neighbors)
    bank.meta["stats"] = {**summarize(bank), **rewrite_stats(questions)}
    return bank


def load_forget_rows(
    path: str = "locuslab/TOFU",
    name: str = "forget10",
    split: str = "train",
    **kwargs: Any,
) -> List[Dict[str, str]]:
    """Load a forget split with the `datasets` library."""
    import datasets  # local import: only needed when reading from the Hub

    data = datasets.load_dataset(path, name=name, split=split, **kwargs)
    return [dict(row) for row in data]
