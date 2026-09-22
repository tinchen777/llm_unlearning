"""Data structures for the attribute-matched neighbour bank.

The bank produced by `fwt/scripts/build_neighbors.py` is the single artefact
shared by every downstream stage (layer selection, training, evaluation), so it
is kept as a plain JSON document with an explicit schema:

```
{
  "meta":      {...},                     # provenance: dataset, split, M, backend, ...
  "entities":  [Entity, ...],             # one per forgotten entity
  "questions": [QuestionRecord, ...]      # one per row of the forget split
}
```

`QuestionRecord.neighbor_questions[j]` is the forget question rewritten for the
j-th neighbour of its entity, so a trainer only needs the row index of a forget
sample to look up its redirection targets.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

# Attributes that must MATCH between a forgotten entity and its neighbours.
# These define the "same kind of person" equivalence class.
MATCHED_FIELDS = (
    "nationality",
    "birth_decade",
    "gender",
    "profession",
    "genre",
    "parent_professions",
)
# Attributes that must DIFFER: they are the identity-specific facts that get
# substituted when a forget question is rewritten for a neighbour.
SPECIFIC_FIELDS = (
    "name",
    "birth_city",
    "birth_date",
    "book_titles",
    "award",
)


@dataclass
class Attributes:
    """Structured profile of an entity (matched fields + specific facts)."""

    name: str = ""
    nationality: str = ""
    birth_decade: str = ""
    birth_city: str = ""
    birth_date: str = ""
    gender: str = ""
    profession: str = ""
    genre: str = ""
    award: str = ""
    parent_professions: List[str] = field(default_factory=list)
    book_titles: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def matched(self) -> Dict[str, Any]:
        """The sub-profile a neighbour has to reproduce."""
        return {k: getattr(self, k) for k in MATCHED_FIELDS}

    def specific(self) -> Dict[str, Any]:
        """The sub-profile a neighbour has to avoid."""
        return {k: getattr(self, k) for k in SPECIFIC_FIELDS}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Attributes:
        data = dict(data or {})
        known = {f: data.pop(f) for f in list(data) if f in cls.__dataclass_fields__}
        extra = known.pop("extra", {}) or {}
        extra.update(data)  # keep unknown keys instead of dropping them
        return cls(extra=extra, **known)


@dataclass
class Neighbor:
    """A synthetic stranger that shares the matched attributes of an entity."""

    name: str
    attributes: Attributes
    # entity-specific string -> neighbour-specific string, used to rewrite questions
    substitutions: Dict[str, str] = field(default_factory=dict)
    verification: Dict[str, Any] = field(default_factory=dict)
    passed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "attributes": self.attributes.to_dict(),
            "substitutions": self.substitutions,
            "verification": self.verification,
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Neighbor:
        return cls(
            name=data["name"],
            attributes=Attributes.from_dict(data.get("attributes")),
            substitutions=dict(data.get("substitutions", {})),
            verification=dict(data.get("verification", {})),
            passed=bool(data.get("passed", True)),
        )


@dataclass
class Entity:
    """A forgotten entity together with its accepted neighbours."""

    entity_id: str
    name: str
    attributes: Attributes
    question_indices: List[int] = field(default_factory=list)
    neighbors: List[Neighbor] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "attributes": self.attributes.to_dict(),
            "question_indices": self.question_indices,
            "neighbors": [n.to_dict() for n in self.neighbors],
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Entity:
        return cls(
            entity_id=data["entity_id"],
            name=data["name"],
            attributes=Attributes.from_dict(data.get("attributes")),
            question_indices=list(data.get("question_indices", [])),
            neighbors=[Neighbor.from_dict(n) for n in data.get("neighbors", [])],
            stats=dict(data.get("stats", {})),
        )


@dataclass
class QuestionRecord:
    """One row of the forget split and its neighbour-rewritten counterparts."""

    index: int
    entity_id: str
    question: str
    answer: str = ""
    neighbor_questions: List[str] = field(default_factory=list)
    # True when the rewrite actually changed the text (a question that mentions
    # neither the name nor any specific fact cannot be personalised)
    rewritten: List[bool] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QuestionRecord:
        return cls(
            index=int(data["index"]),
            entity_id=data["entity_id"],
            question=data["question"],
            answer=data.get("answer", ""),
            neighbor_questions=list(data.get("neighbor_questions", [])),
            rewritten=list(data.get("rewritten", [])),
        )


@dataclass
class NeighborBank:
    """The full artefact: entities, their neighbours and the rewritten questions."""

    meta: Dict[str, Any] = field(default_factory=dict)
    entities: List[Entity] = field(default_factory=list)
    questions: List[QuestionRecord] = field(default_factory=list)

    # ------------------------------------------------------------------ io
    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": self.meta,
            "entities": [e.to_dict() for e in self.entities],
            "questions": [q.to_dict() for q in self.questions],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NeighborBank:
        return cls(
            meta=dict(data.get("meta", {})),
            entities=[Entity.from_dict(e) for e in data.get("entities", [])],
            questions=[QuestionRecord.from_dict(q) for q in data.get("questions", [])],
        )

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> NeighborBank:
        path = Path(path)
        try:
            with path.open("r", encoding="utf-8") as f:
                return cls.from_dict(json.load(f))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Neighbour bank not found at `{path}`. Build it first with "
                f"`python fwt/scripts/build_neighbors.py --out {path}`."
            ) from e

    # -------------------------------------------------------------- lookups
    @property
    def num_neighbors(self) -> int:
        """Number of neighbours per entity (M); 0 if the bank is empty."""
        if not self.entities:
            return 0
        return min(len(e.neighbors) for e in self.entities)

    def entity(self, entity_id: str) -> Entity:
        for e in self.entities:
            if e.entity_id == entity_id:
                return e
        raise KeyError(f"Unknown entity_id `{entity_id}`.")

    def entity_of_question(self, index: int) -> Entity:
        return self.entity(self.question(index).entity_id)

    def question(self, index: int) -> QuestionRecord:
        record = self._question_by_index.get(int(index))
        if record is None:
            raise KeyError(
                f"Row index {index} is not covered by this neighbour bank "
                f"(it holds {len(self.questions)} questions)."
            )
        return record

    def neighbor_questions(self, index: int, num: Optional[int] = None) -> List[str]:
        """Neighbour-rewritten variants of the forget question at `index`."""
        qs = self.question(index).neighbor_questions
        return qs if num is None else qs[:num]

    def neighbor_names(self) -> List[str]:
        return [n.name for e in self.entities for n in e.neighbors]

    def entity_names(self) -> List[str]:
        return [e.name for e in self.entities]

    def covered_indices(self) -> List[int]:
        return sorted(self._question_by_index)

    def validate(self, expect_m: Optional[int] = None) -> None:
        """Raise if the bank is internally inconsistent (cheap sanity gate)."""
        ids = {e.entity_id for e in self.entities}
        if len(ids) != len(self.entities):
            raise ValueError("Duplicate entity_id in neighbour bank.")
        for q in self.questions:
            if q.entity_id not in ids:
                raise ValueError(
                    f"Question {q.index} refers to unknown entity `{q.entity_id}`."
                )
            n_expected = len(self.entity(q.entity_id).neighbors)
            if len(q.neighbor_questions) != n_expected:
                raise ValueError(
                    f"Question {q.index} has {len(q.neighbor_questions)} neighbour "
                    f"questions but its entity has {n_expected} neighbours."
                )
        if expect_m is not None and self.num_neighbors < expect_m:
            raise ValueError(
                f"Bank provides only {self.num_neighbors} neighbours per entity, "
                f"but {expect_m} were requested."
            )

    @property
    def _question_by_index(self) -> Dict[int, QuestionRecord]:
        cache = getattr(self, "_q_cache", None)
        if cache is None or len(cache) != len(self.questions):
            cache = {q.index: q for q in self.questions}
            object.__setattr__(self, "_q_cache", cache)
        return cache


def summarize(bank: NeighborBank) -> Dict[str, Any]:
    """Aggregate statistics for the appendix table of the paper."""
    attempted = sum(e.stats.get("attempted", 0) for e in bank.entities)
    rejected = sum(e.stats.get("rejected", 0) for e in bank.entities)
    rewritten = [r for q in bank.questions for r in q.rewritten]
    return {
        "num_entities": len(bank.entities),
        "num_questions": len(bank.questions),
        "neighbors_per_entity": bank.num_neighbors,
        "candidates_attempted": attempted,
        "candidates_rejected": rejected,
        "rejection_rate": (rejected / attempted) if attempted else 0.0,
        "question_rewrite_rate": (sum(rewritten) / len(rewritten)) if rewritten else 0.0,
    }


def dedup_preserve_order(items: Sequence[str]) -> List[str]:
    seen, out = set(), []
    for it in items:
        key = it.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out
