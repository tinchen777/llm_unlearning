"""Rewriting forget questions into neighbour questions.

The redirection target of a forget question is *the same question asked about a
stranger*. Rewriting therefore substitutes every identity-specific string of the
target entity (name, birth city, exact date, titles, award) with the neighbour's
counterpart, while leaving the matched attributes (nationality, decade, genre,
...) untouched - those are exactly what makes the neighbour attribute-matched.

Questions that mention none of those strings (rare: generic "what genre does the
author write in?" phrasings) cannot be personalised; they are kept verbatim and
flagged via `QuestionRecord.rewritten` so training can down-weight or drop them.
"""

from __future__ import annotations
import re
from typing import Dict, List, Sequence, Tuple

from .schema import Entity, QuestionRecord


def apply_substitutions(text: str, substitutions: Dict[str, str]) -> Tuple[str, bool]:
    """Replace every key by its value, longest key first, word-boundary aware.

    Returns the rewritten text and whether anything changed.
    """
    if not text or not substitutions:
        return text, False

    ordered = sorted(substitutions.items(), key=lambda kv: len(kv[0]), reverse=True)
    pattern = "|".join(re.escape(k) for k, _ in ordered)
    # \b only works next to word characters; names may end with punctuation
    # (Al-Kuwaiti), so guard with lookarounds on word characters instead.
    regex = re.compile(rf"(?<!\w)(?:{pattern})(?!\w)")
    lookup = {k: v for k, v in ordered}

    changed = False

    def _replace(match: re.Match) -> str:
        nonlocal changed
        changed = True
        return lookup[match.group(0)]

    return regex.sub(_replace, text), changed


def build_question_records(
    rows: Sequence[Dict[str, str]],
    entity_names: Sequence[str],
    entities: Dict[str, Entity],
    question_key: str = "question",
    answer_key: str = "answer",
) -> List[QuestionRecord]:
    """One `QuestionRecord` per row, with one rewritten question per neighbour."""
    name_to_entity = {e.name: e for e in entities.values()}
    records: List[QuestionRecord] = []

    for index, (row, entity_name) in enumerate(zip(rows, entity_names)):
        entity = name_to_entity[entity_name]
        question = str(row.get(question_key, ""))
        neighbor_questions: List[str] = []
        rewritten_flags: List[bool] = []
        for neighbor in entity.neighbors:
            text, changed = apply_substitutions(question, neighbor.substitutions)
            neighbor_questions.append(text)
            rewritten_flags.append(changed)
        records.append(
            QuestionRecord(
                index=index,
                entity_id=entity.entity_id,
                question=question,
                answer=str(row.get(answer_key, "")),
                neighbor_questions=neighbor_questions,
                rewritten=rewritten_flags,
            )
        )
    return records


def rewrite_stats(records: Sequence[QuestionRecord]) -> Dict[str, float]:
    flags = [flag for r in records for flag in r.rewritten]
    unpersonalised = [r.index for r in records if r.rewritten and not any(r.rewritten)]
    return {
        "rewrite_rate": (sum(flags) / len(flags)) if flags else 0.0,
        "num_unpersonalised_questions": len(unpersonalised),
        "unpersonalised_indices": unpersonalised[:20],
    }
