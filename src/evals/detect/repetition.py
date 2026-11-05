"""Repetition in answers: a looping answer ("a diligent researcher, a diligent researcher, a, a, a ...") or the same canned answer to many
prompts is a tell of an unlearning intervention (`is_degenerate` of the text metrics flags the first only for short n-gram windows)."""

from __future__ import annotations
from typing import List, Sequence, Set


def _shingles(text: str, n: int) -> Set[tuple]:
    words = text.lower().split()
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def intra_repetition(text: str, n: int = 3) -> float:
    """1 - (distinct word n-grams / word n-grams) of ONE answer: 0 = no repeated n-gram, towards 1 = a loop. Answers shorter than 2n words: 0."""
    words = text.lower().split()
    if len(words) < 2 * n:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def near_duplicates(texts: Sequence[str], threshold: float = 0.8, n: int = 3) -> List[bool]:
    """For every answer of a SET: is there another one with a Jaccard similarity of the word n-gram sets >= `threshold`? (the same canned
    answer to different questions). Empty answers are not duplicates of each other."""
    sets = [_shingles(t, n) for t in texts]
    flags = [False] * len(sets)
    for i in range(len(sets)):
        if not sets[i]:
            continue
        for j in range(i + 1, len(sets)):
            if not sets[j]:
                continue
            if len(sets[i] & sets[j]) / len(sets[i] | sets[j]) >= threshold:
                flags[i] = flags[j] = True
    return flags
