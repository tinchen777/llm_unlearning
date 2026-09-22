"""Attribute-matched neighbour generation (stage 1 of the FWT pipeline)."""

from __future__ import annotations

from .attributes import (
    HeuristicAttributeExtractor,
    LLMAttributeExtractor,
    extract_attributes,
    infer_entity_names,
)
from .generate import (
    LLMNeighborGenerator,
    TemplateNeighborGenerator,
    build_substitutions,
    get_generator,
)
from .llm import EchoClient, OpenAICompatibleClient, get_client
from .pipeline import build_neighbor_bank, load_forget_rows
from .questions import apply_substitutions, build_question_records
from .schema import Attributes, Entity, Neighbor, NeighborBank, QuestionRecord, summarize
from .verify import (
    FamiliarityProbe,
    FamiliarityResult,
    NameCollisionChecker,
    VerificationThresholds,
    verify_neighbors,
)

__all__ = [
    "Attributes", "Entity", "Neighbor", "NeighborBank", "QuestionRecord", "summarize",
    "HeuristicAttributeExtractor", "LLMAttributeExtractor", "extract_attributes",
    "infer_entity_names", "TemplateNeighborGenerator", "LLMNeighborGenerator",
    "get_generator", "build_substitutions", "apply_substitutions",
    "build_question_records", "build_neighbor_bank", "load_forget_rows",
    "NameCollisionChecker", "FamiliarityProbe", "FamiliarityResult",
    "VerificationThresholds", "verify_neighbors",
    "OpenAICompatibleClient", "EchoClient", "get_client",
]
