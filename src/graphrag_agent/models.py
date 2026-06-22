"""Typed data models for the GraphRAG pipeline."""
from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field


def slugify(name: str) -> str:
    """Normalise an entity name into a stable id."""
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    return s.strip("_")


class Chunk(BaseModel):
    """A windowed span of a source document."""

    id: str
    text: str
    source: str


class Entity(BaseModel):
    """A node in the knowledge graph."""

    name: str
    type: str = "concept"
    mentions: list[str] = Field(default_factory=list)  # chunk ids

    @property
    def id(self) -> str:
        return slugify(self.name)


class Relation(BaseModel):
    """A typed, directed edge between two entities."""

    source: str  # entity name (raw); resolved to id at graph-build time
    target: str
    type: str = "related_to"
    description: str = ""
    evidence: Optional[str] = None  # chunk id

    @property
    def source_id(self) -> str:
        return slugify(self.source)

    @property
    def target_id(self) -> str:
        return slugify(self.target)


class Extraction(BaseModel):
    """What the extractor returns for a single chunk."""

    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


class Citation(BaseModel):
    """A reference the answer leaned on."""

    kind: str  # "entity" | "relation" | "chunk"
    ref: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.kind}] {self.ref}"


class Answer(BaseModel):
    """The final grounded response."""

    text: str
    citations: list[Citation] = Field(default_factory=list)
    subgraph_triples: list[str] = Field(default_factory=list)
