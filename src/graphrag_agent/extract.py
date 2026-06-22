"""Entity + relation extraction from a chunk.

Uses the LLM when available; otherwise a transparent regex/co-occurrence
heuristic so the pipeline runs (and CI passes) without an API key.
"""
from __future__ import annotations

import re

from .llm import LLM
from .models import Chunk, Entity, Extraction, Relation

EXTRACT_SYSTEM = (
    "You extract a knowledge graph from text. Return STRICT JSON with two keys: "
    '"entities" and "relations". '
    'Each entity: {"name": str, "type": one of [person, org, product, technology, '
    'concept, place, event]}. '
    'Each relation: {"source": entity name, "target": entity name, "type": short '
    'snake_case verb phrase, "description": one short clause}. '
    "Only use entity names that appear in the text. Do not invent facts."
)

# words that look capitalised but are not entities
_STOP = {
    "the", "a", "an", "this", "that", "these", "those", "it", "its", "i", "we",
    "they", "he", "she", "but", "and", "or", "for", "in", "on", "at", "to", "of",
    "with", "as", "by", "from", "most", "some", "each", "both", "when", "while",
    "if", "so", "is", "are", "was", "were", "be",
}
_CANDIDATE = re.compile(
    r"\b([A-Z][a-zA-Z0-9]+(?:[ -][A-Z][a-zA-Z0-9]+)*|[A-Z]{2,}(?:\d+)?)\b"
)


def _clean(name: str) -> str:
    """Drop leading stopword tokens from a candidate (e.g. 'Its AI' -> 'AI')."""
    tokens = name.split()
    while tokens and tokens[0].lower() in _STOP:
        tokens.pop(0)
    return " ".join(tokens).strip()


def _heuristic(chunk: Chunk) -> Extraction:
    seen: list[str] = []
    for m in _CANDIDATE.finditer(chunk.text):
        name = _clean(m.group(1).strip())
        if not name or name.lower() in _STOP or len(name) < 2:
            continue
        if name not in seen:
            seen.append(name)
    entities = [Entity(name=n, mentions=[chunk.id]) for n in seen]
    # link entities that co-occur, in order of first appearance (a readable chain)
    relations = [
        Relation(source=seen[i], target=seen[i + 1], type="co_occurs_with",
                 description="appears together in the source text", evidence=chunk.id)
        for i in range(len(seen) - 1)
    ]
    return Extraction(entities=entities, relations=relations)


def _from_llm(chunk: Chunk, llm: LLM) -> Extraction:
    data = llm.complete_json(EXTRACT_SYSTEM, chunk.text)
    entities = [
        Entity(name=e["name"], type=e.get("type", "concept"), mentions=[chunk.id])
        for e in data.get("entities", [])
        if e.get("name")
    ]
    valid = {e.name for e in entities}
    relations = [
        Relation(
            source=r["source"], target=r["target"],
            type=r.get("type", "related_to"),
            description=r.get("description", ""), evidence=chunk.id,
        )
        for r in data.get("relations", [])
        if r.get("source") in valid and r.get("target") in valid
    ]
    return Extraction(entities=entities, relations=relations)


def extract_chunk(chunk: Chunk, llm: LLM) -> Extraction:
    if llm.available:
        try:
            return _from_llm(chunk, llm)
        except Exception:  # noqa: BLE001 - degrade gracefully, never crash a build
            return _heuristic(chunk)
    return _heuristic(chunk)
