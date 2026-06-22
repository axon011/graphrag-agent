"""Graph-augmented retrieval: link a question to the graph and pull a subgraph."""
from __future__ import annotations

from dataclasses import dataclass

from .graph import KnowledgeGraph


@dataclass
class Retrieval:
    seeds: list[str]
    nodes: set[str]
    triples: list[str]
    chunks: list[tuple[str, str]]  # (chunk id, text)


def retrieve(kg: KnowledgeGraph, question: str, hops: int = 2,
             max_chunks: int = 6) -> Retrieval:
    seeds = kg.find(question)
    nodes = kg.k_hop(seeds, hops) if seeds else set()
    triples = kg.triples(nodes)
    chunk_ids = kg.evidence_chunks(nodes)[:max_chunks]
    chunks = [(cid, kg.chunks.get(cid, "")) for cid in chunk_ids]
    return Retrieval(seeds=seeds, nodes=nodes, triples=triples, chunks=chunks)


def build_context(r: Retrieval) -> str:
    parts: list[str] = []
    if r.triples:
        parts.append("KNOWLEDGE GRAPH (relations):\n" + "\n".join(r.triples))
    if r.chunks:
        src = "\n\n".join(f"[{cid}] {text}" for cid, text in r.chunks)
        parts.append("SOURCE PASSAGES:\n" + src)
    return "\n\n".join(parts) if parts else "(no relevant context found)"
