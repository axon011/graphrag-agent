"""GraphRAG Agent — knowledge-graph construction + graph-augmented retrieval."""
from .agent import GraphRAGAgent
from .chunk import chunk_text, load_chunks, relative_source, source_of
from .graph import KnowledgeGraph
from .models import Answer, Citation, Chunk, Entity, Relation

__version__ = "0.1.0"
__all__ = [
    "GraphRAGAgent",
    "KnowledgeGraph",
    "Answer",
    "Chunk",
    "Citation",
    "Entity",
    "Relation",
    "chunk_text",
    "load_chunks",
    "relative_source",
    "source_of",
]
