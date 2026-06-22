"""GraphRAGAgent — build a knowledge graph and answer questions over it."""
from __future__ import annotations

from pathlib import Path

from .chunk import load_chunks, chunk_text
from .config import Config, load_config
from .extract import extract_chunk
from .graph import KnowledgeGraph
from .llm import LLM
from .models import Answer, Citation
from .retrieve import build_context, retrieve

ANSWER_SYSTEM = (
    "You answer questions using ONLY the provided knowledge graph relations and "
    "source passages. Be concise and concrete. If the context does not contain "
    "the answer, say so. Do not invent facts."
)


class GraphRAGAgent:
    def __init__(self, config: Config | None = None, kg: KnowledgeGraph | None = None):
        self.cfg = config or load_config()
        self.llm = LLM(self.cfg)
        self.kg = kg or KnowledgeGraph()

    # ---- build ----
    def build(self, paths: list[str | Path]) -> KnowledgeGraph:
        for p in paths:
            chunks = load_chunks(p, self.cfg.chunk_size, self.cfg.chunk_overlap)
            self._ingest(chunks)
        return self.kg

    def build_texts(self, texts: list[str], source: str = "inline") -> KnowledgeGraph:
        chunks = []
        for i, t in enumerate(texts):
            chunks += chunk_text(t, f"{source}-{i}", self.cfg.chunk_size, self.cfg.chunk_overlap)
        self._ingest(chunks)
        return self.kg

    def _ingest(self, chunks) -> None:
        for ch in chunks:
            self.kg.add_chunk(ch)
            self.kg.add_extraction(extract_chunk(ch, self.llm))

    # ---- query ----
    def query(self, question: str) -> Answer:
        r = retrieve(self.kg, question, hops=self.cfg.hops)
        citations = (
            [Citation(kind="entity", ref=self.kg.name(n)) for n in r.seeds]
            + [Citation(kind="relation", ref=tr) for tr in r.triples[:8]]
            + [Citation(kind="chunk", ref=cid) for cid, _ in r.chunks]
        )
        context = build_context(r)

        if self.llm.available:
            try:
                text = self.llm.complete(
                    ANSWER_SYSTEM, f"Question: {question}\n\nContext:\n{context}"
                )
            except Exception:  # noqa: BLE001
                text = self._heuristic_answer(question, r)
        else:
            text = self._heuristic_answer(question, r)

        return Answer(text=text, citations=citations, subgraph_triples=r.triples)

    def _heuristic_answer(self, question: str, r) -> str:
        if not r.seeds:
            return ("I couldn't link the question to any entity in the graph. "
                    "Try terms that appear in the documents.")
        names = ", ".join(self.kg.name(n) for n in r.seeds)
        lines = [f"Entities matched: {names}."]
        if r.triples:
            lines.append("Connections found:")
            lines += [f"  - {t}" for t in r.triples[:8]]
        else:
            lines.append("No relations connect these entities in the graph.")
        lines.append("(Set OPENAI_API_KEY for a synthesised natural-language answer.)")
        return "\n".join(lines)

    # ---- persistence ----
    def save(self, path: str | Path) -> None:
        self.kg.save(path)

    @classmethod
    def from_graph(cls, path: str | Path, config: Config | None = None) -> "GraphRAGAgent":
        return cls(config=config, kg=KnowledgeGraph.load(path))
