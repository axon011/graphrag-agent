"""Knowledge graph: a thin, purposeful wrapper over networkx."""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from .models import Chunk, Entity, Extraction, Relation, slugify


class KnowledgeGraph:
    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()
        self.chunks: dict[str, str] = {}  # chunk id -> text

    # ---- construction ----
    def add_chunk(self, chunk: Chunk) -> None:
        self.chunks[chunk.id] = chunk.text

    def add_extraction(self, ext: Extraction) -> None:
        for e in ext.entities:
            self._merge_entity(e)
        for r in ext.relations:
            self._add_relation(r)

    def _merge_entity(self, e: Entity) -> None:
        nid = e.id
        if self.g.has_node(nid):
            node = self.g.nodes[nid]
            node["mentions"] = sorted(set(node.get("mentions", [])) | set(e.mentions))
            # keep a non-"concept" type if we learn one
            if node.get("type", "concept") == "concept" and e.type != "concept":
                node["type"] = e.type
        else:
            self.g.add_node(nid, name=e.name, type=e.type, mentions=list(e.mentions))

    def _add_relation(self, r: Relation) -> None:
        s, t = r.source_id, r.target_id
        if not self.g.has_node(s):
            self.g.add_node(s, name=r.source, type="concept", mentions=[])
        if not self.g.has_node(t):
            self.g.add_node(t, name=r.target, type="concept", mentions=[])
        self.g.add_edge(s, t, key=r.type, type=r.type,
                        description=r.description, evidence=r.evidence)

    # ---- access ----
    def num_nodes(self) -> int:
        return self.g.number_of_nodes()

    def num_edges(self) -> int:
        return self.g.number_of_edges()

    def name(self, nid: str) -> str:
        return self.g.nodes[nid].get("name", nid) if self.g.has_node(nid) else nid

    def find(self, text: str) -> list[str]:
        """Entity ids whose name appears in `text` (exact slug, substring, fuzzy)."""
        text_l = text.lower()
        hits: list[str] = []
        for nid, data in self.g.nodes(data=True):
            nm = data.get("name", nid).lower()
            if nm and nm in text_l:
                hits.append(nid)
        if hits:
            return hits
        # fall back to token overlap on the query slug
        q_tokens = set(slugify(text).split("_"))
        for nid in self.g.nodes:
            if set(nid.split("_")) & q_tokens:
                hits.append(nid)
        return hits

    def k_hop(self, seeds: list[str], hops: int) -> set[str]:
        """Undirected k-hop neighbourhood around the seed nodes."""
        nodes = {n for n in seeds if self.g.has_node(n)}
        frontier = set(nodes)
        und = self.g.to_undirected(as_view=True)
        for _ in range(max(0, hops)):
            nxt: set[str] = set()
            for n in frontier:
                nxt |= set(und.neighbors(n))
            new = nxt - nodes
            nodes |= nxt
            frontier = new
            if not frontier:
                break
        return nodes

    def triples(self, nodes: set[str]) -> list[str]:
        out: list[str] = []
        for s, t, data in self.g.edges(data=True):
            if s in nodes and t in nodes:
                out.append(f"{self.name(s)} --{data.get('type','related_to')}--> {self.name(t)}")
        return out

    def evidence_chunks(self, nodes: set[str]) -> list[str]:
        ids: list[str] = []
        for n in nodes:
            for cid in self.g.nodes[n].get("mentions", []):
                if cid not in ids:
                    ids.append(cid)
        for s, t, data in self.g.edges(data=True):
            if s in nodes and t in nodes and data.get("evidence") and data["evidence"] not in ids:
                ids.append(data["evidence"])
        return ids

    # ---- persistence ----
    def save(self, path: str | Path) -> None:
        data = {
            "nodes": [{"id": n, **d} for n, d in self.g.nodes(data=True)],
            "edges": [{"source": s, "target": t, **d} for s, t, d in self.g.edges(data=True)],
            "chunks": self.chunks,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeGraph":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        kg = cls()
        for n in data["nodes"]:
            nid = n.pop("id")
            kg.g.add_node(nid, **n)
        for e in data["edges"]:
            s, t = e.pop("source"), e.pop("target")
            kg.g.add_edge(s, t, key=e.get("type", "related_to"), **e)
        kg.chunks = data.get("chunks", {})
        return kg
