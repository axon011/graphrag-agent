# GraphRAG Agent

Build a **knowledge graph** from your documents, then answer questions by **traversing the graph** instead of just stuffing chunks into a prompt. Entities and relations are extracted with an LLM, stored as a graph, and at query time the agent finds the relevant entities, expands their neighbourhood, and answers from that subgraph — with citations back to the source.

Plain vector RAG retrieves *passages*. GraphRAG retrieves *structure* — so it can answer multi-hop questions ("how is X connected to Y?") that flat retrieval misses.

```
docs ──▶ chunk ──▶ extract (entities + relations) ──▶ knowledge graph
                                                            │
question ──▶ link to entities ──▶ k-hop subgraph ──▶ grounded answer + citations
```

## Why it exists

Most RAG systems flatten everything into a pile of text chunks and hope the embedding similarity surfaces the right ones. That breaks down the moment a question spans more than one document or needs a *relationship* ("which products use the component that the recalled supplier shipped?"). A knowledge graph keeps the relationships explicit, so the retrieval step can actually follow them.

This project is a small, readable implementation of that idea — the pieces you'd actually build, without a heavyweight framework.

## Features

- **LLM entity + relation extraction** with structured (Pydantic-validated) output
- **Knowledge graph** over the corpus (`networkx`), deduplicated and typed
- **Graph-augmented retrieval** — entity linking → k-hop expansion → subgraph context
- **Grounded answers** with citations to the entities, relations and source chunks used
- **Offline fallback** — runs end to end with a heuristic extractor when no API key is set, so the demo and tests work out of the box
- **Provider-agnostic** LLM client (any OpenAI-compatible endpoint)
- Typed, tested, containerised, CI on every push

## Quickstart

```bash
pip install -r requirements.txt

# runs offline (heuristic extractor) — no key needed
python -m graphrag_agent.cli build examples/sample.txt
python -m graphrag_agent.cli ask "How is Perinet connected to MQTT?"

# with a real LLM (better extraction + answers)
export OPENAI_API_KEY=sk-...
python -m graphrag_agent.cli build examples/sample.txt --llm
python -m graphrag_agent.cli ask "How is Perinet connected to MQTT?" --llm
```

Or from Python:

```python
from graphrag_agent import GraphRAGAgent

agent = GraphRAGAgent()
agent.build(["docs/a.md", "docs/b.md"])
answer = agent.query("Which systems depend on the sensor stream?")
print(answer.text)
for c in answer.citations:
    print(" -", c)
```

## How it works

1. **Chunk** — documents are split into overlapping windows.
2. **Extract** — each chunk goes to the LLM with a schema; it returns typed entities and relations. Entities are normalised and merged across chunks by name + type.
3. **Build** — entities become graph nodes, relations become typed edges, each carrying the chunk it came from as evidence.
4. **Link** — a question is mapped to the entities it mentions (exact + fuzzy match against the graph).
5. **Expand** — the agent takes those seed entities and pulls their k-hop neighbourhood.
6. **Answer** — the subgraph (as triples) plus the underlying chunks become the context; the LLM answers and is asked to cite the entities/relations it used.

## Architecture

```
src/graphrag_agent/
  models.py     # Entity, Relation, Chunk, Answer  (Pydantic)
  llm.py        # OpenAI-compatible client + offline heuristic fallback
  chunk.py      # text chunking
  extract.py    # LLM entity/relation extraction (+ heuristic fallback)
  graph.py      # KnowledgeGraph: networkx wrapper, dedup, k-hop, serialise
  retrieve.py   # entity linking + subgraph retrieval
  agent.py      # GraphRAGAgent: build() / query()
  cli.py        # build / ask commands
```

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | – | enables LLM extraction/answers (omit to run offline) |
| `OPENAI_BASE_URL` | OpenAI | any OpenAI-compatible endpoint |
| `GRAPHRAG_MODEL` | `gpt-4o-mini` | model id |
| `GRAPHRAG_HOPS` | `2` | k-hop expansion radius at query time |

## Tests

```bash
pytest -q          # runs fully offline
```

## License

MIT — see [LICENSE](LICENSE).
