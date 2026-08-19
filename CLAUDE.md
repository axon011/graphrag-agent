# graphrag-agent — working context

Standalone library: chunk → extract → `KnowledgeGraph` → k-hop retrieval → cited answers.
No web layer, no database. It must stay useful without any consumer.

Primary consumer today is **`../graphrag-studio`**, which is being extended into a document
librarian. Changes needed by that work land *here* when they belong to the library — see
`../graphrag-studio/CLAUDE.md` for the wider project.

---

## `source` is a document identity — treat it as one

This is the thing to understand before touching `chunk.py`.

Chunk ids are built as `f"{source}#{i}"`, and `Entity.mentions` / `Relation.evidence` store
those ids. So `source` is not a cosmetic label — it is how any consumer recovers *which
document* an entity came from:

```
entity → mentions[] → chunk_id → source_of(chunk_id) → document
```

### The bug that was fixed (2026-08-15)

`load_chunks()` set `source=p.name` — the **basename**. In a real corpus with many
`README.md` files, every one of them resolved to the same source and merged into a single
phantom document whose labels were the union of unrelated files. No error, no symptom, just
quietly wrong data for every consumer downstream.

Now:

```python
load_chunks(path, root=corpus_root)     # source = "alpha/docs/README.md"
load_chunks(path)                       # source = "README.md"  (legacy, collides)
```

- `relative_source(path, root)` — repo-relative, POSIX slashes, falls back to the basename
  when `root` is None or the path is outside it.
- `source_of(chunk_id)` — the canonical inverse. Consumers should call this, not re-split.
- `chunk_text()` **raises** on a source containing `#`, because the id would be ambiguous.
- `GraphRAGAgent.build(paths, root=...)` threads `root` through.

`tests/test_chunk.py` pins both behaviours, including a test asserting the *old* collision
when `root` is omitted — so a regression is visible rather than silent.

**When adding an API that takes or returns `source`, pass `root`.** Defaulting to the
basename is the trap.

---

## `Extraction` is bound to its chunk

`extract.py` writes `chunk.id` into every entity and relation:

```python
Entity(name=n, mentions=[chunk.id])
Relation(..., evidence=chunk.id)
```

An `Extraction` is therefore **not portable between chunks**. Anything that stores and
replays one — a cache, a batch retry, a fixture — must strip the ids and rebind them at the
destination, or it will attribute entities to the wrong document. Studio's
`app/cache.py` does this via `dumps()` / `loads(payload, chunk_id)`; copy that shape rather
than inventing another.

---

## Ingest is the performance problem

```python
def _ingest(self, chunks) -> None:
    for ch in chunks:
        self.kg.add_chunk(ch)
        self.kg.add_extraction(extract_chunk(ch, self.llm))
```

Serial, one LLM call per chunk, no cache, no resume. On the target corpus (~4,900 chunks)
that is ~2.7 hours with total loss on any crash.

Work in progress adds bounded concurrency. Two constraints:

- **Extraction parallelises; merging does not.** `KnowledgeGraph._merge_entity` mutates a
  shared `networkx` graph. Run `extract_chunk` on a worker pool, merge on one thread.
- **Caching lives in the consumer, not here.** The library stays dependency-light and
  storage-free; Studio owns the SQLite cache.

---

## LLM access

`llm.py` resolves `GRAPHRAG_LLM=auto|api|codex|claude|gemini|off` and exposes `complete()` /
`complete_json()`. `auto` picks the first available: API key, then codex, claude, gemini,
else off.

This is **the** LLM abstraction for the whole project. Do not add LangChain chat models
anywhere in this repo or in Studio — the provider switch already covers subscription-backed
CLIs with no API key.

`extract_chunk` degrades to `_heuristic()` on any provider exception rather than crashing a
build. That is deliberate: a long ingest should not die on one bad response. It also means a
silent quality drop, so a consumer that cares must check `llm.available` itself.

---

## Testing

```bash
python -m pytest tests/ -q      # 21 tests, offline, no network
```

`tests/conftest.py` forces `GRAPHRAG_LLM=off` for the whole suite. Keep it hermetic — a test
that needs a provider is not a test that runs anywhere useful.

---

## Conventions

- `from __future__ import annotations`, type hints throughout, Pydantic models in `models.py`.
- The public surface is `__init__.py`'s `__all__`. Adding a helper consumers need means
  exporting it there.
- Keep the dependency list small: `networkx`, `pydantic`, `httpx`. No storage, no web
  framework, no orchestration library.
- Don't commit unless asked. No Co-Authored-By lines.
