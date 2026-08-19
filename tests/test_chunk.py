"""Chunk sourcing — `source` is a document identity, so it must be unambiguous."""
import pytest

from graphrag_agent import GraphRAGAgent, chunk_text, load_chunks, relative_source, source_of


def _write(root, rel: str, text: str = "Perinet uses MQTT and Docker."):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_relative_source_uses_posix_path(tmp_path):
    p = _write(tmp_path, "docs/sub/README.md")
    assert relative_source(p, tmp_path) == "docs/sub/README.md"


def test_relative_source_falls_back_to_name_without_root(tmp_path):
    p = _write(tmp_path, "docs/README.md")
    assert relative_source(p) == "README.md"


def test_relative_source_falls_back_when_outside_root(tmp_path):
    outside = _write(tmp_path / "other", "README.md")
    root = tmp_path / "root"
    root.mkdir()
    assert relative_source(outside, root) == "README.md"


def test_same_filename_in_different_dirs_stays_distinct(tmp_path):
    """The bug this fix exists for: a corpus has many README.md files."""
    a = _write(tmp_path, "alpha/README.md")
    b = _write(tmp_path, "beta/README.md")

    sa = {c.source for c in load_chunks(a, root=tmp_path)}
    sb = {c.source for c in load_chunks(b, root=tmp_path)}

    assert sa == {"alpha/README.md"}
    assert sb == {"beta/README.md"}
    assert sa != sb


def test_without_root_same_filenames_collide(tmp_path):
    """Documents the old behaviour so a regression is visible, not silent."""
    a = _write(tmp_path, "alpha/README.md")
    b = _write(tmp_path, "beta/README.md")
    assert load_chunks(a)[0].source == load_chunks(b)[0].source == "README.md"


def test_source_of_inverts_chunk_id():
    chunks = chunk_text("Perinet uses MQTT.", source="docs/sub/README.md")
    assert chunks
    for c in chunks:
        assert source_of(c.id) == "docs/sub/README.md"


def test_source_of_survives_long_documents():
    """Multi-chunk documents: every id must map back to the same source."""
    text = "Perinet uses MQTT and Docker. " * 400
    chunks = chunk_text(text, source="a/b/c.md", size=200, overlap=40)
    assert len(chunks) > 1
    assert {source_of(c.id) for c in chunks} == {"a/b/c.md"}


def test_hash_in_source_is_rejected():
    """'#' would make the document unrecoverable from the chunk id."""
    with pytest.raises(ValueError, match="must not contain"):
        chunk_text("text", source="docs/we#ird.md")


def test_agent_build_threads_root_through(tmp_path):
    a = _write(tmp_path, "alpha/README.md", "Perinet runs MQTT services.")
    b = _write(tmp_path, "beta/README.md", "Docker containers host the broker.")

    agent = GraphRAGAgent()
    agent.build([a, b], root=tmp_path)

    sources = {source_of(cid) for cid in agent.kg.chunks}
    assert sources == {"alpha/README.md", "beta/README.md"}


def test_mentions_map_back_to_documents(tmp_path):
    """The join the document layer is built on: entity -> chunk id -> document."""
    a = _write(tmp_path, "alpha/notes.md", "Perinet runs MQTT services.")
    b = _write(tmp_path, "beta/notes.md", "Docker containers host the broker.")

    agent = GraphRAGAgent()
    agent.build([a, b], root=tmp_path)

    seen = {
        source_of(chunk_id)
        for _, data in agent.kg.g.nodes(data=True)
        for chunk_id in data.get("mentions", [])
    }
    assert seen <= {"alpha/notes.md", "beta/notes.md"}
    assert seen  # the offline extractor still produces mentions
