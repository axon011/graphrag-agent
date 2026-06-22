from graphrag_agent.config import Config
from graphrag_agent.extract import extract_chunk
from graphrag_agent.llm import LLM, loads_lenient
from graphrag_agent.models import Chunk


def _offline_llm():
    return LLM(Config(api_key=None))


def test_heuristic_finds_entities():
    ch = Chunk(id="d#0", text="Perinet connects MQTT to Kubernetes via FastAPI.",
               source="d")
    ext = extract_chunk(ch, _offline_llm())
    names = {e.name for e in ext.entities}
    assert {"Perinet", "MQTT", "Kubernetes", "FastAPI"} <= names


def test_heuristic_makes_relations():
    ch = Chunk(id="d#0", text="Perinet uses MQTT and Docker.", source="d")
    ext = extract_chunk(ch, _offline_llm())
    assert len(ext.relations) >= 1
    assert all(r.evidence == "d#0" for r in ext.relations)


def test_stopwords_filtered():
    ch = Chunk(id="d#0", text="The system runs. It works.", source="d")
    ext = extract_chunk(ch, _offline_llm())
    names = {e.name for e in ext.entities}
    assert "The" not in names and "It" not in names


def test_loads_lenient_strips_fences():
    assert loads_lenient('```json\n{"a": 1}\n```') == {"a": 1}
    assert loads_lenient('{"a": 1} trailing prose') == {"a": 1}
