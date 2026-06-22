from graphrag_agent.graph import KnowledgeGraph
from graphrag_agent.models import Chunk, Entity, Extraction, Relation


def _kg():
    kg = KnowledgeGraph()
    kg.add_chunk(Chunk(id="d#0", text="Perinet uses MQTT.", source="d"))
    kg.add_extraction(Extraction(
        entities=[Entity(name="Perinet", type="org", mentions=["d#0"]),
                  Entity(name="MQTT", type="technology", mentions=["d#0"])],
        relations=[Relation(source="Perinet", target="MQTT", type="uses",
                            description="connects to streams", evidence="d#0")],
    ))
    return kg


def test_build_and_counts():
    kg = _kg()
    assert kg.num_nodes() == 2
    assert kg.num_edges() == 1


def test_merge_dedup():
    kg = _kg()
    kg.add_extraction(Extraction(entities=[Entity(name="Perinet", mentions=["d#1"])]))
    assert kg.num_nodes() == 2  # not duplicated
    assert "d#1" in kg.g.nodes["perinet"]["mentions"]


def test_find_and_khop_and_triples():
    kg = _kg()
    seeds = kg.find("tell me about Perinet")
    assert "perinet" in seeds
    nodes = kg.k_hop(seeds, hops=1)
    assert "mqtt" in nodes
    triples = kg.triples(nodes)
    assert any("uses" in t for t in triples)


def test_save_load(tmp_path):
    kg = _kg()
    p = tmp_path / "kg.json"
    kg.save(p)
    kg2 = KnowledgeGraph.load(p)
    assert kg2.num_nodes() == 2
    assert kg2.num_edges() == 1
    assert kg2.chunks["d#0"] == "Perinet uses MQTT."
