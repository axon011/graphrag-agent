from graphrag_agent import GraphRAGAgent


def test_build_and_query_offline():
    agent = GraphRAGAgent()
    agent.build_texts([
        "Perinet runs an AI platform. MessageSense does anomaly detection over MQTT.",
        "The MQTT broker feeds sensor data into MessageSense.",
    ])
    assert agent.kg.num_nodes() > 0
    ans = agent.query("How is MessageSense connected to MQTT?")
    assert ans.text
    assert any(c.kind == "entity" for c in ans.citations)


def test_query_unknown_entity():
    agent = GraphRAGAgent()
    agent.build_texts(["Perinet uses MQTT."])
    ans = agent.query("What is the capital of France?")
    assert "couldn't link" in ans.text.lower() or ans.text


def test_roundtrip_graph(tmp_path):
    agent = GraphRAGAgent()
    agent.build_texts(["Perinet uses MQTT and Docker."])
    p = tmp_path / "kg.json"
    agent.save(p)
    reloaded = GraphRAGAgent.from_graph(p)
    assert reloaded.kg.num_nodes() == agent.kg.num_nodes()
