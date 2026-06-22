"""Minimal end-to-end example (runs offline)."""
from graphrag_agent import GraphRAGAgent

agent = GraphRAGAgent()
agent.build(["examples/sample.txt"])
print(f"graph: {agent.kg.num_nodes()} entities, {agent.kg.num_edges()} relations\n")

answer = agent.query("How is MessageSense connected to MQTT?")
print(answer.text)
print("\ncitations:")
for c in answer.citations:
    print(" ", c)
