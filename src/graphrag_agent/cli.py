"""Command-line interface: build a graph, then ask it questions.

    python -m graphrag_agent.cli build docs/*.md [--graph kg.json]
    python -m graphrag_agent.cli ask "question" [--graph kg.json]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import GraphRAGAgent

DEFAULT_GRAPH = "kg.json"


def _build(args: argparse.Namespace) -> int:
    agent = GraphRAGAgent()
    agent.build(args.paths)
    agent.save(args.graph)
    print(
        f"Built graph: {agent.kg.num_nodes()} entities, "
        f"{agent.kg.num_edges()} relations -> {args.graph}"
    )
    if not agent.llm.available:
        print("(offline heuristic extractor — set OPENAI_API_KEY for richer extraction)")
    return 0


def _ask(args: argparse.Namespace) -> int:
    if not Path(args.graph).exists():
        print(f"No graph at {args.graph}. Run `build` first.", file=sys.stderr)
        return 2
    agent = GraphRAGAgent.from_graph(args.graph)
    ans = agent.query(args.question)
    print(ans.text)
    if ans.citations:
        print("\nCitations:")
        for c in ans.citations:
            print(f"  {c}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="graphrag_agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build a knowledge graph from documents")
    b.add_argument("paths", nargs="+", help="document paths")
    b.add_argument("--graph", default=DEFAULT_GRAPH)
    b.add_argument("--llm", action="store_true", help="(uses OPENAI_API_KEY if set)")
    b.set_defaults(func=_build)

    a = sub.add_parser("ask", help="ask the graph a question")
    a.add_argument("question")
    a.add_argument("--graph", default=DEFAULT_GRAPH)
    a.add_argument("--llm", action="store_true")
    a.set_defaults(func=_ask)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
