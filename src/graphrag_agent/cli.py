"""Command-line interface: build a graph, then ask it questions.

    python -m graphrag_agent.cli build docs/*.md [--graph kg.json]
    python -m graphrag_agent.cli ask "question" [--graph kg.json]
    python -m graphrag_agent.cli chat [--graph kg.json]   # interactive REPL

Running with no command drops into the interactive chat (if a graph exists).
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

from .agent import GraphRAGAgent

DEFAULT_GRAPH = "kg.json"

BANNER = r"""
  ____                 _     ____      _    ____
 / ___|_ __ __ _ _ __ | |__ |  _ \    / \  / ___|
| |  _| '__/ _` | '_ \| '_ \| |_) |  / _ \| |  _
| |_| | | | (_| | |_) | | | |  _ <  / ___ \ |_| |
 \____|_|  \__,_| .__/|_| |_|_| \_\/_/   \_\____|
                |_|        interactive graph agent
"""

HELP = """\
Commands:
  <question>          ask the knowledge graph a question
  /help, /?           show this help
  /stats              entities / relations in the loaded graph
  /hops <n>           set k-hop expansion radius (current query depth)
  /load <path>        load a different graph file
  /build <paths...>   ingest more documents into the current graph (then /save)
  /save [path]        persist the current graph (defaults to the loaded file)
  /llm                show which LLM provider is active
  /exit, /quit, Ctrl-D  leave
"""


def _print_answer(ans) -> None:
    print(f"\n{ans.text}")
    if ans.citations:
        print("\n citations:")
        for c in ans.citations:
            print(f"   {c}")
    print()


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


def _chat(args: argparse.Namespace) -> int:
    graph_path = args.graph
    if not Path(graph_path).exists():
        print(
            f"No graph at {graph_path}. Build one first:\n"
            f"  graphrag-agent build docs/*.md --graph {graph_path}",
            file=sys.stderr,
        )
        return 2

    agent = GraphRAGAgent.from_graph(graph_path)
    print(BANNER)
    provider = "LLM" if agent.llm.available else "offline heuristic"
    print(
        f" graph: {graph_path}  |  {agent.kg.num_nodes()} entities, "
        f"{agent.kg.num_edges()} relations  |  {provider}"
    )
    print(" type a question, or /help for commands. Ctrl-D to exit.\n")

    while True:
        try:
            line = input("graphrag> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue

        if line.startswith("/"):
            parts = line.split()
            cmd, rest = parts[0].lower(), parts[1:]
            if cmd in ("/exit", "/quit", "/q"):
                return 0
            if cmd in ("/help", "/?"):
                print(HELP)
            elif cmd == "/stats":
                print(f" {agent.kg.num_nodes()} entities, {agent.kg.num_edges()} relations "
                      f"(graph: {graph_path})")
            elif cmd == "/llm":
                print(f" provider available: {agent.llm.available} "
                      f"(GRAPHRAG_LLM={os.getenv('GRAPHRAG_LLM', 'auto')})")
            elif cmd == "/hops":
                if rest and rest[0].isdigit():
                    agent.cfg = dataclasses.replace(agent.cfg, hops=int(rest[0]))
                    print(f" k-hop radius set to {agent.cfg.hops}")
                else:
                    print(f" current k-hop radius: {agent.cfg.hops}  (usage: /hops <n>)")
            elif cmd == "/load":
                if not rest:
                    print(" usage: /load <path>")
                elif not Path(rest[0]).exists():
                    print(f" no such graph: {rest[0]}")
                else:
                    graph_path = rest[0]
                    agent = GraphRAGAgent.from_graph(graph_path)
                    print(f" loaded {graph_path}: {agent.kg.num_nodes()} entities, "
                          f"{agent.kg.num_edges()} relations")
            elif cmd == "/build":
                if not rest:
                    print(" usage: /build <path> [path...]")
                else:
                    missing = [p for p in rest if not Path(p).exists()]
                    if missing:
                        print(f" no such file(s): {', '.join(missing)}")
                    else:
                        agent.build(rest)
                        print(f" ingested {len(rest)} doc(s) -> now "
                              f"{agent.kg.num_nodes()} entities, "
                              f"{agent.kg.num_edges()} relations. /save to persist.")
            elif cmd == "/save":
                target = rest[0] if rest else graph_path
                agent.save(target)
                print(f" saved -> {target}")
            else:
                print(f" unknown command {cmd}. /help for the list.")
            continue

        try:
            _print_answer(agent.query(line))
        except Exception as e:  # noqa: BLE001
            print(f" error: {e}\n")

    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="graphrag-agent")
    sub = p.add_subparsers(dest="cmd")

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

    c = sub.add_parser("chat", help="interactive REPL over the graph")
    c.add_argument("--graph", default=DEFAULT_GRAPH)
    c.set_defaults(func=_chat)

    args = p.parse_args(argv)
    if args.cmd is None:  # bare invocation -> interactive chat
        args.graph = DEFAULT_GRAPH
        return _chat(args)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
