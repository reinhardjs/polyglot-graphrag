#!/usr/bin/env python3
"""
main.py — CLI for the GraphRAG engineering knowledge base.

Usage:
    python main.py ingest <file> [--type ADR] [--author alice]
    python main.py ingest-folder <dir>
    python main.py ask "<question>"
    python main.py serve           # interactive REPL (models loaded once)
"""
from __future__ import annotations
import os, sys, argparse

os.environ.setdefault("HF_HOME", "<hf-cache>")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_qdrant, init_collections, get_neo4j
from ingest import JinaLateChunker, ingest_doc
from router import SemanticRouter
from retrieve import QueryEmbedder, answer, Reranker


def cmd_ingest(args):
    init_collections(get_qdrant())
    ingest_doc(args.path, doc_type=args.type, author=args.author)


def cmd_ingest_folder(args):
    from pathlib import Path
    exts = {".txt", ".md", ".markdown", ".pdf"}
    p = Path(args.folder)
    for f in sorted(p.iterdir()):
        if f.suffix.lower() in exts:
            print("-->", f.name)
            cmd_ingest(argparse.Namespace(path=str(f), type=None, author=None))


def cmd_ask(args):
    embedder = QueryEmbedder()
    router = SemanticRouter()
    reranker = Reranker()
    result = answer(embedder, router, reranker, args.query)
    print(f"\n[path: {result['path']}]  (source: {result['source']})")
    print("-" * 60)
    print(result["answer"])
    print("-" * 60)
    if result["ctx"]:
        print("Context used:")
        for c in result["ctx"]:
            print("  •", c[:160])


def cmd_serve(args):
    embedder = QueryEmbedder()
    router = SemanticRouter()
    reranker = Reranker()
    print("GraphRAG ready. Type a question (Ctrl-D exits).")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q:
            continue
        result = answer(embedder, router, reranker, q)
        print(f"[{result['path']}/{result['source']}] {result['answer']}")


def main():
    ap = argparse.ArgumentParser(description="GraphRAG engineering KB")
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("ingest"); p.add_argument("path")
    p.add_argument("--type", default=None); p.add_argument("--author", default=None)
    p.set_defaults(func=cmd_ingest)

    p = sp.add_parser("ingest-folder"); p.add_argument("folder")
    p.set_defaults(func=cmd_ingest_folder)

    p = sp.add_parser("ask"); p.add_argument("query")
    p.set_defaults(func=cmd_ask)

    p = sp.add_parser("serve"); p.set_defaults(func=cmd_serve)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
