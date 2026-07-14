"""domains/engineering/retrieve.py — PRIMARY retriever for the `engineering` domain.

This folder exists so the PRIMARY corpus has a clear home in ``domains/``
alongside companions (engineering_docs, etc.). The actual logic is the shared
default retriever (Qdrant ``engineering_chunks`` + Neo4j ``Engineering`` graph)
in ``domains/_default.py`` — the same fallback any domain without a custom
retrieve.py gets. We don't redefine it here; we just point at it, so there's
one source of truth.

If you later need engineering-specific retrieval (e.g. a custom rerank or
query expansion), replace this file's ``retrieve`` body — no other code changes.
"""

from domains._default import default_retrieve


def retrieve(query: str, top_k: int = 5) -> list:
    """Return primary-corpus hits (vector + graph), tagged by _signal."""
    return default_retrieve("engineering", query, top_k)
