"""domains/_default.py — default (generic) retriever + ingestor for ANY domain.

This is the fallback used when a domain has NO custom ``domains/<name>/retrieve.py``
or ``ingest.py``. It is also the single source of truth for the PRIMARY path
(domains like ``engineering``): search the domain's Qdrant ``collection``
(vector) + optional Neo4j ``neo4j_label`` (graph), and ingest via the generic
``ingest.py::ingest_text`` used by ``sync_docs.py``.

Why a default?
  A domain's primary corpus needs no custom code — it's just "embed + search
  my collection + walk my graph". Only COMPANIONS (different source, different
  chunking) or special domains (e.g. snomed's term-match) need custom plugins.
  So the primary lives in ``domains/`` for uniformity, but uses this default
  unless you override it with a custom file.

Records returned by ``default_retrieve`` carry a ``_sig`` field (``<domain>``
for vector hits, ``<domain>:graph`` for graph hits) so ``federated.assemble``
can tag + fuse them generically alongside companion signals.
"""

from __future__ import annotations

from typing import Optional

from domain_loader import get_domain


def _as_record(x) -> dict:
    if isinstance(x, dict):
        return x
    return {"text": x, "doc_id": "", "doc_type": "", "chunk_idx": -1}


def default_retrieve(name: str, query: str, top_k: int = 5,
                     vec: Optional[list] = None) -> list:
    """Retrieve from a domain's primary corpus: Qdrant collection + Neo4j graph.

    Used as the retriever for the PRIMARY corpus of any domain (and as the
    fallback when ``domains/<name>/retrieve.py`` is absent).

    ``vec`` is the precomputed query embedding (from /ask). If provided, it is
    reused; otherwise the query is embedded here. Reusing avoids a second embed
    on the primary path (companions still self-embed, as before).

    The effective top_k is taken from the domain profile (default 5 if not
    configured). Callers can override by passing top_k explicitly.
    """
    profile = get_domain(name)
    collection = profile.get("collection")
    neo4j_label = profile.get("neo4j_label")
    entry_strategy = profile.get("entry_strategy", "keyword")
    # Use domain-configured top_k unless caller explicitly overrides
    domain_top_k = profile.get("top_k", top_k)

    import ask as rag
    from ask import resolve_collections, embed_query

    if vec is None:
        vec = embed_query(query)
    out = []
    if collection:
        collections = resolve_collections(collection)
        for r in rag.qdrant_search_multi(vec, query, collections, top_k=domain_top_k):
            rec = _as_record(r)
            rec["_sig"] = name
            out.append(rec)
    if neo4j_label:
        graph_results = rag.neo4j_subgraph(
                query, label=neo4j_label, entry_strategy=entry_strategy)
        # Limit graph results to a small fixed number (5) to avoid
        # overwhelming Qdrant results for dense_prose domains. Graph edges
        # are supplementary evidence, not the primary retrieval signal.
        for r in graph_results[:5]:
            rec = _as_record(r)
            rec["_sig"] = name + ":graph"
            out.append(rec)
    return out


def default_ingest(name: str, source=None, **kw) -> int:
    """Generic ingest for a domain's primary corpus.

    Mirrors what ``sync_docs.py`` / ``POST /ingest`` does: embed + optional
    graph extraction into the domain's collection. ``source`` is an optional
    list of ``(rel_path, text)``; if omitted, this is a no-op (the primary is
    normally ingested by ``sync_docs.py``, not ``POST /ingest_domain``).
    """
    import ingest
    if not source:
        return 0
    total = 0
    for rel, text in source:
        ingest.ingest_text(text, doc_id=rel, domain=name, **kw)
        total += 1
    return total