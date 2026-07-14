"""domains/clinical_prose/retrieve.py — semantic retrieval over clinical prose.

This retriever answers symptom queries by SEMANTIC similarity against the
free-text disease descriptions ingested by ``ingest.py`` (Wikipedia medicine
articles, etc.) into the ``clinical_prose`` Qdrant collection. Unlike the
SNOMED term-match, this matches a query like "fever, cough, rash" to a Measles
article even though the SNOMED concept name contains none of those words.

Registry contract: expose ``retrieve(query, top_k=5) -> list[dict]`` with the
standard {"text","doc_id","doc_type","chunk_idx"} shape.

Query embedding is delegated to the resident GPU daemon (serve_gpu.py
/embed_query) so we never load a second Jina-v3 copy.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import requests

import config as C

COLLECTION = "clinical_prose"


def _embed_query(text: str) -> list:
    """Embed a query via the resident daemon (Jina-v3, retrieval.query task)."""
    resp = requests.post(C.DAEMON_EMBED_QUERY, json={"text": text}, timeout=60)
    resp.raise_for_status()
    return resp.json()["vector"]


def _qdrant_search(vec: list, query_text: str, top_k: int) -> list:
    """Search the clinical_prose collection (dense vector similarity)."""
    from qdrant_client import QdrantClient
    qc = QdrantClient(url=C.QDRANT_URL, prefer_grpc=False)
    if not qc.collection_exists(COLLECTION):
        return []
    hits = qc.query_points(
        COLLECTION,
        query=vec,
        with_payload=True,
        limit=top_k,
    ).points
    out = []
    for h in hits:
        p = h.payload or {}
        out.append({
            "text": p.get("text", ""),
            "doc_id": p.get("doc_id", ""),
            "doc_type": p.get("doc_type", "clinical_prose"),
            "chunk_idx": p.get("chunk_idx", -1),
            "score": float(h.score),
            "title": (p.get("metadata") or {}).get("title", ""),
        })
    return out


def retrieve(query: str, top_k: int = 5) -> list:
    """Registry entry point: semantic search over clinical-prose corpus."""
    try:
        vec = _embed_query(query)
    except Exception as e:
        print(f"[clinical_prose] embed failed: {e}", flush=True)
        return []
    hits = _qdrant_search(vec, query, top_k)
    # Normalise to the registry shape (text only; score/title kept for context).
    out = []
    for h in hits:
        title = h.get("title", "")
        prefix = f"[CLINICAL PROSE] {title}\n" if title else "[CLINICAL PROSE]\n"
        out.append({
            "text": prefix + h["text"],
            "doc_id": h["doc_id"],
            "doc_type": h["doc_type"],
            "chunk_idx": h["chunk_idx"],
        })
    return out


# Convenience for direct debugging / tests.
if __name__ == "__main__":
    import json
    q = sys.argv[1] if len(sys.argv) > 1 else "fever and rash and cough"
    print(json.dumps(retrieve(q, top_k=5), indent=2)[:1500])
