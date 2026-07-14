"""
domains/clinical_prose/retrieve.py — semantic retrieval over clinical prose.

This retriever answers symptom queries by SEMANTIC similarity against the
free-text disease descriptions ingested by ``ingest.py`` (Wikipedia medicine
articles, etc.) into the ``clinical_prose`` Qdrant collection. Unlike the
SNOMED term-match, this matches a query like "fever, cough, rash" to a Measles
article even though the SNOMED concept name contains none of those words.

Registry contract: expose ``retrieve(query, top_k=5, vec=None) -> list[dict]``
with the standard {"text","doc_id","doc_type","chunk_idx"} shape.

Query embedding is delegated to the resident GPU daemon (serve_gpu.py
/embed_query) ONLY when vec is not already provided by the caller (the /ask
fusion passes the precomputed vec to avoid a second Jina-v3 inference pass).
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import requests

import config as C

COLLECTION = "clinical_prose"

# Module-level Qdrant client singleton (no per-call construction).
_QC = None
_QC_LOCK = threading.Lock()


def _qc():
    global _QC
    if _QC is None:
        with _QC_LOCK:
            if _QC is None:
                from qdrant_client import QdrantClient
                _QC = QdrantClient(url=C.QDRANT_URL, prefer_grpc=False)
    return _QC


def _embed_query(text: str) -> list:
    """Embed a query via the resident daemon (Jina-v3, retrieval.query task)."""
    resp = requests.post(C.DAEMON_EMBED_QUERY, json={"text": text}, timeout=60)
    resp.raise_for_status()
    return resp.json()["vector"]


def _qdrant_search(vec: list, query_text: str, top_k: int) -> list:
    """Search the clinical_prose collection (dense vector similarity)."""
    qc = _qc()
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
            "concept_id": (p.get("metadata") or {}).get("concept_id"),
        })
    return out


def retrieve(query: str, top_k: int = 5, vec: Optional[list] = None) -> list:
    """Registry entry point: semantic search over clinical-prose corpus.

    If ``vec`` (precomputed query embedding) is provided, skip the HTTP embed
    round-trip to the daemon and reuse it directly.
    """
    try:
        if vec is None:
            vec = _embed_query(query)
    except Exception as e:
        print(f"[clinical_prose] embed failed: {e}", flush=True)
        return []
    hits = _qdrant_search(vec, query, top_k)
    out = []
    for h in hits:
        title = h.get("title", "")
        prefix = f"[CLINICAL PROSE] {title}\n" if title else "[CLINICAL PROSE]\n"
        out.append({
            "text": prefix + h["text"],
            "doc_id": h["doc_id"],
            "doc_type": h["doc_type"],
            "chunk_idx": h["chunk_idx"],
            "concept_id": h.get("concept_id"),
        })
    return out


# Convenience for direct debugging / tests.
if __name__ == "__main__":
    import json
    q = sys.argv[1] if len(sys.argv) > 1 else "fever and rash and cough"
    print(json.dumps(retrieve(q, top_k=5), indent=2)[:1500])
