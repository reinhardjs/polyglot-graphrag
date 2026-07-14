"""domains/engineering_docs/retrieve.py — REAL engineering companion.

A 2nd semantic corpus for the `engineering` domain: it ingests the repo's
`docs/` tree (architecture, guides, roadmaps — real engineering knowledge)
into its OWN Qdrant collection (`engineering_docs`). The primary `engineering`
domain searches `engineering_chunks` (Qdrant) + the `Engineering` Neo4j
graph. This companion adds a NARRATIVE/PROSE signal: architecture questions
phrased in natural language that the structured graph entities (Microservice,
Bug, PR) don't directly answer, but the docs prose does.

Registry contract: expose `retrieve(query, top_k=5) -> list[dict]` with the
standard {"text","doc_id","doc_type","chunk_idx"} shape. Embed via the
resident GPU daemon (/embed_query) so no 2nd model is loaded. Prefix each
record with `[ENGINEERING DOCS] ` so the `_signal` is visible in context
and the cross-signal dual-evidence boost can fire.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import requests

import config as C

COLLECTION = "engineering_docs"


def _embed_query(text: str) -> list:
    """Embed a query via the resident daemon (Jina-v3, retrieval.query task)."""
    resp = requests.post(C.DAEMON_EMBED_QUERY, json={"text": text}, timeout=60)
    resp.raise_for_status()
    return resp.json()["vector"]


def _qdrant_search(vec: list, query_text: str, top_k: int) -> list:
    """Search the engineering_docs collection (dense + sparse hybrid)."""
    from qdrant_client import QdrantClient, models
    qc = QdrantClient(url=C.QDRANT_URL, prefer_grpc=False)
    if not qc.collection_exists(COLLECTION):
        return []
    prefetch = [models.Prefetch(query=vec, using="", limit=top_k * 2)]
    if query_text:
        prefetch.append(models.Prefetch(
            query=_sparse(query_text), using="text", limit=top_k * 2))
    hits = qc.query_points(
        COLLECTION, query=vec, prefetch=prefetch,
        with_payload=True, limit=top_k).points
    out = []
    for h in hits:
        p = h.payload or {}
        out.append({
            "text": p.get("text", ""),
            "doc_id": p.get("doc_id", ""),
            "doc_type": p.get("doc_type", "engineering_docs"),
            "chunk_idx": p.get("chunk_idx", -1),
            "score": float(h.score),
            "title": (p.get("metadata") or {}).get("title", ""),
        })
    return out


# Consistent-hash sparse leg (mirrors ask._sparse) so sparse search matches
# how the corpus was ingested.
import re
import hashlib
from collections import Counter
from qdrant_client import models as _models

_VOCAB = 65536


def _sparse(text: str) -> _models.SparseVector:
    toks = re.findall(r"\w+", text.lower())
    c = Counter(toks)
    idxs, vals = [], []
    for tok, f in c.items():
        idxs.append(int.from_bytes(hashlib.md5(tok.encode()).digest()[:4],
                                  "little") % _VOCAB)
        vals.append(float(f))
    return _models.SparseVector(indices=idxs, values=vals)


def retrieve(query: str, top_k: int = 5) -> list:
    """Registry entry point: semantic search over the engineering-docs corpus."""
    try:
        vec = _embed_query(query)
    except Exception as e:
        print(f"[engineering_docs] embed failed: {e}", flush=True)
        return []
    hits = _qdrant_search(vec, query, top_k)
    out = []
    for h in hits:
        title = h.get("title", "")
        prefix = f"[ENGINEERING DOCS] {title}\n" if title else \
            "[ENGINEERING DOCS]\n"
        out.append({
            "text": prefix + h["text"],
            "doc_id": h["doc_id"],
            "doc_type": h["doc_type"],
            "chunk_idx": h["chunk_idx"],
        })
    return out


if __name__ == "__main__":
    import json
    q = sys.argv[1] if len(sys.argv) > 1 else \
        "what is the total VRAM budget on the RTX 3060"
    print(json.dumps(retrieve(q, top_k=5), indent=2)[:1600])
