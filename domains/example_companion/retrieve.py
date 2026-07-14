"""domains/example_companion/retrieve.py — TEMPLATE companion corpus.

COPY THIS FOLDER to add a 2nd semantic corpus to ANY domain. A "companion"
is just another domain that a primary domain lists under `companions:` in
domain_config.yaml. The federated factory runs it as its own retrieval
STRATEGY in parallel and tags every record with `_signal=<domain name>`,
so the cross-signal (dual-evidence) boost works WITHOUT any daemon change.

This template is a self-contained semantic retriever over a Qdrant collection
named `example_companion`. It embeds the query via the resident GPU daemon
(/embed_query) so it never loads a second embedding model.

Registry contract (same as every domain):
    retrieve(query: str, top_k: int = 5) -> list[dict]
    each dict: {"text", "doc_id", "doc_type", "chunk_idx"}
Optionally expose retrieve_temporal(presentation: dict, top_k=5) too if your
corpus has graduated presentations.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import requests

import config as C

# The Qdrant collection this companion lives in. Declare it in
# config.QDRANT_COLLECTIONS (or create it ad-hoc) — but the canonical place
# is domain_config.yaml. Keep this name in sync with the YAML block.
COLLECTION = "example_companion"


def _embed_query(text: str) -> list:
    """Embed a query via the resident daemon (delegated, never local model)."""
    resp = requests.post(C.DAEMON_EMBED_QUERY, json={"text": text}, timeout=60)
    resp.raise_for_status()
    return resp.json()["vector"]


def _qdrant_search(vec: list, query_text: str, top_k: int) -> list:
    """Dense + sparse hybrid search over this companion's collection."""
    from qdrant_client import QdrantClient, models
    qc = QdrantClient(url=C.QDRANT_URL, prefer_grpc=False)
    if not qc.collection_exists(COLLECTION):
        return []
    prefetch = [models.Prefetch(query=vec, using="", limit=top_k * 2)]
    if query_text:
        # Reuse ask._sparse for a consistent sparse leg (kept module-private
        # there, so replicate a tiny inline version if you need it).
        prefetch.append(models.Prefetch(
            query=_sparse(query_text), using="text", limit=top_k * 2))
    hits = qc.query_points(COLLECTION, query=vec, prefetch=prefetch,
                            with_payload=True, limit=top_k).points
    out = []
    for h in hits:
        p = h.payload or {}
        out.append({
            "text": p.get("text", ""),
            "doc_id": p.get("doc_id", ""),
            "doc_type": p.get("doc_type", "example_companion"),
            "chunk_idx": p.get("chunk_idx", -1),
            "score": float(h.score),
        })
    return out


# Consistent-hash sparse leg mirroring ask._sparse (so sparse search
# is identical to how the corpus was ingested). Copy from your ingest step.
import re
import hashlib
from collections import Counter
from qdrant_client import models as _models

_VOCAB = 65536


def _sparse(text: str) -> _models.SparseVector:
    # Qdrant requires unique, ascending sparse indices. Two tokens can hash to
    # the same 64K bin and repeated tokens duplicate an index -> 'must be
    # unique' upsert error. Aggregate values per index, emit sorted-unique.
    toks = re.findall(r"\w+", text.lower())
    agg = {}
    for tok, f in Counter(toks).items():
        idx = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4],
                              "little") % _VOCAB
        agg[idx] = agg.get(idx, 0.0) + float(f)
    indices = sorted(agg.keys())
    return _models.SparseVector(indices=indices, values=[agg[i] for i in indices])


def retrieve(query: str, top_k: int = 5) -> list:
    """Registry entry point: semantic search over the companion corpus."""
    try:
        vec = _embed_query(query)
    except Exception as e:
        print(f"[example_companion] embed failed: {e}", flush=True)
        return []
    hits = _qdrant_search(vec, query, top_k)
    out = []
    for h in hits:
        prefix = "[EXAMPLE COMPANION] "  # signal label for the fusion step
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
    q = sys.argv[1] if len(sys.argv) > 1 else "example query"
    print(json.dumps(retrieve(q, top_k=5), indent=2)[:1500])
