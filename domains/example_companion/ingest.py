"""domains/example_companion/ingest.py — TEMPLATE companion ingest.

Ingests sample documents into the `example_companion` Qdrant collection so the
retriever above has something to search. Replace _sample_docs() with your real
source (files, API, DB).

Embedding uses the resident daemon's /embed_late endpoint with the SAME
contract as domains/clinical_prose/ingest.py:
    POST /embed_late {"text": <str>, "doc_id": <id>, "strategy": "fixed",
                       "chunk_size": N, "overlap": M}
    -> {"chunks": [{"text": ..., "vector": [...]}, ...]}
This keeps a single Jina-v3 copy resident (no 2nd model loaded).

Run:  POST /ingest_domain {"domain":"example_companion"}
   or: python -m domains.example_companion.ingest
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import hashlib
import re
from collections import Counter

import config as C
import requests

COLLECTION = "example_companion"
_EMBED_LATE = C.DAEMON_EMBED_LATE
_CHUNK_SIZE = 512
_CHUNK_OVERLAP = 64


def _sparse(text: str):
    """Consistent-hash sparse leg (mirrors ask._sparse) so sparse search
    matches how the corpus was ingested."""
    from qdrant_client import models
    toks = re.findall(r"\w+", text.lower())
    c = Counter(toks)
    idxs, vals = [], []
    for tok, f in c.items():
        idxs.append(int.from_bytes(hashlib.md5(tok.encode()).digest()[:4],
                                  "little") % 65536)
        vals.append(float(f))
    return models.SparseVector(indices=idxs, values=vals)


def _sample_docs():
    """Replace with your real loader. Returns list of {text, doc_id, doc_type}."""
    return [
        {"text": "Example document one: how companion corpora add semantic "
                 "recall that a term-match graph misses.", "doc_id": "ex-1",
         "doc_type": "example_companion"},
        {"text": "Example document two: dual-signal boosting prefers candidates "
                 "confirmed by both the primary and the companion corpus.",
         "doc_id": "ex-2", "doc_type": "example_companion"},
    ]


def _embed_chunks(text: str, doc_id: str) -> list:
    """Embed one doc via /embed_late, returning its chunk list."""
    resp = requests.post(_EMBED_LATE, json={
        "text": text, "doc_id": doc_id, "strategy": "fixed",
        "chunk_size": _CHUNK_SIZE, "overlap": _CHUNK_OVERLAP,
    }, timeout=120)
    resp.raise_for_status()
    return resp.json().get("chunks", [])


def ingest(source=None, **kw):
    """Seed the example_companion collection. Returns count ingested."""
    from qdrant_client import QdrantClient, models
    docs = _sample_docs() if not source else source
    qc = QdrantClient(url=C.QDRANT_URL, prefer_grpc=False)
    if not qc.collection_exists(COLLECTION):
        qc.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(
                size=C.VECTOR_DIM, distance=models.Distance.COSINE),
            sparse_vectors_config={"text": models.SparseVectorParams()})
        print(f"[example_companion] created collection '{COLLECTION}'", flush=True)
    points = []
    next_id = 0
    for d in docs:
        chunks = _embed_chunks(d["text"], d["doc_id"])
        for ci, ch in enumerate(chunks):
            payload = {
                "text": ch.get("text", d["text"]),
                "doc_id": d["doc_id"],
                "doc_type": d.get("doc_type", "example_companion"),
                "chunk_idx": ci,
            }
            points.append(models.PointStruct(
                id=next_id,
                vector={"": ch["vector"], "text": _sparse(payload["text"])},
                payload=payload))
            next_id += 1
    if points:
        qc.upsert(COLLECTION, points)
    print(f"[example_companion] ingested {len(points)} chunks -> {COLLECTION}")
    return len(points)


if __name__ == "__main__":
    ingest()
