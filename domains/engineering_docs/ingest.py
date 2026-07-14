"""domains/engineering_docs/ingest.py — REAL engineering companion ingest.

Ingests the repo's `docs/` tree (architecture, guides, roadmaps) into the
`engineering_docs` Qdrant collection. This is the corpus the `engineering`
domain's COMPANION signal searches — distinct from `engineering_chunks`
(the primary, ingested by sync_docs.py). We chunk at a LARGER fixed size
(1024) than the primary's sentence strategy so longer narrative passages
survive intact, giving the semantic retriever purchase on architecture/
design questions the structured `Engineering` graph entities can't answer.

Embedding uses the resident daemon's /embed_late contract (same as
domains/clinical_prose/ingest.py):
    POST /embed_late {"text": <str>, "doc_id": <id>, "strategy": "fixed",
                       "chunk_size": N, "overlap": M}
    -> {"chunks": [{"text": ..., "vector": [...]}, ...]}

Run:  POST /ingest_domain {"domain":"engineering_docs"}
   or: python -m domains.engineering_docs.ingest
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import hashlib

import config as C
import requests

COLLECTION = "engineering_docs"
_DOCS_ROOT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "docs")
_EMBED_LATE = C.DAEMON_EMBED_LATE
_CHUNK_SIZE = 1024
_CHUNK_OVERLAP = 128


def _sparse(text: str):
    """Consistent-hash sparse leg (mirrors ask._sparse) — DEDUPED.

    Qdrant REQUIRES strictly-unique, ascending sparse indices. Two distinct
    tokens can hash to the same 64K bin, and repeated tokens produce the
    same index twice → 'must be unique' upsert error on repetitive docs.
    We aggregate values per index and emit sorted-unique indices.
    """
    from qdrant_client import models
    import re
    from collections import Counter
    import hashlib as _h
    toks = re.findall(r"\w+", text.lower())
    agg = {}
    for tok, f in Counter(toks).items():
        idx = int.from_bytes(_h.md5(tok.encode()).digest()[:4], "little") % 65536
        agg[idx] = agg.get(idx, 0.0) + float(f)
    indices = sorted(agg.keys())
    values = [agg[i] for i in indices]
    return models.SparseVector(indices=indices, values=values)


def _iter_docs(root: str):
    """Yield (rel_path, text) for every .md under root."""
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            if not fn.endswith(".md"):
                continue
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, root)
            try:
                with open(full, encoding="utf-8") as f:
                    yield rel, f.read()
            except Exception as e:
                print(f"[engineering_docs] skip {rel}: {e}", flush=True)


def _embed_chunks(text: str, doc_id: str) -> list:
    resp = requests.post(_EMBED_LATE, json={
        "text": text, "doc_id": doc_id, "strategy": "fixed",
        "chunk_size": _CHUNK_SIZE, "overlap": _CHUNK_OVERLAP,
    }, timeout=300)
    resp.raise_for_status()
    return resp.json().get("chunks", [])


def ingest(source=None, **kw):
    """Ingest docs/ into engineering_docs. Returns count ingested."""
    from qdrant_client import QdrantClient, models
    docs = list(_iter_docs(_DOCS_ROOT)) if not source else source
    if not docs:
        print(f"[engineering_docs] no .md found under {_DOCS_ROOT}", flush=True)
        return 0
    qc = QdrantClient(url=C.QDRANT_URL, prefer_grpc=False)
    if not qc.collection_exists(COLLECTION):
        qc.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(
                size=C.VECTOR_DIM, distance=models.Distance.COSINE),
            sparse_vectors_config={"text": models.SparseVectorParams()})
        print(f"[engineering_docs] created collection '{COLLECTION}'", flush=True)
    points = []
    next_id = 0
    for rel, text in docs:
        doc_id = "docs:" + hashlib.md5(rel.encode()).hexdigest()[:12]
        chunks = _embed_chunks(text, doc_id)
        for ci, ch in enumerate(chunks):
            payload = {
                "text": ch.get("text", text),
                "doc_id": doc_id,
                "doc_type": "engineering_docs",
                "chunk_idx": ci,
                "metadata": {"title": rel, "source": rel},
            }
            points.append(models.PointStruct(
                id=next_id,
                vector={"": ch["vector"], "text": _sparse(payload["text"])},
                payload=payload))
            next_id += 1
    if points:
        qc.upsert(COLLECTION, points)
    print(f"[engineering_docs] ingested {len(points)} chunks from "
          f"{len(docs)} docs -> {COLLECTION}")
    return len(points)


if __name__ == "__main__":
    ingest()
