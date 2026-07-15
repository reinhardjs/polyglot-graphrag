"""domains/enterprise/seed.py — idempotent self-docs seeding.

On first startup (or via `serve_gpu.py --demo`) this seeds the system's
OWN documentation (the repo's `docs/` tree) into the `enterprise` domain,
so a brand-new user gets a queryable knowledge base of how the GraphRAG
system works — with zero external corpus required.

Uses the resident GPU daemon's `/ingest` endpoint (same pipeline as a
real corpus ingest: embed + rerank + optional graph extraction). Idempotent
via content checksum; tagged `source=self-docs` so it is distinguishable
from a user's own ingested corpus. Non-fatal: if the daemon is down or the
collection already holds self-docs, it skips gracefully.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time

import requests

import config as C

COLLECTION = "enterprise"
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCS_ROOT = os.path.join(BASE, "docs")
SOURCE = "self-docs"
INGEST_API = C.DAEMON_URL + "/ingest"
STATUS_API = C.DAEMON_URL + "/ingest/status"


def _collect_md(root: str):
    out = []
    for dp, _, fnames in os.walk(root):
        for fn in sorted(fnames):
            if fn.lower().endswith(".md"):
                out.append(os.path.join(dp, fn))
    return out


def _already_seeded(qc) -> bool:
    """True if the enterprise collection already holds self-docs chunks."""
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        hits = qc.scroll(
            COLLECTION,
            scroll_filter=Filter(must=[FieldCondition(
                key="metadata.source", match=MatchValue(value=SOURCE))]),
            limit=1, with_payload=False,
        )[0]
        return bool(hits)
    except Exception:
        return False


def seed(domain: str = COLLECTION, docs_root: str = DOCS_ROOT,
         source: str = SOURCE) -> str:
    """Seed the repo's own docs into `domain`. Returns a status string."""
    if not os.path.isdir(docs_root):
        return f"{domain}: self-docs skip (no {docs_root})"

    # Health check the daemon first (non-fatal).
    try:
        h = requests.get(C.DAEMON_URL + "/health", timeout=5)
        h.raise_for_status()
    except Exception as e:
        return f"{domain}: self-docs skip (daemon not ready: {e})"

    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(url=C.QDRANT_URL, prefer_grpc=False)
    except Exception as e:
        return f"{domain}: self-docs skip (qdrant unavailable: {e})"

    if _already_seeded(qc):
        return f"{domain}: self-docs skipped (already present)"

    files = _collect_md(docs_root)
    if not files:
        return f"{domain}: self-docs skip (no .md in {docs_root})"

    submitted = 0
    skipped = 0
    for path in files:
        rel = os.path.relpath(path, docs_root)
        doc_id = f"{source}::{rel.replace(os.sep, '::')}"
        text = open(path, encoding="utf-8", errors="replace").read()
        checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
        payload = {
            "text": text,
            "doc_id": doc_id,
            "doc_type": "md",
            "domain": domain,
            "extract_graph": False,
            "if_checksum": checksum,
            "metadata": {"source": source, "rel_path": rel},
        }
        try:
            r = requests.post(INGEST_API, json=payload, timeout=120)
            if r.status_code == 304:
                skipped += 1
            elif r.ok:
                submitted += 1
            else:
                print(f"  [self-docs] ingest {rel} -> HTTP {r.status_code}",
                      file=sys.stderr)
        except Exception as e:
            print(f"  [self-docs] ingest {rel} -> {e}", file=sys.stderr)
        time.sleep(0.1)

    return f"{domain}: self-docs seeded {submitted} docs (skipped {skipped})"


if __name__ == "__main__":
    print(seed())
