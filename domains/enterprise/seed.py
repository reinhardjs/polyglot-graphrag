"""domains/enterprise/seed.py — idempotent demo seeding for --demo flag.

Seeds 4 synthetic enterprise documents (ADR-021, BUG-204, PR-482,
runbook-checkout) into the `enterprise` Qdrant collection via the resident
GPU daemon's /embed_query endpoint. Idempotent: skips doc_ids that already exist.
"""
from __future__ import annotations

import requests
import config as C
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, Filter, FieldCondition, MatchValue

COLLECTION = "enterprise"
DAEMON_EMBED = C.DAEMON_EMBED_QUERY

_DOCS = [
    {
        "doc_id": "adr-021-payment-service",
        "title": "ADR-021: Payment Service Extraction",
        "text": (
            "ADR-021: Payment Service Extraction. Context: the monolith handled "
            "payments inline, causing cascading failures when the payments provider "
            "was slow. Decision: extract payment-processing into its own service "
            "(payment-service) with a dedicated connection pool and circuit breaker. "
            "Consequences: services depending on payment-service must handle its "
            "outages via the circuit breaker; billing and checkout now depend on "
            "payment-service. Risk: if payment-service is down, checkout is degraded."
        ),
    },
    {
        "doc_id": "bug-204-checkout-deadlock",
        "title": "BUG-204: Checkout Deadlock under Load",
        "text": (
            "BUG-204: Checkout deadlock under high concurrency. Symptoms: checkout "
            "requests hang when payment-service latency spikes. Root cause: synchronous "
            "call chain checkout -> payment-service -> ledger with shared connection "
            "pool exhaustion. Impact: checkout and billing both blocked. Fix: add "
            "timeout + retry with backoff; cap concurrent calls to payment-service."
        ),
    },
    {
        "doc_id": "pr-482-circuit-breaker",
        "title": "PR-482: Circuit Breaker for payment-service",
        "text": (
            "PR-482: Add circuit breaker around payment-service calls. Introduces a "
            "resilience4j circuit breaker so checkout degrades gracefully instead of "
            "blocking when payment-service is unhealthy. Dependencies: checkout now "
            "depends on the breaker config; billing depends on checkout. Related to "
            "BUG-204 and ADR-021."
        ),
    },
    {
        "doc_id": "runbook-checkout-outage",
        "title": "Runbook: Checkout Outage",
        "text": (
            "Runbook: Checkout Outage. First check payment-service health endpoint. "
            "If payment-service is down, the circuit breaker should be OPEN and checkout "
            "returns a cached price with a warning. Verify billing and ledger are not "
            "blocked. Escalate to the payments on-call if the breaker stays open > 5 min. "
            "payment-service, checkout, and billing form the critical path."
        ),
    },
]


def _embed(text: str) -> list:
    resp = requests.post(DAEMON_EMBED, json={"text": text}, timeout=60)
    resp.raise_for_status()
    return resp.json()["vector"]


def seed() -> str:
    """Seed enterprise demo docs idempotently. Returns status string."""
    qc = QdrantClient(url=C.QDRANT_URL, prefer_grpc=False)
    if not qc.collection_exists(COLLECTION):
        qc.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=C.VECTOR_DIM, distance=Distance.COSINE),
        )
    # Find existing doc_ids
    existing = set()
    for doc in _DOCS:
        hits = qc.scroll(
            COLLECTION,
            scroll_filter=Filter(must=[FieldCondition(
                key="doc_id", match=MatchValue(value=doc["doc_id"]))]),
            limit=1, with_payload=False,
        )[0]
        if hits:
            existing.add(doc["doc_id"])
    new_docs = [d for d in _DOCS if d["doc_id"] not in existing]
    if not new_docs:
        return f"enterprise: skipped ({len(_DOCS)} already exist)"
    points = []
    for d in new_docs:
        vec = _embed(d["title"] + "\n" + d["text"])
        points.append(PointStruct(
            id=abs(hash(d["doc_id"])) % (2 ** 63),
            vector=vec,
            payload={
                "doc_id": d["doc_id"],
                "doc_type": "enterprise",
                "chunk_idx": 0,
                "text": d["text"],
                "metadata": {"title": d["title"], "source": "demo"},
            },
        ))
    qc.upsert(COLLECTION, points, wait=True)
    return f"enterprise: seeded {len(new_docs)} docs"
