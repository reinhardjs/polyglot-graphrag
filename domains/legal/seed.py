"""domains/legal/seed.py — idempotent GDPR/compliance demo seeding for --demo.

Seeds synthetic (clearly-labelled) compliance documents into the `legal`
Qdrant collection via the resident GPU daemon's /embed_query endpoint.
Idempotent: skips doc_ids that already exist.
"""
from __future__ import annotations

import requests
import config as C
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, Filter, FieldCondition, MatchValue

COLLECTION = "legal"
DAEMON_EMBED = C.DAEMON_EMBED_QUERY

# NOTE: synthetic demonstration text only — not legal advice.
_DOCS = [
    {
        "doc_id": "gdpr-cross-border-transfer",
        "title": "GDPR: Cross-Border Data Transfers",
        "text": (
            "GDPR Article 44-49: cross-border data transfers outside the EEA require "
            "an adequacy decision or appropriate safeguards (Standard Contractual "
            "Clauses, Binding Corporate Rules). Data subjects must be informed. "
            "Transfers to the US under the EU-US Data Privacy Framework need DPF "
            "certification. A transfer impact assessment is required when the "
            "destination country has surveillance laws that may undermine the SCCs."
        ),
    },
    {
        "doc_id": "gdpr-breach-notification",
        "title": "GDPR: Breach Notification (72h)",
        "text": (
            "GDPR Article 33: a personal data breach must be notified to the "
            "supervisory authority within 72 hours of becoming aware, unless "
            "unlikely to result in risk to natural persons. Article 34: data subjects "
            "must be notified without undue delay when the breach is high risk. The "
            "Data Protection Officer coordinates breach response. Document the breach "
            "register entry regardless of notification threshold."
        ),
    },
    {
        "doc_id": "soc2-access-control",
        "title": "SOC 2: Access Control (CC6)",
        "text": (
            "SOC 2 Common Criteria CC6: logical access. Implement least-privilege "
            "role-based access control, periodic access reviews, MFA for privileged "
            "accounts, and deprovisioning within 24h of termination. Segregation of "
            "duties must separate production access from code deployment. Audit logs "
            "must capture who accessed what and when."
        ),
    },
    {
        "doc_id": "hipaa-phi-safeguards",
        "title": "HIPAA: PHI Safeguards",
        "text": (
            "HIPAA Security Rule: protect Protected Health Information (PHI) with "
            "administrative, physical, and technical safeguards. Encrypt PHI at rest "
            "and in transit, apply minimum-necessary access, and execute Business "
            "Associate Agreements (BAAs) with vendors touching PHI. Breach of unencrypted "
            "PHI triggers notification rules analogous to GDPR."
        ),
    },
]


def _embed(text: str) -> list:
    resp = requests.post(DAEMON_EMBED, json={"text": text}, timeout=60)
    resp.raise_for_status()
    return resp.json()["vector"]


def seed() -> str:
    qc = QdrantClient(url=C.QDRANT_URL, prefer_grpc=False)
    if not qc.collection_exists(COLLECTION):
        qc.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=C.VECTOR_DIM, distance=Distance.COSINE),
        )
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
        return f"legal: skipped ({len(_DOCS)} already exist)"
    points = []
    for d in new_docs:
        vec = _embed(d["title"] + "\n" + d["text"])
        points.append(PointStruct(
            id=abs(hash(d["doc_id"])) % (2 ** 63),
            vector=vec,
            payload={
                "doc_id": d["doc_id"],
                "doc_type": "legal",
                "chunk_idx": 0,
                "text": d["text"],
                "metadata": {"title": d["title"], "source": "demo"},
            },
        ))
    qc.upsert(COLLECTION, points, wait=True)
    return f"legal: seeded {len(new_docs)} docs"
