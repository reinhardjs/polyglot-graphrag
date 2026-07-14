"""domains/enterprise/retrieve.py — Enterprise Knowledge Management retrieval.

Uses the generic dense-prose retriever (domains/_default.py) since enterprise
documents (ADRs, runbooks, postmortems) are free-text and best served by
semantic vector search. No custom graph retrieval needed for v1.0.

Registry contract: expose ``retrieve(query, top_k=5) -> list[dict]``.
"""
from __future__ import annotations

from typing import Optional

from domains._default import default_retrieve


def retrieve(query: str, top_k: int = 5, vec: Optional[list] = None) -> list:
    """Semantic search over the enterprise knowledge corpus (Qdrant)."""
    return default_retrieve("enterprise", query, top_k=top_k, vec=vec)
