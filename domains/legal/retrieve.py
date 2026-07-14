"""domains/legal/retrieve.py — Legal & Compliance retrieval (dense prose)."""
from __future__ import annotations

from typing import Optional

from domains._default import default_retrieve


def retrieve(query: str, top_k: int = 5, vec: Optional[list] = None) -> list:
    """Semantic search over the legal/compliance corpus (Qdrant)."""
    return default_retrieve("legal", query, top_k=top_k, vec=vec)
