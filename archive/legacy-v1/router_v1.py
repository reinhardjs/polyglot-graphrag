"""
router.py — Semantic caching & zero-shot routing.

Two stages before any LLM token is spent:
    1. Semantic cache: embed query → check Qdrant cache_collection at >0.95
       cosine → return cached answer (0 tokens).
    2. Zero-shot routing: embed query with all-MiniLM-L6-v2 → compare against
       VECTOR_INTENTS / GRAPH_INTENTS centroids → pick the path.
"""
from __future__ import annotations
import os
import numpy as np

os.environ.setdefault("HF_HOME", "/mnt/data-970-plus/hf_cache")

from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct, Distance, VectorParams

from config import (
    ROUTER_MODEL, CACHE_COLLECTION, VECTOR_DIM,
    VECTOR_INTENTS, GRAPH_INTENTS,
    get_qdrant, init_collections,
)


# ─────────────────────────────────────────────────────────────
#  ROUTER (MiniLM on CPU)
# ─────────────────────────────────────────────────────────────

class SemanticRouter:
    """Zero-shot intent router using all-MiniLM-L6-v2 centroid comparison."""

    def __init__(self, model_name: str = ROUTER_MODEL):
        self.model = SentenceTransformer(model_name, device="cpu")
        self.vec_centroid = self._centroid(VECTOR_INTENTS)
        self.graph_centroid = self._centroid(GRAPH_INTENTS)

    @staticmethod
    def _centroid(texts: list[str]) -> np.ndarray:
        v = SentenceTransformer(ROUTER_MODEL, device="cpu").encode(
            texts, normalize_embeddings=True, convert_to_numpy=True)
        return v.mean(axis=0)

    def route(self, query: str) -> tuple[str, float]:
        """Return ('vector'|'graph', confidence_score)."""
        q = self.model.encode([query], normalize_embeddings=True,
                              convert_to_numpy=True)[0]
        s_vec = float(np.dot(q, self.vec_centroid))
        s_graph = float(np.dot(q, self.graph_centroid))
        return ("vector", s_vec) if s_vec >= s_graph else ("graph", s_graph)


# ─────────────────────────────────────────────────────────────
#  SEMANTIC CACHE (Qdrant-backed)
# ─────────────────────────────────────────────────────────────

def cache_lookup(query_vec: list[float], threshold: float = 0.95) -> str | None:
    """Check cache. Returns cached answer string or None."""
    client = get_qdrant()
    if not client.collection_exists(CACHE_COLLECTION):
        return None
    hits = client.query_points(
        collection_name=CACHE_COLLECTION, query=query_vec,
        limit=1, score_threshold=threshold).points
    if hits:
        return hits[0].payload.get("answer")
    return None


def cache_store(query: str, query_vec: list[float], answer: str):
    """Store query+answer in the cache collection."""
    client = get_qdrant()
    if not client.collection_exists(CACHE_COLLECTION):
        return
    pid = abs(hash(query)) % (2**63)
    client.upsert(collection_name=CACHE_COLLECTION, points=[
        PointStruct(id=pid, vector=query_vec,
                    payload={"query": query, "answer": answer}),
    ])
