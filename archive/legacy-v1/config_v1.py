"""
config.py — LEGACY v1 config (superseded by v2/config.py).

DEAD FILE: nothing in the live v2 system imports this. The v2 daemons use
v2/config.py where synthesis is Gemma 4 E4B on :8084 and extraction is
Gemma 4 E2B on :8082 (both config-driven). Kept for historical reference.

VRAM STRATEGY (v1, retired): Sequential execution. Only ONE model runs at a time:
  - Ornith 9B (port 8081) for ingestion (graph extraction, KV profiling)
  - Gemma 4 12B (port 8083) for retrieval (final answer generation)
  - Embedding / routing / reranking models run on CPU (no VRAM impact)

MAX_CONTEXT = 32768 for all LLM prompts — keeps VRAM under control.
"""
from __future__ import annotations
import os
from pathlib import Path

BASE = Path("<data-root>")
PROJECT_ROOT = BASE / "rag-system"
os.environ.setdefault("HF_HOME", str(BASE / "hf_cache"))

# --- VRAM limit for all LLM prompts ---
MAX_CONTEXT = 32768  # token limit; keeps VRAM under control

# --- Dual LLM endpoints (both point to Gemma on :8083) ---
# Ingestion and retrieval both use the same Gemma 4 12B instance.
# The "ornith" name is kept for code compatibility.
ORNITH_BASE = "http://localhost:8083/v1"
ORNITH_KEY  = "not-needed"
ORNITH_MODEL = "gemma-4-12B-it-QAT-Q4_0.gguf"

# Gemma 4 12B (Synthesizer) — retrieval only
GEMMA_BASE  = "http://localhost:8083/v1"
GEMMA_KEY   = "not-needed"
GEMMA_MODEL = "gemma-4-12B-it-QAT-Q4_0.gguf"

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
CHUNKS_COLLECTION = "engineering_chunks"
CACHE_COLLECTION  = "query_cache"
VECTOR_DIM = 1024  # jina-embeddings-v3 output dim

# --- Neo4j ---
NEO4J_URI  = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "ragpassword123"

# --- Embedding / routing / reranking (all on CPU) ---
LATE_CHUNK_MODEL = "<hf-cache>/jina-v3-flat"  # jinaai/jina-embeddings-v3 (local flat)
ROUTER_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL   = "BAAI/bge-reranker-base"

# --- Intent definitions for zero-shot routing ---
VECTOR_INTENTS = [
    "what is the exact configuration value for the payment service timeout",
    "who authored pull request 482 and when was it merged",
    "find ticket BUG-204 latency spike root cause",
    "list the API endpoints exposed by the auth microservice",
]
GRAPH_INTENTS = [
    "how does a failure in the payment gateway cascade to the checkout flow",
    "what systems are impacted if the auth service goes down",
    "trace the dependency chain from the orders service to the database",
    "which microservices depend on the deprecated user service",
    "explain the architectural tradeoffs between the two candidate designs",
]


# ---------- Helper: get an OpenAI-compatible client for a given port ----------
def _make_client(api_base: str, api_key: str):
    from openai import OpenAI
    return OpenAI(base_url=api_base, api_key=api_key, timeout=180)


def ornith_client():
    """OpenAI-compatible client for Ornith 9B on port 8081 (ingestion)."""
    return _make_client(ORNITH_BASE, ORNITH_KEY)


def gemma_client():
    """OpenAI-compatible client for Gemma 4 12B on port 8083 (retrieval)."""
    return _make_client(GEMMA_BASE, GEMMA_KEY)


# ---------- Qdrant init ----------
def get_qdrant():
    from qdrant_client import QdrantClient
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def init_collections(client):
    """Ensure chunks + cache collections exist with correct settings."""
    from qdrant_client.models import (Distance, VectorParams, BinaryQuantization,
                                      BinaryQuantizationConfig, SparseVectorParams)

    # --- chunks collection: BinaryQuantized dense + sparse ---
    if not client.collection_exists(CHUNKS_COLLECTION):
        client.create_collection(
            CHUNKS_COLLECTION,
            vectors_config=VectorParams(
                size=VECTOR_DIM, distance=Distance.COSINE,
                quantization_config=BinaryQuantization(
                    binary=BinaryQuantizationConfig())),
            sparse_vectors_config={"text": SparseVectorParams()},
        )

    # --- cache collection: dense only, no quantization ---
    if not client.collection_exists(CACHE_COLLECTION):
        client.create_collection(
            CACHE_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )


# ---------- Neo4j init ----------
def get_neo4j():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
