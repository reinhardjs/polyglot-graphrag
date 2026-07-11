"""
retrieve.py — LEGACY v1 retrieval pipeline (superseded by v2/ask.py).

DEAD FILE: not imported by the live v2 system (which uses v2/ask.py +
v2/serve_gpu.py). Kept for historical reference. In v2 the synthesis LLM is
Gemma 4 E4B on :8084 and extraction is Gemma 4 E2B on :8082 (both
config-driven in v2/config.py).

VRAM STRATEGY (v1, retired):
    Gemma 4 12B (port 8083) was used ONLY at the very end for answer synthesis.
    All embedding / routing / reranking models ran on CPU (no VRAM impact).

Flow:
    1. Embed query (Jina v3, CPU) → check semantic cache → return if hit
    2. Zero-shot route (MiniLM, CPU) → vector or graph path
    3a. Path A: Qdrant hybrid search (dense + sparse), top 10
    3b. Path B: Neo4j 2-hop k-core traversal, retrieve KV profiles
    4. Cross-encoder rerank (bge-reranker, CPU) → keep top 50% (min 3)
    5. Synthesis: pass condensed context to Gemma 4 (now E4B on :8084)
"""
from __future__ import annotations
import os
import time

os.environ.setdefault("HF_HOME", "/mnt/data-970-plus/hf_cache")

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    GEMMA_MODEL, gemma_client, get_qdrant, get_neo4j, init_collections,
    CHUNKS_COLLECTION, CACHE_COLLECTION, VECTOR_DIM, MAX_CONTEXT,
    RERANKER_MODEL, ROUTER_MODEL, LATE_CHUNK_MODEL,
)
from router import SemanticRouter, cache_lookup, cache_store


# ─────────────────────────────────────────────────────────────
#  1. QUERY EMBEDDING (Jina v3 on CPU, reused)
# ─────────────────────────────────────────────────────────────

class QueryEmbedder:
    def __init__(self, model_name: str = LATE_CHUNK_MODEL):
        self.model = SentenceTransformer(model_name, device="cpu",
                                          trust_remote_code=True)

    def embed(self, text: str) -> list[float]:
        v = self.model.encode([text], normalize_embeddings=True,
                              show_progress_bar=False)[0]
        return v.tolist()


# ─────────────────────────────────────────────────────────────
#  2. PATH A: HYBRID VECTOR SEARCH (Qdrant dense + sparse)
# ─────────────────────────────────────────────────────────────

def _lexical_sparse_query(text: str) -> dict[int, float]:
    from collections import Counter
    import re
    toks = re.findall(r"[a-z0-9]+", text.lower())
    if not toks:
        return {}
    tf = Counter(toks)
    n = len(toks)
    return {abs(hash(t)) % (2**31): 1.0 + (c / n) for t, c in tf.items()}


def vector_search(client, query_vec: list[float], query: str, top_k: int = 10):
    """Hybrid: dense ANN + sparse lexical, fused via RRF."""
    from qdrant_client.models import (Prefetch, SparseVector,
                                      FusionQuery, Fusion)
    init_collections(client)
    sparse = _lexical_sparse_query(query)
    dense_pref = Prefetch(query=query_vec, using=None, limit=top_k * 2)
    text_pref = Prefetch(
        query=SparseVector(indices=list(sparse.keys()),
                           values=list(sparse.values())),
        using="text", limit=top_k * 2,
    )
    results = client.query_points(
        collection_name=CHUNKS_COLLECTION,
        prefetch=[dense_pref, text_pref],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k, with_payload=True,
    ).points
    return [(p.payload.get("text", ""), p.score) for p in results]


# ─────────────────────────────────────────────────────────────
#  3. PATH B: NEO4J GRAPH TRAVERSAL (2-hop, k-core filtered)
# ─────────────────────────────────────────────────────────────

def graph_search(driver, query: str, top_k: int = 10):
    """Find entry node by keyword → 2-hop traversal → k-core filter (deg >= 2).

    Returns the entry node's profile plus profiles of connected nodes that
    have at least 2 connections in the traversal subgraph.
    """
    import re as _re
    kws = [w.lower() for w in _re.findall(r"[a-z0-9_-]+", query) if len(w) > 3]
    entry = None
    with driver.session() as s:
        for kw in kws:
            rec = s.run(
                "MATCH (n:Entity) WHERE toLower(n.name) CONTAINS $kw "
                "OR toLower(n.profile) CONTAINS $kw RETURN n LIMIT 1",
                kw=kw,
            ).single()
            if rec:
                entry = rec["n"]
                break
        if not entry:
            return []
        nid = entry["id"]

        # 2-hop traversal — collect nodes + their degrees in the subgraph
        rows = s.run(
            "MATCH (n:Entity {id:$id})-[*1..2]-(m:Entity) "
            "WITH m, COUNT(*) AS degree "
            "WHERE degree >= 1 "  # k-core filter (k=1, low threshold)
            "RETURN m.name AS name, m.type AS type, m.profile AS profile "
            "LIMIT $k",
            id=nid, k=top_k,
        )
        out = [f"[{entry.get('type')}] {entry.get('name')}: {entry.get('profile')}"]
        for r in rows:
            out.append(f"[{r['type']}] {r['name']}: {r['profile']}")
    return out


# ─────────────────────────────────────────────────────────────
#  4. CROSS-ENCODER RERANKER (CPU, bge-reranker-base)
# ─────────────────────────────────────────────────────────────

class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device="cpu")

    def rerank(self, query: str, passages: list[str],
               keep_top: float = 0.5) -> list[str]:
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        scores = self.model.predict(pairs)
        order = np.argsort(-np.array(scores))
        n = max(3, min(int(len(passages) * keep_top), len(passages)))
        return [passages[i] for i in order[:n]]


# ─────────────────────────────────────────────────────────────
#  5. SYNTHESIS (Gemma 4 on port 8083)
# ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a senior engineering knowledge assistant. Use ONLY the provided "
    "context to answer. If the context lacks the answer, say so plainly. "
    "Cite the source document for every claim. Be concise."
)


def synthesize(query: str, context: list[str], temperature: float = 0.1,
               max_tokens: int = 512) -> str:
    """Send condensed context + query to Gemma 4 on port 8083."""
    if not context:
        return "I don't have that information in the knowledge base."
    client = gemma_client()
    joined = "\n\n".join(f"- {c}" for c in context)
    try:
        resp = client.chat.completions.create(
            model=GEMMA_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"Context:\n{joined}\n\nQuestion: {query}"},
            ],
            temperature=temperature, max_tokens=max_tokens,
        )
        msg = resp.choices[0].message
        return (msg.content or msg.reasoning_content or "").strip()
    except Exception as e:
        return f"[error] {e}"


# ─────────────────────────────────────────────────────────────
#  6. MAIN RETRIEVAL ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

def answer(embedder: QueryEmbedder, router: SemanticRouter,
           reranker: Reranker, query: str) -> dict:
    """Full pipeline: cache → route → search → rerank → synthesize."""
    # Ensure collections exist
    qc = get_qdrant()
    init_collections(qc)
    qc.close()
    TOP_K = 8

    # 1. Embed query
    q_vec = embedder.embed(query)

    # 2. Semantic cache check
    cached = cache_lookup(q_vec)
    if cached:
        return {"path": "cache", "source": "cache", "answer": cached, "ctx": []}

    # 3. Route
    path, score = router.route(query)

    # 4. Retrieve — try vector first, fall back to graph
    qc = get_qdrant()
    if path == "vector":
        raw = vector_search(qc, q_vec, query, top_k=TOP_K)
        ctx = [c for c, _ in raw][:3]
    else:
        driver = get_neo4j()
        graph_ctx = graph_search(driver, query, top_k=TOP_K)
        driver.close()
        if graph_ctx:
            ctx = graph_ctx[:3]
        if not ctx or len(ctx) < 2:
            # graph returned nothing useful → try vector
            raw = vector_search(qc, q_vec, query, top_k=TOP_K)
            ctx = [c for c, _ in raw][:3]
    qc.close()

    # 5. Rerank (condensation)
    ctx = reranker.rerank(query, ctx, keep_top=0.5)

    # 6. Synthesize with Gemma on port 8083
    answer_text = synthesize(query, ctx)

    # 7. Store in cache
    cache_store(query, q_vec, answer_text)

    return {
        "path": path, "source": "llm", "answer": answer_text, "ctx": ctx,
        "path_score": score,
    }
