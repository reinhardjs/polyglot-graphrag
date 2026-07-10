"""
ask.py — CLI retriever implementing Parallel Hybrid Fusion.

NO torch / transformers imports. All model work is delegated to the GPU daemon
(serve_gpu.py FastAPI :8000) or the GPU LLMs (E2B :8082 / E4B :8084).
The CLI itself only does HTTP, threading, and string assembly.

Flow:
  1. Embed query via daemon /embed_query
  2. PARALLEL: Thread-1 Qdrant dense+sparse hybrid top-10
            + Thread-2 Neo4j k-hop subgraph (vector-based entry)
  3. Pool + rerank (daemon /rerank) -> keep top 5
  4. Synthesize with Gemma 4 E4B (streamed), context capped to MAX_TOKENS_CONTEXT
"""
import os
os.environ["HF_HOME"] = "/mnt/data-970-plus/hf_cache"
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import re
import requests
import json
import threading
from collections import Counter
from qdrant_client import QdrantClient, models
from neo4j import GraphDatabase
from openai import OpenAI

import config as C


# ── Sparse term vector helper (consistent hash across storage + query) ────────
_SPARSE_VOCAB = 65536


def _sparse(text: str) -> models.SparseVector:
    """Hash-based sparse vector for CONSISTENT cross-chunk term matching.

    Using enumerate(Counter(toks)) produces per-chunk indices 0..N — different
    vocabularies per chunk, so sparse search was returning noise. Hashing each
    token to the same 64K-bin space ensures "checkout" always maps to the same
    index in every chunk AND in the query.
    """
    toks = re.findall(r"\w+", text.lower())
    c = Counter(toks)
    idxs, vals = [], []
    for tok, f in c.items():
        idxs.append(hash(tok) % _SPARSE_VOCAB)
        vals.append(float(f))
    return models.SparseVector(indices=idxs, values=vals)


# ── Embedding ─────────────────────────────────────────────────────────────────
def embed_query(text: str) -> list:
    return requests.post(C.DAEMON_EMBED_QUERY,
                         json={"text": text}, timeout=60).json()["vector"]


# ── Retrieval ─────────────────────────────────────────────────────────────────
def qdrant_search(vec: list, query_text: str = "",
                  top_k: int = C.QDRANT_SEARCH_TOP_K) -> list:
    """Dense + sparse hybrid Qdrant search with RRF fusion."""
    qc = QdrantClient(url=C.QDRANT_URL, prefer_grpc=False)
    prefetch = [
        # Dense vector leg
        models.Prefetch(query=vec, using="", limit=top_k * 2),
    ]
    # Sparse leg — only if we have query text to tokenize
    if query_text:
        prefetch.append(models.Prefetch(
            query=_sparse(query_text),
            using="text", limit=top_k * 2,
        ))
    res = qc.query_points(
        C.COLL_CHUNKS,
        query=vec,
        prefetch=prefetch,
        limit=top_k,
        with_payload=True,
    ).points
    return [p.payload.get("text", "") for p in res if p.payload]


def neo4j_subgraph(query: str, hops: int = C.GRAPH_HOPS) -> list:
    """Keyword-overlap entry + k-hop Neo4j subgraph traversal.

    Finds the Entity node whose id has the most keyword overlap with the query,
    then traverses its neighbourhood. Returns populated profiles (written by
    ingest.py) for retrieval. Fast — one Cypher query, no embedding calls.
    """
    driver = GraphDatabase.driver(C.NEO4J_URI,
                                  auth=(C.NEO4J_USER, C.NEO4J_PASSWORD))

    ql = set(re.findall(r"\w+", query.lower()))
    with driver.session() as s:
        rows = s.run(
            "MATCH (n:Entity) "
            "RETURN n.id AS id, n.name AS name, n.profile AS prof "
            "LIMIT 500"
        ).data()

    # Score candidates by keyword overlap with query tokens
    scored = []
    for r in rows:
        if not r.get("id"):
            continue
        # Overlap against both the canonical id AND the display name.
        # Use \w+ tokenization so "bug-204" splits into {"bug","204"}
        tokens = set(re.findall(r"\w+", r["id"].lower())) | \
                 set(re.findall(r"\w+", (r.get("name") or "").lower()))
        overlap = len(ql & tokens)
        if overlap:
            scored.append((overlap, r))
    scored.sort(key=lambda x: -x[0])

    if not scored:
        # Fallback: embed query, then cosine-similarity score entity names
        # (slower — only triggered when keyword overlap finds nothing)
        q_vec = embed_query(query)
        import numpy as np
        q_np = np.array(q_vec) / (np.linalg.norm(q_vec) + 1e-8)
        candidates = []
        for r in rows:
            name = r.get("name") or r.get("id") or ""
            try:
                v = np.array(embed_query(name))
                score = float(np.dot(q_np, v / (np.linalg.norm(v) + 1e-8)))
                candidates.append((score, r))
            except Exception:
                continue
        candidates.sort(key=lambda x: -x[0])
        scored = candidates[:5]  # top 5 by vector similarity

    out = []
    if scored:
        entry = scored[0][1]["id"]
        with driver.session() as s:
            rec = s.run(
                "MATCH (n:Entity {id:$e})-[*1..%d]-(m:Entity) "
                "RETURN DISTINCT m.id AS id, m.profile AS prof" % hops,
                e=entry,
            ).data()
        # Include the entry node's own profile first
        if scored[0][1].get("prof"):
            out.append(scored[0][1]["prof"])
        for r in rec:
            if r.get("prof"):
                out.append(r["prof"])

    driver.close()
    return out


def parallel_retrieve(vec: list, query: str):
    q_res, g_res = [], []
    t1 = threading.Thread(target=lambda: q_res.extend(qdrant_search(vec, query)))
    t2 = threading.Thread(target=lambda: g_res.extend(neo4j_subgraph(query)))
    t1.start(); t2.start(); t1.join(); t2.join()
    return q_res, g_res


# ── Condense ──────────────────────────────────────────────────────────────────
def condense(query: str, q_res: list, g_res: list) -> list:
    pool = list(dict.fromkeys(q_res + g_res))  # dedupe, keep order
    if not pool:
        return []
    ranked = requests.post(C.DAEMON_RERANK,
                           json={"query": query, "docs": pool},
                           timeout=300).json()["ranked"]
    top = sorted(ranked, key=lambda x: x[1], reverse=True)[:C.RERANK_TOP_K]
    return [pool[i] for i, _ in top]


# ── Synthesize ────────────────────────────────────────────────────────────────
def synthesize(query: str, contexts: list) -> str:
    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = (
        "Answer the question using ONLY the context. "
        "Cite source numbers. If missing, say so.\n\n"
        f"CONTEXT:\n{ctx}\n\nQUESTION: {query}"
    )[: C.MAX_TOKENS_CONTEXT * 4]
    client = OpenAI(base_url=C.SYNTHESIS_LLM_BASE_URL, api_key=C.SYNTHESIS_LLM_API_KEY)
    stream = client.chat.completions.create(
        model=C.SYNTHESIS_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=C.LLM_MAX_TOKENS_OUT,
        stream=True,
    )
    reasoning_parts = []  # E4B chain-of-thought (debug only)
    answer_parts = []     # clean final answer
    for chunk in stream:
        delta = chunk.choices[0].delta
        # E4B puts reasoning in reasoning_content, clean answer in content
        r = getattr(delta, "reasoning_content", None) or ""
        c = getattr(delta, "content", None) or ""
        if r:
            print(r, end="", flush=True)  # stdout for debugging
            reasoning_parts.append(r)
        if c:
            answer_parts.append(c)
    # Print the clean answer on a new line
    clean = "".join(answer_parts)
    if clean:
        print()  # separator between reasoning trace and answer
        print(clean, flush=True)
    return clean


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "who reported BUG-204?"
    vec = embed_query(q)
    q_res, g_res = parallel_retrieve(vec, q)
    ctx = condense(q, q_res, g_res)
    synthesize(q, ctx)
