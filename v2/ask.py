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
import hashlib
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
        # Deterministic across processes (Python's built-in hash() is randomized
        # per process via PYTHONHASHSEED, which silently breaks sparse search
        # when ingest and query run in different processes).
        idxs.append(int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little") % _SPARSE_VOCAB)
        vals.append(float(f))
    return models.SparseVector(indices=idxs, values=vals)


# ── Embedding ─────────────────────────────────────────────────────────────────
def embed_query(text: str) -> list:
    return requests.post(C.DAEMON_EMBED_QUERY,
                         json={"text": text}, timeout=60).json()["vector"]


# ── Retrieval ─────────────────────────────────────────────────────────────────
def qdrant_search(vec: list, query_text: str = "",
                  top_k: int = C.QDRANT_SEARCH_TOP_K,
                  collection: str = C.COLL_CHUNKS) -> list:
    """Dense + sparse hybrid Qdrant search with RRF fusion.

    `collection` selects the Qdrant collection to search (multi-domain support).
    Defaults to C.COLL_CHUNKS (engineering_chunks).

    Returns a list of dict records:
        {"text": str, "doc_id": str, "doc_type": str, "chunk_idx": int}
    (graph results from neo4j_subgraph are also records with doc_id resolved
    from each entity's source_doc.)
    """
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
        collection,
        query=vec,
        prefetch=prefetch,
        limit=top_k,
        with_payload=True,
    ).points
    out = []
    for p in res:
        if not p.payload:
            continue
        out.append({
            "text": p.payload.get("text", ""),
            "doc_id": p.payload.get("doc_id", ""),
            "doc_type": p.payload.get("doc_type", ""),
            "chunk_idx": p.payload.get("chunk_idx", -1),
        })
    return out


def qdrant_search_multi(vec: list, query_text: str,
                        collections: list,
                        top_k: int = C.QDRANT_SEARCH_TOP_K) -> list:
    """Federated search across multiple collections (cross-domain queries).

    Runs each collection search in parallel threads, merges results. The rerank
    step (in parallel_retrieve / condense) handles fusion downstream.
    """
    results = []
    threads = []
    local = [[] for _ in collections]

    def _worker(idx, coll):
        local[idx].extend(qdrant_search(vec, query_text, top_k, coll))

    for i, coll in enumerate(collections):
        t = threading.Thread(target=_worker, args=(i, coll))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    for r in local:
        results.extend(r)
    return results


def neo4j_subgraph(query: str, hops: int = C.GRAPH_HOPS, label: str = None,
                   entry_strategy: str = "keyword") -> list:
    """Keyword / vector / hybrid entry + k-hop Neo4j subgraph traversal.

    Finds the Entity node to enter the graph from, then traverses its
    neighbourhood. Returns populated profiles (written by ingest.py) as records
    {"text": prof, "doc_id": source_doc, ...} so graph hits resolve to their
    source document for citation (v2.6.0).

    `entry_strategy`:
      "keyword" (default) — score candidates by token overlap with the query.
        Fast (1 Cypher read, no embed). Works for technical IDs like `bug-204`.
      "vector"           — embed the query + each candidate name, pick the
        highest cosine similarity. Needed for natural-language queries
        ("chest pain treatment") where no token overlap exists.
      "hybrid"           — run BOTH in parallel; use keyword entry when its
        overlap >= 2, otherwise fall back to the vector entry. Best recall.

    If `label` is given, restricts the entry-node match to `:Entity:<Label>`
    (v2.6.0 domain isolation).
    """
    driver = GraphDatabase.driver(C.NEO4J_URI,
                                  auth=(C.NEO4J_USER, C.NEO4J_PASSWORD))

    label_clause = f":{label}" if label else ""
    ql = set(re.findall(r"\w+", query.lower()))
    with driver.session() as s:
        rows = s.run(
            f"MATCH (n:Entity{label_clause}) "
            "RETURN n.id AS id, n.name AS name, n.profile AS prof, "
            "n.source_doc AS source_doc "
            "LIMIT 500"
        ).data()

    def _keyword_scored():
        scored = []
        for r in rows:
            if not r.get("id"):
                continue
            tokens = set(re.findall(r"\w+", r["id"].lower())) | \
                     set(re.findall(r"\w+", (r.get("name") or "").lower()))
            overlap = len(ql & tokens)
            if overlap:
                scored.append((overlap, r))
        scored.sort(key=lambda x: -x[0])
        return scored

    def _vector_scored():
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
        return candidates[:5]

    # Resolve the entry node according to strategy.
    if entry_strategy == "vector":
        scored = _vector_scored()
    elif entry_strategy == "hybrid":
        # Run both in parallel threads, then pick.
        import threading
        kw, vec = [], []
        t1 = threading.Thread(target=lambda: kw.extend(_keyword_scored()))
        t2 = threading.Thread(target=lambda: vec.extend(_vector_scored()))
        t1.start(); t2.start(); t1.join(); t2.join()
        if kw and kw[0][0] >= 2:
            scored = kw            # keyword confident enough
        elif vec:
            scored = vec          # fall back to vector
        else:
            scored = kw
    else:  # keyword (default)
        scored = _keyword_scored()
        if not scored:
            # No keyword overlap — auto-fall back to vector so NL queries still work.
            scored = _vector_scored()

    out = []
    if scored:
        entry = scored[0][1]["id"]
        with driver.session() as s:
            rec = s.run(
                "MATCH (n:Entity {id:$e})-[*1..%d]-(m:Entity) "
                "RETURN DISTINCT m.id AS id, m.profile AS prof, "
                "m.source_doc AS source_doc" % hops,
                e=entry,
            ).data()
        # Include the entry node's own profile first (with its source_doc)
        e0 = scored[0][1]
        if e0.get("prof"):
            out.append({"text": e0["prof"], "doc_id": e0.get("source_doc", ""),
                        "doc_type": "", "chunk_idx": -1})
        for r in rec:
            if r.get("prof"):
                out.append({"text": r["prof"], "doc_id": r.get("source_doc", ""),
                            "doc_type": "", "chunk_idx": -1})

    driver.close()
    return out


def parallel_retrieve(vec: list, query: str,
                      collections: list = None, entry_strategy: str = "keyword"):
    """Retrieve from Qdrant (one or more collections) + Neo4j graph in parallel.

    `collections` is a list of Qdrant collection names to search. If None,
    defaults to [C.COLL_CHUNKS] (backward compatible). Cross-domain queries
    pass multiple collections; the rerank step fuses them.
    `entry_strategy` is forwarded to neo4j_subgraph (v2.6.0 REQ-6).
    """
    if collections is None:
        collections = [C.COLL_CHUNKS]
    q_res, g_res = [], []
    t1 = threading.Thread(
        target=lambda: q_res.extend(qdrant_search_multi(vec, query, collections)))
    t2 = threading.Thread(target=lambda: g_res.extend(
        neo4j_subgraph(query, entry_strategy=entry_strategy)))
    t1.start(); t2.start(); t1.join(); t2.join()
    return q_res, g_res


# ── Condense ──────────────────────────────────────────────────────────────────
def condense(query: str, q_res: list, g_res: list) -> list:
    # Normalize records (q_res/g_res may be dicts {text,doc_id,...} or strings).
    def _as_record(x):
        if isinstance(x, dict):
            return x
        return {"text": x, "doc_id": "", "doc_type": "", "chunk_idx": -1}

    def _key(r):
        return (r.get("doc_id", ""), r.get("text", ""))

    recs = [_as_record(x) for x in q_res] + [_as_record(x) for x in g_res]
    seen = set()
    pool = []
    for r in recs:
        k = _key(r)
        if k in seen:
            continue
        seen.add(k)
        pool.append(r)
    if not pool:
        return []
    texts = [r["text"] for r in pool]
    ranked = requests.post(C.DAEMON_RERANK,
                           json={"query": query, "docs": texts},
                           timeout=300).json()["ranked"]
    top = sorted(ranked, key=lambda x: x[1], reverse=True)[:C.RERANK_TOP_K]
    return [pool[i]["text"] for i, _ in top]


# ── Synthesize ────────────────────────────────────────────────────────────────
def synthesize(query: str, contexts: list, profile: dict = None) -> str:
    # Prompt built by shared prompts.py (v2.6.0 REQ-5) — single source of truth
    # reused by retrieve_json.py. Domain-aware via profile[synthesis].
    import prompts as P
    prompt = P.build_synthesis_prompt(query, contexts, profile)
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
