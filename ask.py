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
os.environ["HF_HOME"] = os.environ.get(
    "HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "hf"))
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
from typing import Dict, List, Optional

# Cached Neo4j driver (recreated connection per call was ~tens of ms overhead
# and, combined with the full 500-node scan below, made neo4j_subgraph the
# dominant /ask latency cost). One driver for the process.
_NEO4J_DRIVER = None
_NEO4J_DRIVER_LOCK = threading.Lock()


def _get_driver():
    global _NEO4J_DRIVER
    if _NEO4J_DRIVER is not None:
        return _NEO4J_DRIVER
    with _NEO4J_DRIVER_LOCK:
        if _NEO4J_DRIVER is None:
            _NEO4J_DRIVER = GraphDatabase.driver(
                C.NEO4J_URI, auth=(C.NEO4J_USER, C.NEO4J_PASSWORD))
    return _NEO4J_DRIVER


def resolve_collections(collection: Optional[str | List[str]]) -> List[str]:
    """Resolve a collection request into a list of Qdrant collection names.

    - None → [QDRANT_COLLECTION_DEFAULT] (engineering_chunks)
    - "legal" → [QDRANT_COLLECTIONS["legal"]] (domain alias)
    - "legal_chunks" → ["legal_chunks"] (direct name, if exists)
    - ["eng", "legal"] → [engineering_chunks, legal_chunks] (cross-domain)
    - "all" → all registered collections in QDRANT_COLLECTIONS
    """
    if collection is None:
        return [C.QDRANT_COLLECTION_DEFAULT]
    if isinstance(collection, str):
        if collection == "all":
            return list(C.QDRANT_COLLECTIONS.values())
        if collection in C.QDRANT_COLLECTIONS:
            return [C.QDRANT_COLLECTIONS[collection]]
        return [collection]
    out = []
    for c in collection:
        if c == "all":
            out.extend(C.QDRANT_COLLECTIONS.values())
        elif c in C.QDRANT_COLLECTIONS:
            out.append(C.QDRANT_COLLECTIONS[c])
        else:
            out.append(c)
    seen = set()
    return [x for x in out if not (x in seen or seen.add(x))]


# ── Sparse term vector helper (consistent hash across storage + query) ───────
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
    # Qdrant REQUIRES strictly-unique, ascending sparse indices. Two distinct
    # tokens can hash to the same 64K bin, and repeated tokens produce the
    # same index twice -> 'must be unique' upsert error on repetitive docs.
    # Aggregate values per index and emit sorted-unique indices.
    agg = {}
    for tok, f in c.items():
        idx = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4],
                                  "little") % _SPARSE_VOCAB
        agg[idx] = agg.get(idx, 0.0) + float(f)
    indices = sorted(agg.keys())
    vals = [agg[i] for i in indices]
    return models.SparseVector(indices=indices, values=vals)

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
    # Sparse leg — only if we have query text to tokenize. The sparse vector
    # is named "text" and only exists on collections that were created with a
    # sparse vector (e.g. clinical_prose). Collections with a plain unnamed
    # dense vector have no sparse "text" vector, so we guard the prefetch: if
    # the query_points call fails on the missing sparse vector, retry dense-only.
    if query_text:
        prefetch.append(models.Prefetch(
            query=_sparse(query_text),
            using="text", limit=top_k * 2,
        ))
    try:
        res = qc.query_points(
            collection,
            query=vec,
            prefetch=prefetch,
            limit=top_k,
            with_payload=True,
        ).points
    except Exception:
        # Retry without the sparse leg (collection has no sparse "text" vector).
        res = qc.query_points(
            collection,
            query=vec,
            prefetch=[models.Prefetch(query=vec, using="", limit=top_k * 2)],
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
    driver = _get_driver()

    label_clause = f":{label}" if label else ""
    ql = set(re.findall(r"\w+", query.lower()))
    with driver.session() as s:
        # KEYWORD ENTRY: match candidate entities SERVER-SIDE by query-token
        # overlap (Cypher CONTAINS), instead of pulling up to 500 nodes and
        # scoring in Python. This turns a full-graph scan + large payload into
        # a targeted index-backed lookup (~10-50ms vs ~300ms on a populated
        # graph). Fall back to the 500-row scan only if NO token matches.
        if ql:
            toks = list(ql)[:8]   # cap candidate clauses
            conds = " OR ".join(
                f"toLower(n.id) CONTAINS $t{i} OR toLower(n.name) CONTAINS $t{i}"
                for i in range(len(toks)))
            params = {f"t{i}": t for i, t in enumerate(toks)}
            rows = s.run(
                f"MATCH (n:Entity{label_clause}) WHERE {conds} "
                f"RETURN n.id AS id, n.name AS name, n.profile AS prof, "
                f"n.source_doc AS source_doc LIMIT 50",
                **params,
            ).data()
        else:
            rows = []
        if not rows:
            # No token overlap — broad fallback (rare; e.g. pure-NL queries).
            rows = s.run(
                f"MATCH (n:Entity{label_clause}) "
                f"RETURN n.id AS id, n.name AS name, n.profile AS prof, "
                f"n.source_doc AS source_doc LIMIT 500"
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
            # Fetch neighbourhood nodes AND the relationships among them so we
            # can rank by structural centrality (Phase 2 pruning).
            rec = s.run(
                "MATCH (n:Entity {id:$e})-[*1..%d]-(m:Entity) "
                "RETURN DISTINCT m.id AS id, m.profile AS prof, "
                "m.source_doc AS source_doc" % hops,
                e=entry,
            ).data()
            edge_rows = s.run(
                "MATCH (n:Entity {id:$e})-[*1..%d]-(m:Entity) "
                "WITH collect(DISTINCT m) AS ms, n AS entry "
                "WITH ms + entry AS ns "
                "UNWIND ns AS a UNWIND ns AS b "
                "MATCH (a)-[r]-(b) WHERE a.id < b.id "
                "RETURN DISTINCT a.id AS src, b.id AS dst" % hops,
                e=entry,
            ).data()
        edges = [(r["src"], r["dst"]) for r in edge_rows
                 if r.get("src") and r.get("dst")]

        # Phase 2: prune the neighbourhood to the Top-N most central nodes.
        strategy = getattr(C, "GRAPH_PRUNE_STRATEGY", "degree")
        top_n = getattr(C, "GRAPH_PRUNE_TOP_N", 10)
        if strategy != "none" and rec:
            try:
                import graph_prune as GP
                rec = GP.prune_records(entry, rec, edges,
                                       strategy=strategy, top_n=top_n)
            except Exception:
                pass  # never let pruning break retrieval

        # Include the entry node's own profile first (with its source_doc)
        e0 = scored[0][1]
        if e0.get("prof"):
            out.append({"text": e0["prof"], "doc_id": e0.get("source_doc", ""),
                        "doc_type": "", "chunk_idx": -1})
        for r in rec:
            if r.get("prof"):
                out.append({"text": r["prof"], "doc_id": r.get("source_doc", ""),
                            "doc_type": "", "chunk_idx": -1})

    # NOTE: do NOT close the driver here — it is process-long-lived (cached via
    # _get_driver) and shared across all /ask calls. Closing it would break
    # every later call in the same process.
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
