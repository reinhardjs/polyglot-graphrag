#!/usr/bin/env python3
"""
ask.py — Full RAG pipeline (retrieval + optional synthesis) exposed as a library
and a CLI. The daemon (serve_gpu.py) imports and calls these functions so
business logic lives in ONE place, not duplicated across process boundaries.

This module is the SINGLE SOURCE OF TRUTH for:
  - Qdrant vector search (multi-collection, server-side)
  - Neo4j graph traversal (k-hop, with entry strategies: keyword/vector/hybrid)
  - Rerank (BGE reranker via daemon /rerank endpoint)
  - Synthesis (E2B LLM via llama-server OpenAI-compatible endpoint)

v2.7: domain-independent retrieval (federated.py) + CRAG (crag_pipeline.py) are
delegated to; this file only contains the primitive ops they compose.
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import threading
from typing import Optional

import config as C
import numpy as np
import requests
from neo4j import GraphDatabase
from openai import OpenAI


# ── Global singletons ────────────────────────────────────────────────────────
_jina_lock = threading.Lock()
_jina = None

# Neo4j driver singleton (module-level; not per-call)
_DRIVER = None
_DRIVER_LOCK = threading.Lock()


def _get_driver():
    """Module-level Neo4j driver singleton (thread-safe lazy init)."""
    global _DRIVER
    if _DRIVER is None:
        with _DRIVER_LOCK:
            if _DRIVER is None:
                _DRIVER = GraphDatabase.driver(
                    C.NEO4J_URI, auth=(C.NEO4J_USER, C.NEO4J_PASSWORD))
    return _DRIVER


def _get_jina():
    global _jina
    if _jina is None:
        with _jina_lock:
            if _jina is None:
                from sentence_transformers import SentenceTransformer
                _jina = SentenceTransformer(
                    C.EMBED_MODEL_NAME, device=C.EMBED_DEVICE if hasattr(C, 'EMBED_DEVICE') else ("cuda" if __import__('torch').cuda.is_available() else "cpu"),
                    model_kwargs={"attn_implementation": "eager" if (__import__('torch').cuda.is_available()) else None},
                    trust_remote_code=C.EMBED_TRUST_REMOTE if hasattr(C, 'EMBED_TRUST_REMOTE') else True)
    return _jina


# ── Embedding ────────────────────────────────────────────────────────────────
def embed_query(text: str) -> list[float]:
    """Embed a single query string (in-process Jina v3 on GPU)."""
    model = _get_jina()
    encode_kw = {"convert_to_numpy": True, "show_progress_bar": False}
    if C.EMBED_TASK_QUERY:
        encode_kw["task"] = C.EMBED_TASK_QUERY
    return model.encode([text], **encode_kw)[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed texts (for ingest)."""
    model = _get_jina()
    encode_kw = {"convert_to_numpy": True, "show_progress_bar": False}
    return model.encode(texts, **encode_kw).tolist()


# ── Qdrant search ────────────────────────────────────────────────────────────
def resolve_collections(collection_or_list: str | list) -> list:
    """Normalize a single collection name or list to a list."""
    if isinstance(collection_or_list, str):
        return [collection_or_list]
    return list(collection_or_list)


def qdrant_search_multi(query_vec: list[float], query_text: str,
                         collections: list, top_k: int = 10) -> list:
    """Search multiple Qdrant collections in parallel, merge results."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    qc = QdrantClient(url=C.QDRANT_URL, prefer_grpc=False)
    all_hits = []

    for coll in collections:
        if not qc.collection_exists(coll):
            continue
        hits = qc.query_points(
            collection_name=coll,
            query=query_vec,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        ).points
        for h in hits:
            p = h.payload
            all_hits.append({
                "text": p.get("text", ""),
                "doc_id": p.get("doc_id", ""),
                "doc_type": p.get("doc_type", ""),
                "chunk_idx": p.get("chunk_idx", -1),
                "score": h.score,
                "_collection": coll,
            })
    # Sort by score desc, dedupe by (doc_id, chunk_idx)
    all_hits.sort(key=lambda x: -x["score"])
    seen = set()
    deduped = []
    for h in all_hits:
        key = (h.get("doc_id", ""), h.get("chunk_idx", -1))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(h)
    return deduped[:top_k]


# ── Neo4j graph traversal ────────────────────────────────────────────────────
def neo4j_subgraph(query: str, hops: int = C.GRAPH_HOPS, label: str = None,
                    entry_strategy: str = "keyword") -> list:
    """
    Traverse k-hop subgraph from an entry node.

    entry_strategy:
      - "keyword": server-side CONTAINS on query tokens (fast, exact)
      - "vector":  vector index lookup (semantic, ~10ms)
      - "hybrid":  try keyword first, fall back to vector if overlap < 2
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
        # For 'vector' strategy, skip keyword search entirely and use vector index.
        if entry_strategy == "vector":
            rows = []
        elif ql:
            toks = list(ql)[:8]   # cap candidate clauses
            conds = " OR ".join(
                f"toLower(n.id) CONTAINS $t{i} OR toLower(n.name) CONTAINS $t{i}"
                for i in range(len(toks)))
            params = {f"t{i}": t for i, t in enumerate(toks)}
            rows = s.run(
                f"MATCH (n:Entity{label_clause}) WHERE {conds} "
                f"RETURN n.id AS id, n.name AS name, n.profile AS prof, "
                f"n.source_docs AS source_docs LIMIT 50",
                **params,
            ).data()
        else:
            rows = []
        if not rows:
            # No token overlap. Instead of scanning ALL ~500 entities and
            # embedding each in Python (the old fallback — ~3-4s for an
            # out-of-graph query), do a SERVER-SIDE vector lookup against the
            # entity_vector_idx (top-5 nearest entities in ~10ms). This gives
            # real cosine similarity without the embed storm.
            try:
                q_vec = embed_query(query)
                res = s.run(
                    "CALL db.index.vector.queryNodes($idx, 5, $qvec) "
                    "YIELD node, score "
                    "RETURN node.id AS id, node.name AS name, "
                    "node.profile AS prof, score AS sim",
                    idx=C.ENTITY_VECTOR_INDEX, qvec=q_vec,
                ).data()
                rows = [
                    {"id": r["id"], "name": r.get("name"),
                     "prof": r.get("prof"), "sim": r.get("sim", 0.0)}
                    for r in res if r.get("id")]
            except Exception:
                rows = []

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
        # Prefer a precomputed similarity (e.g. from the server-side vector
        # index lookup in the `sim` field) instead of re-embedding every row.
        if rows and rows[0].get("sim") is not None:
            return sorted(
                ((r["sim"], r) for r in rows if r.get("id")),
                key=lambda x: -x[0],
            )[:5]
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
    # For 'vector' strategy, skip keyword search and go straight to vector index.
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
        # Fall back to vector when keyword has no confident lexical match
        # (overlap < 2). A weak 1-token overlap must NOT suppress the
        # vector entry — otherwise indirect NL queries ("how is bob connected
        # to the GPU through his bug") get a poor keyword entry and the gate
        # wrongly skips a traversable graph.
        if not scored or (scored and isinstance(scored[0][0], int)
                         and scored[0][0] < 2):
            scored = _vector_scored()

    if not scored:
        return []

    entry = scored[0][1]["id"]

    # Traverse k-hop from the entry node; bound the result set with LIMIT so
    # latency stays predictable even on high-degree hubs (~3.7s unbounded).
    # Phase 2 pruning refines the top-N afterwards.
    _LIMIT = getattr(C, "GRAPH_TRAVERSAL_LIMIT", 200)
    with driver.session() as s:
        rec = s.run(
            "MATCH (n:Entity {id:$e})-[*1..%d]-(m:Entity) "
            "RETURN DISTINCT m.id AS id, m.name AS name, m.profile AS prof, "
            "m.source_docs AS source_docs LIMIT %d" % (hops, _LIMIT),
            e=entry,
        ).data()
        edge_rows = s.run(
            "MATCH (n:Entity {id:$e})-[*1..%d]-(m:Entity) "
            "WITH collect(DISTINCT m) AS ms, n AS entry "
            "WITH ms + entry AS ns "
            "UNWIND ns AS a UNWIND ns AS b "
            "MATCH (a)-[r]-(b) WHERE a.id < b.id "
            "RETURN DISTINCT a.id AS src, b.id AS dst, type(r) AS rel LIMIT %d" % (hops, _LIMIT),
            e=entry,
        ).data()
    edges = [(r["src"], r["dst"], r.get("rel")) for r in edge_rows
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

    # Build an id->name map so we can emit human-readable edge
    # statements even when entity `prof`/`source_docs` are empty.
    _id2name = {r["id"]: (r.get("name") or r["id"]) for r in rec if r.get("id")}
    if scored:
        _e0 = scored[0][1]
        if _e0.get("id"):
            _id2name.setdefault(_e0["id"], _e0.get("name") or _e0["id"])

    # Emit traversed relationship edges as explicit statements (A -[rel]-> B).
    # This is what makes an indirect A->D chain answerable: the LLM sees
    # "bob OWNS BUG-204; BUG-204 CAUSED_BY CPU; CPU DEPENDS_ON GPU"
    # rather than disconnected (often empty) node profiles.
    _seen_edges = set()
    for _src, _dst, _rel in edges:
        _pair = (_src, _dst, _rel)
        if _pair in _seen_edges:
            continue
        _seen_edges.add(_pair)
        _src_name = _id2name.get(_src, _src)
        _dst_name = _id2name.get(_dst, _dst)
        rec.append({
            "text": f"GRAPH EDGE: {_src_name} -[{_rel}]-> {_dst_name}",
            "doc_id": f"graph://{label or 'default'}",
            "doc_type": "graph_edge",
            "chunk_idx": -1,
        })

    return rec


# ── Parallel retrieval (vector + graph) ──────────────────────────────────────
def parallel_retrieve(query_vec: list[float], query_text: str,
                       collections: list = None, entry_strategy: str = "keyword") -> tuple[list, list]:
    """Run Qdrant vector search and Neo4j graph traversal in parallel."""
    import threading
    q_res, g_res = [], []

    def _vec():
        nonlocal q_res
        if collections:
            q_res = qdrant_search_multi(query_vec, query_text, collections)

    def _graph():
        nonlocal g_res
        # Try to get a domain label from the first collection
        label = None
        if collections:
            from domain_loader import get_domain
            for coll in collections:
                # Find domain with this collection
                for dname in get_domain.__globals__.get('_DOMAINS', {}):
                    dom = get_domain(dname)
                    if dom.get('collection') == coll:
                        label = dom.get('neo4j_label')
                        break
        g_res = neo4j_subgraph(query_text, label=label, entry_strategy=entry_strategy)

    t1 = threading.Thread(target=_vec)
    t2 = threading.Thread(target=_graph)
    t1.start(); t2.start()
    t1.join(); t2.join()

    return q_res, g_res


# ── Condense (dedupe + rerank) ───────────────────────────────────────────────
def condense(query: str, q_res: list, g_res: list) -> list:
    """
    Normalize, dedupe, rerank, and return top-K context strings.
    Caps the pool before reranking to avoid 5s+ latency on 40 docs.
    """
    # Normalize records (q_res/g_res may be dicts {text,doc_id,...} or strings).
    # Graph results have keys: id, name, prof, source_docs
    # Qdrant results have keys: text, doc_id, doc_type, chunk_idx, score
    def _as_record(x):
        if isinstance(x, dict):
            if "text" in x:
                return x  # Qdrant format
            elif "prof" in x:
                # Graph format: build text from profile + name
                text = f"{x.get('name', '')}: {x.get('prof', '')}"
                return {
                    "text": text,
                    "doc_id": x.get("id", ""),
                    "doc_type": "graph",
                    "chunk_idx": -1,
                }
        return {"text": str(x), "doc_id": "", "doc_type": "", "chunk_idx": -1}

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

    # Limit pool size before rerank to avoid 5s+ latency on 40 docs.
    # The reranker is ~900ms for 40 docs; cap at 15 for ~300ms.
    MAX_RERANK_DOCS = 15
    if len(pool) > MAX_RERANK_DOCS:
        pool = pool[:MAX_RERANK_DOCS]

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
        max_tokens=C.SYNTH_MAX_TOKENS_OUT,
        temperature=C.SYNTH_TEMPERATURE,
        stream=True,
    )
    reasoning_parts = []  # model chain-of-thought (debug only)
    answer_parts = []     # clean final answer
    for chunk in stream:
        delta = chunk.choices[0].delta
        # the model puts reasoning in reasoning_content, clean answer in content
        r = getattr(delta, "reasoning_content", None) or ""
        c = getattr(delta, "content", None) or ""
        if r:
            reasoning_parts.append(r)
        if c:
            answer_parts.append(c)
    # Print reasoning trace + clean answer ONCE (not per-chunk flush — flushing
    # stdout on every streamed token blocks the daemon event loop and makes
    # synthesis ~10x slower than the model actually needs).
    if reasoning_parts:
        print("".join(reasoning_parts))
    clean = "".join(answer_parts)
    if clean:
        print(clean, flush=True)
    return clean


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "how does the ask pipeline work?"
    vec = embed_query(q)
    q_res, g_res = parallel_retrieve(vec, q)
    ctx = condense(q, q_res, g_res)
    synthesize(q, ctx)