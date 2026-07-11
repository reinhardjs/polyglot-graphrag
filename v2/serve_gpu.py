"""
serve_gpu.py — GPU auxiliary-model daemon (model-agnostic, config-driven).

Loads whatever models are configured in config.py onto the GPU at process
start (device="cuda", fp16). Models are NOT hardcoded — change EMBED_MODEL_NAME,
RERANK_MODEL_NAME, or GLINER_MODEL_NAME in config.py to swap any component.

VRAM budget on RTX 3060 (12 GB), shared with the LLM endpoints:
    Embedding model       ~3.0 GB  (config: EMBED_MODEL_NAME)
    Reranker              ~1.0 GB  (config: RERANK_MODEL_NAME)
    GLiNER (lazy)         ~1.6 GB  (config: GLINER_MODEL_NAME)
    -------------------------------
    Aux subtotal          ~5.6 GB  <- this daemon
    + E2B (:8082)         ~1.5 GB  (config: EXTRACTION_LLM)
    + E4B (:8084)         ~3.0 GB  (config: SYNTHESIS_LLM)
    -------------------------------
    Grand total           ~10.1 GB

Endpoints:
    GET  /health           daemon status + VRAM
    GET  /models           currently loaded model identities
    POST /embed_late       full-doc late-style embedding
    POST /embed_query      single query embedding
    POST /rerank           multilingual cross-encoder rerank
    POST /extract_graph    GLiNER zero-shot NER + co-occurrence edges
    POST /ask              ONE-CALL full RAG: embed → Qdrant||Neo4j → rerank → synth
    POST /ingest           single-doc ingest (BackgroundTasks, non-blocking)
    GET  /ingest/status/{task_id}  poll ingest completion
    DELETE /ingest/{doc_id}        remove a document (Qdrant + Neo4j)
    GET  /ingest                    list ingested documents (per collection)
    GET  /collections               list Qdrant collections + point counts
  Multi-domain: pass `collection` (str | list | "all") to /ask and /ingest.
"""
import os
os.environ["HF_HOME"] = "/mnt/data-970-plus/hf_cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import threading
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

import config as C

app = FastAPI(title="GraphRAG GPU Daemon (config-driven, model-agnostic)")

_jina = None
_reranker = None
_gliner = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Ingest task store (in-process, for BackgroundTasks polling) ───────────────
# Maps task_id → status dict. Lives for the daemon's lifetime. A production
# deployment would use Redis, but in-process is sufficient for single-daemon.
_ingest_tasks: Dict[str, Dict[str, Any]] = {}
_ingest_tasks_lock = threading.Lock()
import uuid as _uuid


def _load_all():
    """Preload the auxiliary models onto the GPU at startup.

    ALL model-specific behavior is driven by config.py flags:
      EMBED_TRUST_REMOTE   → trust_remote_code param
      EMBED_USE_HALF       → fp16 conversion toggle
      EMBED_TASK_PASSAGE   → encode() task for docs (None = vanilla)
      EMBED_TASK_QUERY     → encode() task for queries
      RERANK_USE_HALF      → fp16 for reranker
      VECTOR_DIM           → validated against loaded model output dim
    """
    global _jina, _reranker
    from sentence_transformers import SentenceTransformer, CrossEncoder

    print(f"[gpu-daemon] loading models on {DEVICE} ...", flush=True)
    print(f"[gpu-daemon]   embed:  {C.EMBED_MODEL_NAME}", flush=True)
    print(f"[gpu-daemon]     half: {C.EMBED_USE_HALF}", flush=True)
    print(f"[gpu-daemon]     task: {C.EMBED_TASK_PASSAGE}/{C.EMBED_TASK_QUERY}", flush=True)

    # ── Embedding model (identity + flags from config) ───
    load_kw = {"device": DEVICE}
    if C.EMBED_TRUST_REMOTE:
        load_kw["trust_remote_code"] = True

    try:
        _jina = SentenceTransformer(C.EMBED_MODEL_NAME, **load_kw)
    except Exception as e:
        print(f"[gpu-daemon] FATAL: failed to load embedder "
              f"'{C.EMBED_MODEL_NAME}': {e}", flush=True)
        import sys; sys.exit(1)

    if C.EMBED_USE_HALF:
        try:
            _jina = _jina.half()
        except Exception as e:
            print(f"[gpu-daemon] WARNING: .half() failed ({e}), "
                  f"running fp32 — VRAM will be higher", flush=True)

    # Validate dimension
    dim = _jina.get_sentence_embedding_dimension()
    if dim != C.VECTOR_DIM:
        print(
            f"[gpu-daemon] WARNING: model output dim={dim} "
            f"but VECTOR_DIM={C.VECTOR_DIM} — UPDATE VECTOR_DIM in config!",
            flush=True,
        )

    # ── Reranker ───
    _reranker = CrossEncoder(C.RERANK_MODEL_NAME, device=DEVICE)
    if C.RERANK_USE_HALF:
        try:
            _reranker = _reranker.half()
        except Exception:
            pass  # some rerankers don't support fp16; OK

    # ── Warm models ───
    embed_kw = {}
    if C.EMBED_TASK_PASSAGE:
        embed_kw["task"] = C.EMBED_TASK_PASSAGE
    _jina.encode(["warm"], **embed_kw)
    _reranker.predict([("warm", "warm")])

    print(
        f"[gpu-daemon] resident models loaded on {DEVICE} "
        f"(GLiNER lazy-loaded on demand). "
        f"Allocated: {torch.cuda.memory_allocated()/1e9:.1f} GB",
        flush=True,
    )


def _load_gliner():
    """Lazy-load GLiNER on first /extract_graph call (fallback path only).

    Model identity comes from config.GLINER_MODEL_NAME.
    """
    global _gliner
    if _gliner is None:
        from gliner import GLiNER
        print(f"[gpu-daemon] lazy-loading GLiNER: {C.GLINER_MODEL_NAME} ...",
              flush=True)
        _gliner = GLiNER.from_pretrained(C.GLINER_MODEL_NAME).to(DEVICE).half()
        _gliner.predict_entities("warm", ["Microservice", "Database"])
        print(
            f"[gpu-daemon] GLiNER resident. "
            f"Allocated: {torch.cuda.memory_allocated()/1e9:.1f} GB",
            flush=True,
        )
    return _gliner


# ── Request schemas ──────────────────────────────────────────────────────────
class EmbedLateReq(BaseModel):
    text: str
    doc_id: str = "doc"


class EmbedQueryReq(BaseModel):
    text: str


class RerankReq(BaseModel):
    query: str
    docs: List[str]


class ExtractReq(BaseModel):
    text: str


class AskReq(BaseModel):
    query: str
    top_k: int = 5
    synthesize: bool = True
    skip_cache: bool = False
    collection: Optional[str | List[str]] = None   # Qdrant collection(s); None→default
    domain: Optional[str] = None                     # reserved for v2.6.0 profiles


class IngestReq(BaseModel):
    text: str
    doc_id: str
    doc_type: str = "eng"
    author: str = "unknown"
    extract_graph: bool = True
    if_checksum: Optional[str] = None   # if provided, skip if unchanged
    collection: Optional[str] = None     # Qdrant collection; None→default engineering_chunks


class IngestStatusReq(BaseModel):
    pass


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/embed_late")
def embed_late(req: EmbedLateReq):
    """Late-style embedding: sentence-split, embed each.

    Uses the task from config.EMBED_TASK_PASSAGE (Jina: "retrieval.passage").
    For vanilla models (BGE, Gemma), it's None — standard encode().
    """
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', req.text)
                 if s.strip()]
    if not sentences:
        sentences = [req.text.strip()]
    encode_kw = {"convert_to_numpy": True, "show_progress_bar": False}
    if C.EMBED_TASK_PASSAGE:
        encode_kw["task"] = C.EMBED_TASK_PASSAGE
    vecs = _jina.encode(sentences, **encode_kw)
    return {
        "doc_id": req.doc_id,
        "chunks": [
            {"chunk_idx": i, "vector": vecs[i].tolist(), "text": sentences[i]}
            for i in range(len(sentences))
        ],
    }


@app.post("/embed_query")
def embed_query(req: EmbedQueryReq):
    encode_kw = {"convert_to_numpy": True, "show_progress_bar": False}
    if C.EMBED_TASK_QUERY:
        encode_kw["task"] = C.EMBED_TASK_QUERY
    vec = _jina.encode([req.text], **encode_kw)[0]
    return {"vector": vec.tolist()}


@app.post("/rerank")
def rerank(req: RerankReq):
    if not req.docs:
        return {"ranked": []}
    scores = _reranker.predict([(req.query, d) for d in req.docs])
    ranked = sorted([(i, float(s)) for i, s in enumerate(scores)],
                    key=lambda x: x[1], reverse=True)
    return {"ranked": ranked}


@app.post("/extract_graph")
def extract_graph(req: ExtractReq):
    """GPU GLiNER extraction + co-occurrence heuristic edges (fallback path).

    GLiNER is a zero-shot NER model loaded on CUDA. It predicts entities of our
    GLiNER_LABELS; for each sentence we create ASSOCIATED_WITH edges between
    co-occurring entities. The caller (ingest.py) resolves entity identities
    via vector matching against Neo4j's vector index.
    """
    import re
    gliner = _load_gliner()
    preds = gliner.predict_entities(req.text, C.GLINER_LABELS,
                                    threshold=C.GLINER_THRESHOLD)
    nodes, edges, seen = {}, [], set()
    for sent in [s for s in re.split(r'(?<=[.!?])\s+', req.text) if s.strip()]:
        s_low = sent.lower()
        local = []
        for p in preds:
            if p["text"].lower() in s_low:
                nodes[p["text"].strip().lower()] = p["label"]
                local.append(p["text"].strip().lower())
        for a in range(len(local)):
            for b in range(a + 1, len(local)):
                pair = tuple(sorted((local[a], local[b])))
                if pair not in seen:
                    seen.add(pair)
                    edges.append({
                        "source": pair[0], "target": pair[1],
                        "type": "ASSOCIATED_WITH",
                    })
    return {
        "nodes": [{"id": k, "type": v} for k, v in nodes.items()],
        "edges": edges,
    }


# ── Collection resolver (multi-domain / cross-domain) ────────────────────────
def _resolve_collections(collection: Optional[str | List[str]]) -> List[str]:
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
        # Domain alias? (e.g. "legal" → "legal_chunks")
        if collection in C.QDRANT_COLLECTIONS:
            return [C.QDRANT_COLLECTIONS[collection]]
        # Direct collection name (must exist)
        return [collection]
    # List of aliases / names
    out = []
    for c in collection:
        if c == "all":
            out.extend(C.QDRANT_COLLECTIONS.values())
        elif c in C.QDRANT_COLLECTIONS:
            out.append(C.QDRANT_COLLECTIONS[c])
        else:
            out.append(c)
    # dedupe, preserve order
    seen = set()
    return [x for x in out if not (x in seen or seen.add(x))]


@app.post("/ask")
def ask(req: AskReq):
    """Full RAG answer in ONE call with semantic cache + route metadata."""
    import ask as rag
    import numpy as np
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    # 1. Embed query in-process (resident model on GPU)
    encode_kw = {"convert_to_numpy": True, "show_progress_bar": False}
    if C.EMBED_TASK_QUERY:
        encode_kw["task"] = C.EMBED_TASK_QUERY
    vec = _jina.encode([req.query], **encode_kw)[0].tolist()

    # 2. SEMANTIC CACHE
    if req.synthesize and not req.skip_cache:
        try:
            qc = QdrantClient(url=C.QDRANT_URL, prefer_grpc=False)
            if qc.collection_exists(C.COLL_CACHE):
                hits = qc.query_points(
                    C.COLL_CACHE, query=vec, limit=1,
                    score_threshold=C.CACHE_THRESHOLD, with_payload=True,
                ).points
                if hits:
                    p = hits[0].payload
                    return {
                        "query": req.query,
                        "source": "cache",
                        "path": p.get("path", "cache"),
                        "qdrant_hits": p.get("qdrant_hits", 0),
                        "graph_hits": p.get("graph_hits", 0),
                        "n_contexts": len(p.get("contexts", [])),
                        "contexts": p.get("contexts", []),
                        "rerank_scores": p.get("rerank_scores", []),
                        "answer": p.get("answer", ""),
                        "cache_hit": True,
                    }
        except Exception:
            pass

    # 3. Parallel retrieval: Qdrant + Neo4j
    import threading
    # Resolve collection(s) → list of Qdrant collection names
    collections = _resolve_collections(req.collection)
    q_res, g_res = [], []
    t1 = threading.Thread(
        target=lambda: q_res.extend(rag.qdrant_search_multi(vec, req.query, collections)))
    t2 = threading.Thread(target=lambda: g_res.extend(rag.neo4j_subgraph(req.query)))
    t1.start(); t2.start(); t1.join(); t2.join()

    qdrant_hits = len(q_res)
    graph_hits = len(g_res)
    path = "hybrid" if (qdrant_hits and graph_hits) else (
        "qdrant" if qdrant_hits else ("graph" if graph_hits else "none")
    )

    # 4. Fuse + rerank in-process
    pool = list(dict.fromkeys(q_res + g_res))
    if not pool:
        return {
            "query": req.query, "source": "llm", "path": path,
            "qdrant_hits": qdrant_hits, "graph_hits": graph_hits,
            "n_contexts": 0, "contexts": [], "rerank_scores": [],
            "answer": "", "cache_hit": False,
        }

    scores = _reranker.predict([(req.query, d) for d in pool])
    ranked = sorted([(i, float(s)) for i, s in enumerate(scores)],
                    key=lambda x: x[1], reverse=True)[:req.top_k]
    contexts = [pool[i] for i, _ in ranked]
    rerank_scores = [s for _, s in ranked]

    # 5. Synthesize with E4B (separate process)
    answer = ""
    if req.synthesize:
        answer = rag.synthesize(req.query, contexts)

    result = {
        "query": req.query,
        "source": "llm",
        "path": path,
        "qdrant_hits": qdrant_hits,
        "graph_hits": graph_hits,
        "n_contexts": len(contexts),
        "contexts": contexts,
        "rerank_scores": rerank_scores,
        "answer": answer,
        "cache_hit": False,
    }

    # 6. Store in cache
    if req.synthesize and not req.skip_cache:
        try:
            qc = QdrantClient(url=C.QDRANT_URL, prefer_grpc=False)
            if qc.collection_exists(C.COLL_CACHE):
                qc.upsert(
                    C.COLL_CACHE,
                    [PointStruct(
                        id=abs(hash(req.query)) % (2**63),
                        vector=vec,
                        payload={
                            "query": req.query,
                            "path": path,
                            "qdrant_hits": qdrant_hits,
                            "graph_hits": graph_hits,
                            "contexts": contexts,
                            "rerank_scores": rerank_scores,
                            "answer": answer,
                        },
                    )],
                    wait=True,
                )
        except Exception:
            pass

    return result


@app.post("/ingest")
def ingest(req: IngestReq, background_tasks: BackgroundTasks):
    """Single-document ingestion via HTTP API (non-blocking).

    Returns 202 Accepted immediately with a task_id. Extraction runs in the
    background (FastAPI's thread pool) so it does NOT block /ask queries.
    Poll GET /ingest/status/{task_id} for completion.

    If `if_checksum` matches the stored payload, returns 304 (unchanged)
    immediately — no extraction, <10ms.
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    if not req.doc_id or not req.doc_id.strip():
        raise HTTPException(status_code=400, detail="doc_id is required")

    task_id = _uuid.uuid4().hex
    with _ingest_tasks_lock:
        _ingest_tasks[task_id] = {
            "doc_id": req.doc_id, "status": "queued", "result": None,
            "error": None,
        }

    def _run():
        import ingest as ingest_mod
        with _ingest_tasks_lock:
            _ingest_tasks[task_id]["status"] = "running"
        try:
            res = ingest_mod.ingest_text(
                text=req.text,
                doc_id=req.doc_id,
                meta={"doc_type": req.doc_type, "author": req.author},
                extract_graph=req.extract_graph,
                if_checksum=req.if_checksum,
                collection=req.collection or C.COLL_CHUNKS,
            )
            with _ingest_tasks_lock:
                _ingest_tasks[task_id]["status"] = "done"
                _ingest_tasks[task_id]["result"] = res
        except Exception as e:
            with _ingest_tasks_lock:
                _ingest_tasks[task_id]["status"] = "error"
                _ingest_tasks[task_id]["error"] = str(e)

    background_tasks.add_task(_run)
    return {"task_id": task_id, "status": "accepted", "doc_id": req.doc_id}


@app.get("/ingest/status/{task_id}")
def ingest_status(task_id: str):
    """Poll status of a background ingest task."""
    with _ingest_tasks_lock:
        task = _ingest_tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="task not found")
        return {
            "task_id": task_id,
            "status": task["status"],
            "doc_id": task["doc_id"],
            "result": task["result"],
            "error": task["error"],
        }


@app.delete("/ingest/{doc_id}")
def delete_ingest(doc_id: str, collection: str = C.COLL_CHUNKS):
    """Remove a document from Qdrant + Neo4j in one call.

    Uses the same delete-before-reingest logic as batch ingest. Returns counts
    of deleted vectors / cleaned nodes. 404 if the document does not exist.
    """
    import ingest as ingest_mod
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    # ── Check existence + accurate count (Qdrant payload) ──
    qc = ingest_mod.get_qdrant()
    flt = Filter(must=[FieldCondition(key="doc_id",
                                      match=MatchValue(value=doc_id))])
    existing = qc.scroll(collection, scroll_filter=flt,
                         limit=1, with_payload=False)[0]
    if not existing:
        raise HTTPException(status_code=404, detail=f"doc_id '{doc_id}' not found")

    vectors_deleted = qc.count(collection, count_filter=flt, exact=True).count

    # ── Neo4j: count + clean edges + orphaned nodes for this doc ──
    driver = ingest_mod.get_neo4j()
    nodes_cleaned = 0
    try:
        with driver.session() as s:
            r = s.run(
                "MATCH (n:Entity) WHERE $doc IN n.source_docs "
                "RETURN count(n) AS c", doc=doc_id,
            ).single()
            nodes_cleaned = r["c"] if r else 0
        ingest_mod.delete_doc_neo4j(doc_id, driver)
    finally:
        driver.close()

    # ── Qdrant: delete all points for doc_id ──
    ingest_mod.delete_doc_qdrant(doc_id, collection)

    return {
        "doc_id": doc_id,
        "status": "deleted",
        "vectors_deleted": vectors_deleted,
        "nodes_cleaned": nodes_cleaned,
        "collection": collection,
    }


@app.get("/ingest")
def list_ingest(collection: str = C.COLL_CHUNKS):
    """List all ingested documents with their metadata + checksum.

    `collection` selects which Qdrant collection to list (multi-domain).
    Defaults to engineering_chunks.
    """
    import ingest as ingest_mod
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    qc = ingest_mod.get_qdrant()
    docs = {}
    # Scroll all points, group by doc_id
    offset = None
    while True:
        pts, offset = qc.scroll(
            collection,
            scroll_filter=None,
            limit=256,
            with_payload=["doc_id", "doc_type", "checksum", "chunk_idx"],
            offset=offset,
        )
        for p in pts:
            pl = p.payload or {}
            did = pl.get("doc_id", "unknown")
            if did not in docs:
                docs[did] = {
                    "doc_id": did,
                    "doc_type": pl.get("doc_type", ""),
                    "chunks": 0,
                    "checksum": pl.get("checksum", ""),
                }
            docs[did]["chunks"] += 1
        if offset is None:
            break

    return {"documents": list(docs.values())}


@app.get("/collections")
def list_collections():
    """List all Qdrant collections with point counts."""
    import ingest as ingest_mod
    qc = ingest_mod.get_qdrant()
    cols = {}
    try:
        names = qc.get_collections().collections
        for c in names:
            name = c.name
            try:
                info = qc.get_collection(name)
                cols[name] = {
                    "points": info.points_count or 0,
                    "vectors": info.config.params.vectors,
                }
            except Exception:
                cols[name] = {"points": -1}
    except Exception as e:
        cols["_error"] = str(e)
    return {"collections": cols}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "cuda_alloc_gb": round(torch.cuda.memory_allocated()/1e9, 1),
    }


@app.get("/models")
def models():
    """Return currently loaded model identities AND behavior flags (from config.py).

    Use this to verify which models the daemon loaded and how they're configured.
    Swap any by changing config.py and restarting the daemon.
    """
    return {
        "embedder": {
            "name": C.EMBED_MODEL_NAME,
            "dim": C.VECTOR_DIM,
            "half": C.EMBED_USE_HALF,
            "task_passage": C.EMBED_TASK_PASSAGE,
            "task_query": C.EMBED_TASK_QUERY,
            "trust_remote": C.EMBED_TRUST_REMOTE,
            "matryoshka_dim": C.EMBED_MATRYOSHKA_DIM,
        },
        "reranker": {
            "name": C.RERANK_MODEL_NAME,
            "half": C.RERANK_USE_HALF,
        },
        "gliner": C.GLINER_MODEL_NAME,
        "extraction": {
            "model": C.EXTRACTION_LLM_MODEL,
            "url": C.EXTRACTION_LLM_BASE_URL,
            "reads_reasoning": C.EXTRACTION_READS_REASONING,
        },
        "synthesis": {
            "model": C.SYNTHESIS_LLM_MODEL,
            "url": C.SYNTHESIS_LLM_BASE_URL,
        },
        "entity_resolution": {
            "threshold": C.ENTITY_RESOLUTION_THRESHOLD,
            "index": C.ENTITY_VECTOR_INDEX,
        },
    }


@app.on_event("startup")
def on_startup():
    _load_all()
    # Ensure Qdrant cache collection exists
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        qc = QdrantClient(url=C.QDRANT_URL, prefer_grpc=False)
        if not qc.collection_exists(C.COLL_CACHE):
            qc.create_collection(
                C.COLL_CACHE,
                vectors_config=VectorParams(
                    size=C.VECTOR_DIM, distance=Distance.COSINE,
                ),
            )
            print(f"[gpu-daemon] created cache collection '{C.COLL_CACHE}'",
                  flush=True)
    except Exception as e:
        print(f"[gpu-daemon] cache init skipped: {e}", flush=True)

    # Ensure Neo4j vector index exists for entity resolution
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            C.NEO4J_URI, auth=(C.NEO4J_USER, C.NEO4J_PASSWORD),
        )
        with driver.session() as s:
            s.run(
                "CREATE VECTOR INDEX $idx IF NOT EXISTS "
                "FOR (n:Entity) ON (n.name_vector) "
                "OPTIONS {indexConfig: {"
                "  `vector.dimensions`: $dim,"
                "  `vector.similarity_function`: 'cosine'"
                "}}",
                idx=C.ENTITY_VECTOR_INDEX, dim=C.VECTOR_DIM,
            )
            print(f"[gpu-daemon] Neo4j vector index '{C.ENTITY_VECTOR_INDEX}' "
                  f"ready (dim={C.VECTOR_DIM}, cosine)", flush=True)
    except Exception as e:
        print(f"[gpu-daemon] Neo4j vector index init skipped: {e}", flush=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)
