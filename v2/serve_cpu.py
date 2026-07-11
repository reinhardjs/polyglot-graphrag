"""serve_cpu.py — CPU auxiliary-model daemon (model-agnostic, config-driven).

FALLBACK DAEMON — used when CUDA torch is unavailable (serve_gpu.py is primary).
Models are lazy-loaded on first use; identities and behavior flags come from
config.py (same as serve_gpu.py — swap by changing config, restart daemon).

Exposes the same endpoints as serve_gpu.py: /health, /models, /embed_late,
/embed_query, /rerank, /extract_graph, /ask, /ingest (+status), DELETE /ingest,
/collections. Multi-domain: pass `collection` (str|list|"all") to /ask, /ingest.
"""
import os
os.environ["HF_HOME"] = "/mnt/data-970-plus/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "8"

import uuid
import threading
from fastapi import FastAPI, HTTPException, BackgroundTasks, Response
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np

import config as C

app = FastAPI(title="GraphRAG CPU Daemon (config-driven, model-agnostic)")

# ── Background ingest task store (in-process, same shape as serve_gpu.py) ──────
_ingest_tasks: Dict[str, dict] = {}
_ingest_tasks_lock = threading.Lock()

# ── Model handles (lazy-loaded on first use) ──────────────────────────────────
_jina = None
_reranker = None
_gliner = None


def _load_jina():
    """Lazy-load the embedding model (identity + flags from config)."""
    global _jina
    if _jina is None:
        from sentence_transformers import SentenceTransformer
        print(f"[cpu-daemon] loading embedder: {C.EMBED_MODEL_NAME} ...",
              flush=True)
        load_kw = {}
        if C.EMBED_TRUST_REMOTE:
            load_kw["trust_remote_code"] = True
        _jina = SentenceTransformer(C.EMBED_MODEL_NAME, **load_kw)
        # NOTE: NEVER call .half() on CPU — fp16 on x86 destroys perf.
        # EMBED_USE_HALF is for GPU only; CPU runs fp32.
        # Warm
        ekw = {}
        if C.EMBED_TASK_PASSAGE:
            ekw["task"] = C.EMBED_TASK_PASSAGE
        _jina.encode(["warm"], **ekw)
        print(f"[cpu-daemon] embedder loaded", flush=True)
    return _jina


def _load_reranker():
    """Lazy-load the CPU-optimized reranker (RERANK_MODEL_NAME_CPU from config)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print(f"[cpu-daemon] loading reranker: {C.RERANK_MODEL_NAME_CPU} ...",
              flush=True)
        _reranker = CrossEncoder(C.RERANK_MODEL_NAME_CPU)
        # NOTE: NEVER call .half() on CPU — RERANK_USE_HALF is for GPU only.
        # fp16 on x86 destroys cross-encoder perf (every matmul upcasts).
        _reranker.predict([("warm", "warm")])
        print(f"[cpu-daemon] reranker loaded", flush=True)
    return _reranker


def _load_gliner():
    """Lazy-load GLiNER (identity from config)."""
    global _gliner
    if _gliner is None:
        from gliner import GLiNER
        print(f"[cpu-daemon] loading GLiNER: {C.GLINER_MODEL_NAME} ...",
              flush=True)
        _gliner = GLiNER.from_pretrained(C.GLINER_MODEL_NAME)
        _gliner.predict_entities("warm", ["Microservice", "Database"])
        print(f"[cpu-daemon] GLiNER loaded", flush=True)
    return _gliner


def _embed_encode(texts, passage=False):
    """Encode with config-driven task flags — shared by embed_late and
    embed_query endpoints."""
    model = _load_jina()
    ekw = {"convert_to_numpy": True, "show_progress_bar": False}
    task = C.EMBED_TASK_PASSAGE if passage else C.EMBED_TASK_QUERY
    if task:
        ekw["task"] = task
    return model.encode(texts, **ekw)


# ── Request schemas ──────────────────────────────────────────────────────────
class EmbedLateReq(BaseModel):
    text: str
    doc_id: str = "doc"
    strategy: str = "sentence"          # v2.6.0: sentence|paragraph|section|fixed
    chunk_size: int = 512
    overlap: int = 64
    header_prefix: str = "##"


class EmbedQueryReq(BaseModel):
    text: str
    domain: Optional[str] = None         # v2.6.0 profile (engineering/medical/...)


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
    collection: Optional[str | List[str]] = None
    domain: Optional[str] = None


class IngestReq(BaseModel):
    text: str
    doc_id: str
    doc_type: str = "eng"
    author: str = "unknown"
    extract_graph: bool = True
    if_checksum: Optional[str] = None
    collection: Optional[str] = None
    domain: Optional[str] = None         # v2.6.0 profile
    metadata: Optional[Dict[str, Any]] = None  # v2.6.0 REQ-7


# ── Collection resolver (shared semantics with serve_gpu.py) ─────────────────
def _resolve_collections(collection: Optional[str | List[str]]) -> List[str]:
    """Resolve None/alias/name/list/'all' into a list of Qdrant collection names."""
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


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/embed_late")
def embed_late(req: EmbedLateReq):
    import chunking as CH
    chunks = CH.chunk_text(req.text, strategy=req.strategy,
                           chunk_size=req.chunk_size, overlap=req.overlap,
                           header_prefix=req.header_prefix)
    if not chunks:
        chunks = [req.text.strip()]
    vecs = _embed_encode(chunks, passage=True)
    return {
        "doc_id": req.doc_id,
        "chunks": [
            {"chunk_idx": i, "vector": vecs[i].tolist(), "text": chunks[i]}
            for i in range(len(chunks))
        ],
    }


@app.post("/embed_query")
def embed_query(req: EmbedQueryReq):
    from query_modulator import QueryModulator
    text = QueryModulator.moderate(req.text, domain=req.domain)
    vec = _embed_encode([text], passage=False)[0]
    return {"vector": vec.tolist(), "text_used": text}


@app.post("/rerank")
def rerank(req: RerankReq):
    model = _load_reranker()
    if not req.docs:
        return {"ranked": []}
    scores = model.predict([(req.query, d) for d in req.docs])
    ranked = sorted([(i, float(s)) for i, s in enumerate(scores)],
                    key=lambda x: x[1], reverse=True)
    return {"ranked": ranked}


@app.post("/extract_graph")
def extract_graph(req: ExtractReq):
    """CPU GLiNER NER extraction + co-occurrence edges (fallback path)."""
    import re
    model = _load_gliner()
    preds = model.predict_entities(req.text, C.GLINER_LABELS,
                                   threshold=C.GLINER_THRESHOLD)
    nodes, edges, seen = {}, [], set()
    for sent in [s.strip() for s in re.split(r'(?<=[.!?])\s+', req.text)
                 if s.strip()]:
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


@app.post("/ask")
def ask(req: AskReq):
    """Full RAG in ONE call (CPU path — slower than serve_gpu.py GPU version)."""
    import ask as rag
    import threading

    # 0. Domain profile (v2.6.0)
    profile = C.load_domain_profile(req.domain) if req.domain else None

    # 1. Embed query (CPU, lazy-load on first call)
    from query_modulator import QueryModulator
    query_text = QueryModulator.moderate(req.query, domain=req.domain)
    vec = _embed_encode([query_text], passage=False)[0].tolist()

    # 2. Parallel retrieval: Qdrant (multi-collection) + Neo4j
    if req.collection is None and profile is not None:
        req_collection = profile["domain"]["collection"]
    else:
        req_collection = req.collection
    collections = _resolve_collections(req_collection)
    q_res, g_res = [], []
    t1 = threading.Thread(
        target=lambda: q_res.extend(rag.qdrant_search_multi(vec, req.query, collections)))
    t2 = threading.Thread(target=lambda: g_res.extend(
        rag.neo4j_subgraph(req.query,
                          label=(profile["domain"]["neo4j_label"] if profile else None),
                          entry_strategy=(profile["neo4j_entry"]["strategy"]
                                          if profile else "keyword"))))
    t1.start(); t2.start(); t1.join(); t2.join()

    # 3. Fuse + cap + rerank (CPU-optimized)
    def _as_record(x):
        if isinstance(x, dict):
            return x
        return {"text": x, "doc_id": "", "doc_type": "", "chunk_idx": -1}
    all_records = [_as_record(x) for x in q_res] + [_as_record(x) for x in g_res]
    seen = set()
    pool = []
    for r in all_records:
        key = (r.get("doc_id", ""), r.get("text", ""))
        if key in seen:
            continue
        seen.add(key)
        pool.append(r)
    if C.RERANK_CPU_POOL_CAP and len(pool) > C.RERANK_CPU_POOL_CAP:
        pool = pool[:C.RERANK_CPU_POOL_CAP]
    if not pool:
        return {"query": req.query, "n_contexts": 0, "contexts": [],
                "contexts_numbered": [], "contexts_meta": [], "sources": [],
                "answer": ""}
    reranker = _load_reranker()
    scores = reranker.predict([(req.query, r["text"]) for r in pool])
    ranked = sorted([(i, float(s)) for i, s in enumerate(scores)],
                    key=lambda x: x[1], reverse=True)[:req.top_k]
    contexts = [pool[i]["text"] for i, _ in ranked]
    contexts_meta = [{"doc_id": pool[i].get("doc_id", ""),
                      "doc_type": pool[i].get("doc_type", ""),
                      "chunk_idx": pool[i].get("chunk_idx", -1)}
                     for i, _ in ranked]

    # 4. Synthesize with E4B (:8084)
    answer = ""
    if req.synthesize:
        answer = rag.synthesize(req.query, contexts, profile)

    contexts_numbered = [f"[{i+1}] {c}" for i, c in enumerate(contexts)]
    sources = []
    ref_by_doc = {}
    for meta in contexts_meta:
        did = meta["doc_id"] or "__graph__"
        if did not in ref_by_doc:
            ref_by_doc[did] = len(sources) + 1
            sources.append({"ref": len(sources) + 1,
                             "doc_id": meta["doc_id"],
                             "doc_type": meta["doc_type"],
                             "chunk_idx": meta["chunk_idx"]})

    return {"query": req.query, "n_contexts": len(contexts),
            "contexts": contexts,
            "contexts_numbered": contexts_numbered,
            "contexts_meta": contexts_meta,
            "sources": sources,
            "answer": answer}


# ── Ingest endpoints (parity with serve_gpu.py) ──────────────────────────────
def _run_ingest(task_id: str, req: "IngestReq", target_collection: str = None):
    import ingest as ingest_mod
    # Derive collection from domain profile when not explicitly given.
    if target_collection is None:
        target_collection = req.collection
        if target_collection is None and req.domain:
            prof = C.load_domain_profile(req.domain)
            target_collection = prof["domain"]["collection"]
    try:
        with _ingest_tasks_lock:
            _ingest_tasks[task_id]["status"] = "running"
        res = ingest_mod.ingest_text(
            text=req.text, doc_id=req.doc_id,
            meta=({"doc_type": req.doc_type, "author": req.author}
                  | (req.metadata or {})),
            extract_graph=req.extract_graph, if_checksum=None,  # handled synchronously
            collection=target_collection or C.COLL_CHUNKS,
            domain=req.domain, metadata=req.metadata,
        )
        with _ingest_tasks_lock:
            _ingest_tasks[task_id].update(status="done", result=res)
    except Exception as e:
        with _ingest_tasks_lock:
            _ingest_tasks[task_id].update(status="error", error=str(e))


@app.post("/ingest")
def ingest(req: IngestReq, background_tasks: BackgroundTasks, response: Response):
    """Single-doc ingest (non-blocking). Returns 202 + task_id.

    If `if_checksum` matches the stored payload, returns 304 (unchanged)
    immediately — no background task, <10ms.
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    if not req.doc_id or not req.doc_id.strip():
        raise HTTPException(status_code=400, detail="doc_id is required")

    # Derive collection from domain profile when not explicitly given.
    target_collection = req.collection
    if target_collection is None and req.domain:
        prof = C.load_domain_profile(req.domain)
        target_collection = prof["domain"]["collection"]
    target_collection = target_collection or C.COLL_CHUNKS

    # ── Synchronous incremental guard: skip if unchanged ──
    if req.if_checksum is not None:
        try:
            import json as _json
            import ingest as ingest_mod
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            qc = ingest_mod.get_qdrant()
            hits = qc.scroll(
                target_collection,
                scroll_filter=Filter(
                    must=[FieldCondition(
                        key="doc_id", match=MatchValue(value=req.doc_id))]),
                limit=1, with_payload=["checksum"],
            )[0]
            if hits and hits[0].payload.get("checksum") == req.if_checksum:
                return Response(
                    content=_json.dumps({
                        "doc_id": req.doc_id, "status": "unchanged",
                        "skipped": True, "checksum": req.if_checksum,
                    }),
                    status_code=304,
                    media_type="application/json",
                )
        except Exception:
            pass  # fall through to full ingest if the check fails

    task_id = uuid.uuid4().hex
    with _ingest_tasks_lock:
        _ingest_tasks[task_id] = {
            "status": "accepted", "doc_id": req.doc_id,
            "result": None, "error": None,
        }
    background_tasks.add_task(_run_ingest, task_id, req, target_collection)
    response.status_code = 202
    return {"task_id": task_id, "status": "accepted", "doc_id": req.doc_id}


@app.get("/ingest/status/{task_id}")
def ingest_status(task_id: str):
    with _ingest_tasks_lock:
        task = _ingest_tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="task not found")
        return {"task_id": task_id, "status": task["status"],
                "doc_id": task["doc_id"], "result": task["result"],
                "error": task["error"]}


@app.delete("/ingest/{doc_id}")
def delete_ingest(doc_id: str, collection: str = C.COLL_CHUNKS):
    import ingest as ingest_mod
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    qc = ingest_mod.get_qdrant()
    flt = Filter(must=[FieldCondition(key="doc_id",
                                      match=MatchValue(value=doc_id))])
    existing = qc.scroll(collection, scroll_filter=flt,
                         limit=1, with_payload=False)[0]
    if not existing:
        raise HTTPException(status_code=404, detail=f"doc_id '{doc_id}' not found")
    vectors_deleted = qc.count(collection, count_filter=flt, exact=True).count
    driver = ingest_mod.get_neo4j()
    nodes_cleaned = 0
    try:
        with driver.session() as s:
            r = s.run("MATCH (n:Entity) WHERE $doc IN n.source_docs "
                      "RETURN count(n) AS c", doc=doc_id).single()
            nodes_cleaned = r["c"] if r else 0
        ingest_mod.delete_doc_neo4j(doc_id, driver)
    finally:
        driver.close()
    ingest_mod.delete_doc_qdrant(doc_id, collection)
    return {"doc_id": doc_id, "status": "deleted",
            "vectors_deleted": vectors_deleted, "nodes_cleaned": nodes_cleaned,
            "collection": collection}


@app.get("/ingest")
def list_ingest(collection: str = C.COLL_CHUNKS):
    import ingest as ingest_mod
    qc = ingest_mod.get_qdrant()
    docs = {}
    offset = None
    while True:
        pts, offset = qc.scroll(
            collection, scroll_filter=None, limit=256,
            with_payload=["doc_id", "doc_type", "checksum", "chunk_idx",
                          "metadata"],
            offset=offset,
        )
        for p in pts:
            pl = p.payload or {}
            did = pl.get("doc_id", "unknown")
            if did not in docs:
                docs[did] = {"doc_id": did, "doc_type": pl.get("doc_type", ""),
                             "chunks": 0, "checksum": pl.get("checksum", ""),
                             "metadata": pl.get("metadata", {})}
            docs[did]["chunks"] += 1
        if offset is None:
            break
    return {"documents": list(docs.values())}


@app.get("/collections")
def list_collections():
    import ingest as ingest_mod
    qc = ingest_mod.get_qdrant()
    cols = {}
    try:
        for c in qc.get_collections().collections:
            try:
                info = qc.get_collection(c.name)
                cols[c.name] = {"points": info.points_count or 0}
            except Exception:
                cols[c.name] = {"points": -1}
    except Exception as e:
        cols["_error"] = str(e)
    return {"collections": cols}


@app.get("/profiles")
def list_profiles():
    """List all loaded domain profiles (parity with serve_gpu.py, v2.6.0)."""
    import config as cfg
    import glob as _glob
    import os as _os
    out = []
    ddir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "domains")
    for path in sorted(_glob.glob(_os.path.join(ddir, "*.toml"))):
        try:
            p = cfg.load_domain_profile(_os.path.splitext(_os.path.basename(path))[0])
            out.append({
                "domain": p["domain"]["name"],
                "collection": p["domain"]["collection"],
                "neo4j_label": p["domain"]["neo4j_label"],
                "description": p["domain"].get("description", ""),
                "chunking": p.get("chunking", {}).get("strategy", "sentence"),
                "entry_strategy": p.get("neo4j_entry", {}).get("strategy", "keyword"),
            })
        except Exception as e:
            out.append({"file": _os.path.basename(path), "error": str(e)})
    return {"profiles": out}


@app.get("/health")
def health():
    return {"status": "ok", "device": "cpu"}


@app.get("/models")
def models():
    return {
        "embedder": {
            "name": C.EMBED_MODEL_NAME,
            "dim": C.VECTOR_DIM,
            "half": C.EMBED_USE_HALF,
            "task_passage": C.EMBED_TASK_PASSAGE,
            "task_query": C.EMBED_TASK_QUERY,
        },
        "reranker": {
            "name": C.RERANK_MODEL_NAME_CPU,
            "pool_cap": C.RERANK_CPU_POOL_CAP,
        },
        "gliner": C.GLINER_MODEL_NAME,
        "extraction": {
            "model": C.EXTRACTION_LLM_MODEL,
            "reads_reasoning": C.EXTRACTION_READS_REASONING,
        },
        "synthesis": {
            "model": C.SYNTHESIS_LLM_MODEL,
        },
        "entity_resolution": {
            "threshold": C.ENTITY_RESOLUTION_THRESHOLD,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)
