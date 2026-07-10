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
"""
import os
os.environ["HF_HOME"] = "/mnt/data-970-plus/hf_cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

import config as C

app = FastAPI(title="GraphRAG GPU Daemon (config-driven, model-agnostic)")

_jina = None
_reranker = None
_gliner = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_all():
    """Preload the auxiliary models onto the GPU at startup.

    Model identities are read from config.py at startup. To swap any component,
    change the corresponding constant in config.py and restart the daemon:

        EMBED_MODEL_NAME  → embedder   (re-ingest if VECTOR_DIM changes)
        RERANK_MODEL_NAME → reranker   (restart only)
        GLINER_MODEL_NAME → NER model  (restart only)

    GLiNER is lazy-loaded on first /extract_graph call to save ~1.6 GB VRAM
    when it's not needed (E2B LLM is the primary extractor).
    """
    global _jina, _reranker
    from sentence_transformers import SentenceTransformer, CrossEncoder

    print(f"[gpu-daemon] loading models on {DEVICE} ...", flush=True)
    print(f"[gpu-daemon]   embed:  {C.EMBED_MODEL_NAME}", flush=True)
    print(f"[gpu-daemon]   rerank: {C.RERANK_MODEL_NAME}", flush=True)

    # Embedding model (identity from config)
    _jina = SentenceTransformer(
        C.EMBED_MODEL_NAME,
        trust_remote_code=True,
        device=DEVICE,
    ).half()

    # Validate embedding dimension matches config
    dim = _jina.get_sentence_embedding_dimension()
    if dim != C.VECTOR_DIM:
        print(
            f"[gpu-daemon] WARNING: EMBED_MODEL output dim={dim} "
            f"but VECTOR_DIM={C.VECTOR_DIM} in config — UPDATE config!",
            flush=True,
        )

    # Reranker (identity from config)
    _reranker = CrossEncoder(C.RERANK_MODEL_NAME, device=DEVICE).half()

    # Warm each model with a tiny dummy pass so first real request is fast.
    _jina.encode(["warm"], task="retrieval.passage")
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


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/embed_late")
def embed_late(req: EmbedLateReq):
    """Late-style embedding: sentence-split, embed each with retrieval.passage."""
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', req.text)
                 if s.strip()]
    if not sentences:
        sentences = [req.text.strip()]
    vecs = _jina.encode(sentences, task="retrieval.passage",
                        convert_to_numpy=True, show_progress_bar=False)
    return {
        "doc_id": req.doc_id,
        "chunks": [
            {"chunk_idx": i, "vector": vecs[i].tolist(), "text": sentences[i]}
            for i in range(len(sentences))
        ],
    }


@app.post("/embed_query")
def embed_query(req: EmbedQueryReq):
    vec = _jina.encode([req.text], task="retrieval.query",
                       convert_to_numpy=True, show_progress_bar=False)[0]
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


@app.post("/ask")
def ask(req: AskReq):
    """Full RAG answer in ONE call with semantic cache + route metadata."""
    import ask as rag
    import numpy as np
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    # 1. Embed query in-process (resident model on GPU)
    vec = _jina.encode([req.query], task="retrieval.query",
                       convert_to_numpy=True, show_progress_bar=False)[0].tolist()

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
    q_res, g_res = [], []
    t1 = threading.Thread(target=lambda: q_res.extend(rag.qdrant_search(vec, req.query)))
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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "cuda_alloc_gb": round(torch.cuda.memory_allocated()/1e9, 1),
    }


@app.get("/models")
def models():
    """Return currently loaded model identities (from config.py).

    Use this to verify which models the daemon loaded at startup.
    Swap any by changing config.py and restarting.
    """
    return {
        "embedder": {
            "name": C.EMBED_MODEL_NAME,
            "dim": C.VECTOR_DIM,
        },
        "reranker": C.RERANK_MODEL_NAME,
        "gliner": C.GLINER_MODEL_NAME,
        "extraction_llm": C.EXTRACTION_LLM_MODEL,
        "synthesis_llm": C.SYNTHESIS_LLM_MODEL,
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
