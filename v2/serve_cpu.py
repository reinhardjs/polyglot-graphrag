"""serve_cpu.py — CPU auxiliary-model daemon (model-agnostic, config-driven).

FALLBACK DAEMON — used when CUDA torch is unavailable (serve_gpu.py is primary).
Models are lazy-loaded on first use; identities and behavior flags come from
config.py (same as serve_gpu.py — swap by changing config, restart daemon).

Exposes the same endpoints as serve_gpu.py: /health, /models, /embed_late,
/embed_query, /rerank, /extract_graph, /ask.
"""
import os
os.environ["HF_HOME"] = "/mnt/data-970-plus/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "8"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np

import config as C

app = FastAPI(title="GraphRAG CPU Daemon (config-driven, model-agnostic)")

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
        if C.EMBED_USE_HALF:
            try:
                _jina = _jina.half()
            except Exception:
                pass
        # Warm
        ekw = {}
        if C.EMBED_TASK_PASSAGE:
            ekw["task"] = C.EMBED_TASK_PASSAGE
        _jina.encode(["warm"], **ekw)
        print(f"[cpu-daemon] embedder loaded", flush=True)
    return _jina


def _load_reranker():
    """Lazy-load the reranker (identity from config)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print(f"[cpu-daemon] loading reranker: {C.RERANK_MODEL_NAME} ...",
              flush=True)
        _reranker = CrossEncoder(C.RERANK_MODEL_NAME)
        if C.RERANK_USE_HALF:
            try:
                _reranker = _reranker.half()
            except Exception:
                pass
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
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', req.text)
                 if s.strip()]
    if not sentences:
        sentences = [req.text.strip()]
    vecs = _embed_encode(sentences, passage=True)
    return {
        "doc_id": req.doc_id,
        "chunks": [
            {"chunk_idx": i, "vector": vecs[i].tolist(), "text": sentences[i]}
            for i in range(len(sentences))
        ],
    }


@app.post("/embed_query")
def embed_query(req: EmbedQueryReq):
    vec = _embed_encode([req.text], passage=False)[0]
    return {"vector": vec.tolist()}


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

    # 1. Embed query (CPU, lazy-load on first call)
    vec = _embed_encode([req.query], passage=False)[0].tolist()

    # 2. Parallel retrieval: Qdrant + Neo4j
    q_res, g_res = [], []
    t1 = threading.Thread(target=lambda: q_res.extend(rag.qdrant_search(vec, req.query)))
    t2 = threading.Thread(target=lambda: g_res.extend(rag.neo4j_subgraph(req.query)))
    t1.start(); t2.start(); t1.join(); t2.join()

    # 3. Fuse + rerank (CPU)
    pool = list(dict.fromkeys(q_res + g_res))
    if not pool:
        return {"query": req.query, "n_contexts": 0, "contexts": [], "answer": ""}
    reranker = _load_reranker()
    scores = reranker.predict([(req.query, d) for d in pool])
    ranked = sorted([(i, float(s)) for i, s in enumerate(scores)],
                    key=lambda x: x[1], reverse=True)[:req.top_k]
    contexts = [pool[i] for i, _ in ranked]

    # 4. Synthesize with E4B (:8084)
    answer = ""
    if req.synthesize:
        answer = rag.synthesize(req.query, contexts)

    return {"query": req.query, "n_contexts": len(contexts),
            "contexts": contexts, "answer": answer}


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
            "name": C.RERANK_MODEL_NAME,
            "half": C.RERANK_USE_HALF,
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
