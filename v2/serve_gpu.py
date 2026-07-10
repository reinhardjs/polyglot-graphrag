"""
serve_gpu.py — GPU auxiliary-model daemon (ALL models loaded at startup).

Loads Jina v3, all-MiniLM-L6-v2, BAAI/bge-reranker-v2-m3 and urchade/gliner_multi-v2.1
onto the GPU at process start (device="cuda", fp16) so every request is served from
resident VRAM — no per-request model loading.

VRAM budget on RTX 3060 (12 GB), shared with the LLM endpoints:
    Jina v3 (fp16)        ~5.6 GB
    BGE-reranker-v2-m3     ~1.1 GB
    GLiNER                 ~1.6 GB
    MiniLM                 ~0.1 GB
    -------------------------------
    Aux subtotal           ~8.4 GB   <- this daemon
    + Gemma E2B (:8082)    ~1.5 GB   (separate systemd service, GPU)
    + Gemma E4B (:8084)    ~3.0 GB   (separate systemd service, GPU)
    -------------------------------
    Grand total            ~12.9 GB  -> borderline; close E2B if tight

If you run ONLY this daemon (no E2B/E4B), ~8.4 GB fits comfortably and leaves
room for a larger synthesis model. To load E2B+E4B too, monitor nvidia-smi and
stop one if OOM.

Endpoints (same signatures as serve_cpu.py):
    POST /embed_late    full-doc late-style embedding (per-sentence, retrieval.passage)
    POST /embed_query   single query embedding
    POST /rerank        multilingual BGE rerank
    POST /extract_graph GLiNER zero-shot NER + co-occurrence edges
    POST /ask           ONE-CALL full RAG: embed -> Qdrant||Neo4j -> rerank -> E4B synth
"""
import os
os.environ["HF_HOME"] = "/mnt/data-970-plus/hf_cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # anti-frag
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI(title="GraphRAG GPU Daemon (all aux models on CUDA)")

_jina = None
_mini = None
_reranker = None
_gliner = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_all():
    """Preload the always-on auxiliary models onto the GPU at startup.

    GLiNER is intentionally EXCLUDED here — it is only a fallback for graph
    extraction (E2B LLM is primary), so it is lazy-loaded on first
    /extract_graph call to save ~1.6 GB of resident VRAM.
    """
    global _jina, _mini, _reranker
    from sentence_transformers import SentenceTransformer, CrossEncoder

    print(f"[gpu-daemon] loading models on {DEVICE} ...", flush=True)

    # Jina v3 — late-style embedding (per-sentence, retrieval task)
    _jina = SentenceTransformer(
        "jinaai/jina-embeddings-v3",
        trust_remote_code=True,
        device=DEVICE,
    ).half()

    # MiniLM — zero-shot routing encoder
    _mini = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE).half()

    # BGE multilingual reranker
    _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=DEVICE).half()

    # Warm each model with a tiny dummy pass so first real request is fast.
    _jina.encode(["warm"], task="retrieval.passage")
    _mini.encode(["warm"])
    _reranker.predict([("warm", "warm")])
    print(f"[gpu-daemon] resident models loaded on {DEVICE} "
          f"(GLiNER lazy-loaded on demand). "
          f"Allocated: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)


def _load_gliner():
    """Lazy-load GLiNER on first /extract_graph call (fallback path only)."""
    global _gliner
    if _gliner is None:
        from gliner import GLiNER
        print("[gpu-daemon] lazy-loading GLiNER (fallback) ...", flush=True)
        _gliner = GLiNER.from_pretrained("urchade/gliner_multi-v2.1").to(DEVICE).half()
        _gliner.predict_entities("warm", ["Microservice", "Database"])
        print(f"[gpu-daemon] GLiNER resident. "
              f"Allocated: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
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


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/embed_late")
def embed_late(req: EmbedLateReq):
    """Late-style embedding: sentence-split, embed each with retrieval.passage.

    The installed SentenceTransformer release lacks native late_chunking=True,
    so we replicate it by per-sentence embedding (clean, complete chunks). All
    on GPU — resident, no load latency.
    """
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', req.text) if s.strip()]
    if not sentences:
        sentences = [req.text.strip()]
    vecs = _jina.encode(sentences, task="retrieval.passage",
                        convert_to_numpy=True, show_progress_bar=False)
    return {"doc_id": req.doc_id,
            "chunks": [{"chunk_idx": i, "vector": vecs[i].tolist(),
                        "text": sentences[i]} for i in range(len(sentences))]}


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
    """GPU GLiNER extraction + co-occurrence heuristic edges.

    GLiNER is a zero-shot NER model loaded on CUDA. It predicts entities of our
    GLiNER_LABELS; for each sentence we create ASSOCIATED_WITH edges between
    co-occurring entities. The caller (ingest.py) canonicalises ID->EN.
    """
    from config import GLINER_LABELS, GLINER_THRESHOLD
    import re
    gliner = _load_gliner()   # lazy-load on first use (fallback path)
    preds = gliner.predict_entities(req.text, GLINER_LABELS,
                                    threshold=GLINER_THRESHOLD)
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
                    edges.append({"source": pair[0], "target": pair[1],
                                  "type": "ASSOCIATED_WITH"})
    return {"nodes": [{"id": k, "type": v} for k, v in nodes.items()],
            "edges": edges}


@app.post("/ask")
def ask(req: AskReq):
    """Full RAG answer in ONE call.

    Runs the complete pipeline server-side (reusing ask.py's tested retrieval
    + synthesis) using the resident GPU models for embed/rerank and E4B (:8084)
    for synthesis:
        embed(query) -> [Qdrant || Neo4j] -> rerank -> synthesize.

    No HTTP self-loop for embedding/rerank (uses in-process _jina/_reranker).
    Returns {"contexts": [...], "answer": "..."} (answer omitted if synthesize=false).
    """
    import ask as rag
    import numpy as np

    # 1. Embed query in-process (resident Jina on GPU)
    vec = _jina.encode([req.query], task="retrieval.query",
                       convert_to_numpy=True, show_progress_bar=False)[0].tolist()

    # 2. Parallel retrieval: Qdrant dense+sparse  +  Neo4j k-hop subgraph
    import threading
    q_res, g_res = [], []
    t1 = threading.Thread(target=lambda: q_res.extend(rag.qdrant_search(vec, req.query)))
    t2 = threading.Thread(target=lambda: g_res.extend(rag.neo4j_subgraph(req.query)))
    t1.start(); t2.start(); t1.join(); t2.join()

    # 3. Fuse + rerank in-process (resident BGE on GPU)
    pool = list(dict.fromkeys(q_res + g_res))
    if not pool:
        return {"query": req.query, "contexts": [], "answer": "", "n_contexts": 0}
    scores = _reranker.predict([(req.query, d) for d in pool])
    ranked = sorted([(i, float(s)) for i, s in enumerate(scores)],
                    key=lambda x: x[1], reverse=True)[:req.top_k]
    contexts = [pool[i] for i, _ in ranked]

    # 4. Synthesize with E4B (separate process on :8084)
    answer = ""
    if req.synthesize:
        answer = rag.synthesize(req.query, contexts)

    return {"query": req.query, "n_contexts": len(contexts),
            "contexts": contexts, "answer": answer}


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE,
            "cuda_alloc_gb": round(torch.cuda.memory_allocated()/1e9, 1)}


@app.on_event("startup")
def on_startup():
    _load_all()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)
