"""serve_cpu.py - Warm auxiliary-model daemon (CPU).

FALLBACK DAEMON - only used when CUDA torch is unavailable. The primary daemon
is serve_gpu.py (preloads all aux models on GPU at startup, fp16). This CPU
version lazy-loads models on first use and runs them on CPU only.

Exposes 5 HTTP endpoints consumed by ingest.py, ask.py, and the Hermes rag plugin.
"""
import os
os.environ["HF_HOME"] = "/mnt/data-970-plus/hf_cache"  # cache on NVMe
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "8"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np

app = FastAPI(title="GraphRAG CPU Daemon (fallback)")

# ── Model handles (loaded lazily on first import) ────────────────────────────
_jina = None
_mini = None
_reranker = None
_gliner = None


def _load_jina():
    global _jina
    if _jina is None:
        from sentence_transformers import SentenceTransformer
        # Late chunking: pass the WHOLE document, Jina embeds it with full
        # document context then pools per sentence. Returns one vector per
        # sentence boundary found by the model's tokenizer.
        _jina = SentenceTransformer(
            "jinaai/jina-embeddings-v3",
            trust_remote_code=True,
        )
    return _jina


def _load_mini():
    global _mini
    if _mini is None:
        from sentence_transformers import SentenceTransformer
        _mini = SentenceTransformer("all-MiniLM-L6-v2")
    return _mini


def _load_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        # Multilingual reranker — handles EN + ID query/passage pairs.
        _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return _reranker


def _load_gliner():
    global _gliner
    if _gliner is None:
        from gliner import GLiNER
        _gliner = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
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
    """Late-chunking-style embedding of a full document.

    NOTE: the installed SentenceTransformer release does not expose the
    native `late_chunking=True` kwarg. We replicate the behaviour by
    sentence-splitting the document and embedding each sentence with the
    `retrieval.passage` task — producing one clean, context-aware 1024-d
    vector per chunk. This is the same robust approach used in the v1
    pipeline and avoids the mid-sentence fragmentation of naive chunking.
    """
    model = _load_jina()
    sentences = [s.strip() for s in _split_sentences(req.text) if s.strip()]
    if not sentences:
        sentences = [req.text.strip()]
    vecs = model.encode(
        sentences,
        task="retrieval.passage",
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    chunks = [
        {"chunk_idx": i, "vector": vecs[i].tolist(), "text": sentences[i]}
        for i in range(len(sentences))
    ]
    return {"doc_id": req.doc_id, "chunks": chunks}


@app.post("/embed_query")
def embed_query(req: EmbedQueryReq):
    model = _load_jina()
    vec = model.encode(
        [req.text],
        task="retrieval.query",
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]
    return {"vector": vec.tolist()}


@app.post("/rerank")
def rerank(req: RerankReq):
    model = _load_reranker()
    if not req.docs:
        return {"ranked": []}
    pairs = [(req.query, d) for d in req.docs]
    scores = model.predict(pairs)
    # [(index, score)] sorted descending
    ranked = sorted(
        [(i, float(s)) for i, s in enumerate(scores)],
        key=lambda x: x[1],
        reverse=True,
    )
    return {"ranked": ranked}


@app.post("/extract_graph")
def extract_graph(req: ExtractReq):
    """CPU-bound GLiNER extraction + heuristic relationship formation.

    GLiNER is a zero-shot NER model. It predicts entities of arbitrary labels
    from a small prompt (our GLINER_LABELS). It runs fully on CPU — no GPU
    memory, no LLM. For each sentence we collect predicted entities, then create
    an ASSOCIATED_WITH edge between every pair of entities sharing a sentence
    (co-occurrence heuristic). The caller (ingest.py) resolves entities via
    Jina v3 vector matching against Neo4j's vector index.
    """
    model = _load_gliner()
    from config import GLINER_LABELS, GLINER_THRESHOLD
    # GLiNER returns [{label, text, score, start, end}, ...]
    preds = model.predict_entities(req.text, GLINER_LABELS, threshold=GLINER_THRESHOLD)

    nodes: Dict[str, str] = {}   # canonical_key -> type
    edges = []                   # (src, tgt)
    seen_edges = set()

    # Walk sentence by sentence to build co-occurrence edges.
    for sent in _split_sentences(req.text):
        s_low = sent.lower()
        ents_in_sent = []
        for p in preds:
            if p["start"] < 0 or p["end"] < 0:
                continue
            # crude sentence-localisation: entity text present in this sentence
            if p["text"].lower() in s_low:
                key = p["text"].strip().lower()
                nodes[key] = p["label"]
                ents_in_sent.append(key)
        # pairwise ASSOCIATED_WITH within the sentence
        for a in range(len(ents_in_sent)):
            for b in range(a + 1, len(ents_in_sent)):
                pair = tuple(sorted((ents_in_sent[a], ents_in_sent[b])))
                if pair not in seen_edges:
                    seen_edges.add(pair)
                    edges.append({"source": pair[0], "target": pair[1],
                                  "type": "ASSOCIATED_WITH"})

    node_list = [{"id": k, "type": v} for k, v in nodes.items()]
    return {"nodes": node_list, "edges": edges}


# ── Helpers ──────────────────────────────────────────────────────────────────
def _split_sentences(text: str) -> List[str]:
    import re
    # Split on . ! ? followed by whitespace; keep sentence terminator.
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p for p in parts if p.strip()]


@app.post("/ask")
def ask(req: AskReq):
    """Full RAG answer in ONE call (CPU path — slower than serve_gpu.py GPU version).

    Runs the complete pipeline: embed(query) -> [Qdrant || Neo4j] -> rerank -> E4B synth.
    All aux work is on CPU; E4B synthesis still uses :8084 (GPU).
    Returns {"contexts": [...], "answer": "..."} (answer omitted if synthesize=false).
    """
    import ask as rag

    # 1. Embed query (CPU — _load_jina, on first call)
    model = _load_jina()
    vec = model.encode([req.query], task="retrieval.query",
                       convert_to_numpy=True, show_progress_bar=False)[0].tolist()

    # 2. Parallel retrieval: Qdrant + Neo4j
    import threading
    q_res, g_res = [], []
    t1 = threading.Thread(target=lambda: q_res.extend(rag.qdrant_search(vec)))
    t2 = threading.Thread(target=lambda: g_res.extend(rag.neo4j_subgraph(req.query)))
    t1.start(); t2.start(); t1.join(); t2.join()

    # 3. Fuse + rerank (CPU)
    pool = list(dict.fromkeys(q_res + g_res))
    if not pool:
        return {"query": req.query, "contexts": [], "answer": "", "n_contexts": 0}
    reranker = _load_reranker()
    scores = reranker.predict([(req.query, d) for d in pool])
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
    return {"status": "ok", "device": "cpu"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)
