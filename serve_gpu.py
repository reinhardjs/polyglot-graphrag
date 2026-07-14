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
os.environ["HF_HOME"] = os.environ.get(
    "HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "hf"))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import threading
import importlib
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks, Response
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

import config as C

app = FastAPI(title="GraphRAG GPU Daemon (config-driven, model-agnostic)")

_jina = None
_reranker = None
_gliner = None
# GLiNER is NOT thread-safe: concurrent predict_entities() calls corrupt its
# internal state and raise KeyError. Serialize all GLiNER inference behind
# this lock (sliding_window fires several /extract_entities calls in parallel).
import threading
_gliner_lock = threading.Lock()

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
class EmbedQueryReq(BaseModel):
    text: str
    domain: Optional[str] = None         # v2.6.0 profile (engineering/medical/...)


class RerankReq(BaseModel):
    query: str
    docs: List[str]


class ExtractReq(BaseModel):
    text: str
    labels: Optional[List[str]] = None   # v2.6.0: domain GLiNER entity labels


class AskReq(BaseModel):
    query: str
    top_k: int = 5
    synthesize: bool = True
    skip_cache: bool = False
    collection: Optional[str | List[str]] = None   # Qdrant collection(s); None→default
    domain: Optional[str] = None                     # v2.6.0 profile (engineering/medical/legal/...)
    crag: bool = False                               # Phase 3: Corrective RAG + adaptive routing
    dual_signal_only: bool = False                  # hybrid: keep only candidates confirmed by
                                                    # BOTH signals (SNOMED term-match + clinical prose)
    mode: Optional[str] = None                      # "differential" -> ranked Dx w/ confidence,
                                                    # dual-signal preference, synthesis forced on


class IngestReq(BaseModel):
    text: str
    doc_id: str
    doc_type: str = "eng"
    author: str = "unknown"
    extract_graph: bool = True
    if_checksum: Optional[str] = None   # if provided, skip if unchanged
    collection: Optional[str] = None     # Qdrant collection; None→default engineering_chunks
    domain: Optional[str] = None         # v2.6.0 profile (engineering/medical/...)
    metadata: Optional[Dict[str, Any]] = None  # v2.6.0 REQ-7 domain metadata


class EmbedLateReq(BaseModel):
    text: str
    doc_id: str = "doc"
    strategy: str = "sentence"          # v2.6.0: sentence|paragraph|section|fixed
    chunk_size: int = 512               # max tokens per chunk (token-estimate)
    overlap: int = 64                   # token overlap between chunks
    header_prefix: str = "##"          # for section strategy


@app.post("/embed_late")
def embed_late(req: EmbedLateReq):
    """Late-style embedding: chunk per the requested strategy, embed each.

    Chunking is dispatched via chunking.py (v2.6.0 REQ-3) so it is testable
    and domain-configurable. Uses config.EMBED_TASK_PASSAGE for the encode task.
    """
    import chunking as CH
    chunks = CH.chunk_text(req.text, strategy=req.strategy,
                           chunk_size=req.chunk_size, overlap=req.overlap,
                           header_prefix=req.header_prefix)
    if not chunks:
        chunks = [req.text.strip()]
    encode_kw = {"convert_to_numpy": True, "show_progress_bar": False}
    if C.EMBED_TASK_PASSAGE:
        encode_kw["task"] = C.EMBED_TASK_PASSAGE
    vecs = _jina.encode(chunks, **encode_kw)
    return {
        "doc_id": req.doc_id,
        "chunks": [
            {"chunk_idx": i, "vector": vecs[i].tolist(), "text": chunks[i]}
            for i in range(len(chunks))
        ],
    }


@app.post("/embed_query")
def embed_query(req: EmbedQueryReq):
    # Phase 1: domain-agnostic query modulation (symbolic pre-filtering) —
    # expands known aliases before embedding so Jina v3 grounds slang correctly.
    from query_modulator import QueryModulator
    text = QueryModulator.moderate(req.text, domain=req.domain)
    encode_kw = {"convert_to_numpy": True, "show_progress_bar": False}
    if C.EMBED_TASK_QUERY:
        encode_kw["task"] = C.EMBED_TASK_QUERY
    vec = _jina.encode([text], **encode_kw)[0]
    return {"vector": vec.tolist(), "text_used": text}


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

    `labels` (v2.6.0): override the entity label set with the domain profile's
    graph_schema.entity_types for domain-specific extraction.
    """
    import re
    gliner = _load_gliner()
    labels = req.labels or C.GLINER_LABELS
    with _gliner_lock:
        try:
            preds = gliner.predict_entities(req.text, labels,
                                            threshold=C.GLINER_THRESHOLD)
        except Exception as e:
            # GLiNER can raise KeyError on certain spans (id_to_class_i bug).
            # Degrade gracefully — return no entities rather than 500 so the
            # caller (sliding_window) can continue with the other windows.
            print(f"[gpu-daemon] GLiNER predict failed: {e}", flush=True)
            preds = []
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


@app.post("/extract_entities")
def extract_entities(req: ExtractReq):
    """GLiNER zero-shot NER → entities with character spans (Index-Routing Stage 1).

    Returns entities with exact `start`/`end` character offsets so the caller
    can build entity pairs (all_pairs or same_sentence) and route them to the
    Qwen index-router LLM for typed relation classification.

    `labels` (v2.6.0): domain GLiNER entity labels from domain_config.yaml
    graph_schema.entity_types. If omitted, falls back to config.GLINER_LABELS.
    """
    gliner = _load_gliner()
    labels = req.labels or C.GLINER_LABELS
    with _gliner_lock:
        try:
            preds = gliner.predict_entities(req.text, labels,
                                            threshold=C.GLINER_THRESHOLD)
        except Exception as e:
            # GLiNER can raise KeyError on certain spans (id_to_class_i bug).
            # Degrade gracefully — return no entities rather than 500 so the
            # caller (sliding_window) can continue with the other windows.
            print(f"[gpu-daemon] GLiNER predict failed: {e}", flush=True)
            preds = []
    entities = []
    seen = set()
    for p in preds:
        name = p["text"].strip()
        key = (name.lower(), p["label"])
        if key in seen:
            continue
        seen.add(key)
        entities.append({
            "name": name,
            "type": p["label"],
            "start": int(p.get("start", 0)),
            "end": int(p.get("end", len(name))),
        })
    return {"entities": entities}


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

    # 0. Domain profile (v0.x YAML) — resolves collection/label/entry strategy
    import domain_loader
    profile = domain_loader.get_domain(req.domain) if req.domain else None

    # 1. Embed query in-process (resident model on GPU)
    #    Phase 1: modulate first (expand domain aliases) so embedding is grounded.
    from query_modulator import QueryModulator
    query_text = QueryModulator.moderate(req.query, domain=req.domain)
    encode_kw = {"convert_to_numpy": True, "show_progress_bar": False}
    if C.EMBED_TASK_QUERY:
        encode_kw["task"] = C.EMBED_TASK_QUERY
    vec = _jina.encode([query_text], **encode_kw)[0].tolist()

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
                        "contexts_numbered": p.get("contexts_numbered",
                                                  [f"[{i+1}] {c}" for i, c in
                                                   enumerate(p.get("contexts", []))]),
                        "contexts_meta": p.get("contexts_meta", []),
                        "sources": p.get("sources", []),
                        "rerank_scores": p.get("rerank_scores", []),
                        "answer": p.get("answer", ""),
                        "cache_hit": True,
                    }
        except Exception:
            pass

    # 2b. CRAG mode (Phase 3): adaptive routing + corrective retrieval.
    if req.crag:
        import crag_pipeline as CR
        result = CR.run_crag(req.query, domain=req.domain,
                             synthesize=req.synthesize)
        # shape into the standard /ask response contract
        contexts = result.get("contexts", [])
        source = result.get("route")  # the route decision (graph|hybrid)
        # Reflect retrieval reality in path (consistent with non-crag semantics).
        actual_path = source  # default; would be "hybrid" if corrective pass widened
        if result.get("corrected"):
            actual_path = "hybrid"  # corrective pass always widens to hybrid
        return {
            "query": req.query,
            "source": "crag",
            "path": actual_path,
            "crag_route": source,
            "crag_grade": result.get("grade"),
            "crag_corrected": result.get("corrected"),
            "crag_rewritten_query": result.get("rewritten_query"),
            "crag_trace": result.get("path"),
            "n_contexts": len(contexts),
            "contexts": contexts,
            "contexts_numbered": [f"[{i+1}] {c}" for i, c in enumerate(contexts)],
            "sources": {},
            "contexts_meta": [],
            "answer": result.get("answer") or "",
        }
    # 2b-. SNOMED domain: terminology graph term-match + companion prose.
    # Initialise so both branches leave these bound (Pyright-friendly).
    q_res, g_res, qdrant_hits, graph_hits, path = [], [], 0, 0, "none"
    if req.domain == "snomed":
        import json as _json
        from domains import get_retriever

        # Detect a temporal/graduated presentation: JSON {"day1":[...],...}
        presentation = None
        q = req.query.strip()
        if q.startswith("{") or q.startswith("day") or "day1" in q.lower():
            try:
                parsed = _json.loads(q) if q.startswith("{") else None
                if isinstance(parsed, dict):
                    presentation = parsed
            except Exception:
                presentation = None

        # Primary: SNOMED term-match (IDF) over the concept graph.
        snomed_retrieve = get_retriever("snomed", temporal=bool(presentation))
        if presentation:
            g_res = snomed_retrieve(presentation, top_k=req.top_k)
        else:
            g_res = snomed_retrieve(req.query, top_k=req.top_k)
        graph_hits = len(g_res)

        # Companion semantic corpus: clinical_prose (free-text disease
        # descriptions SNOMED lacks). Merged into q_res so the reranker sees
        # both signals. Silent if the corpus is empty / not yet ingested.
        q_res = []
        qdrant_hits = 0
        try:
            prose_retrieve = get_retriever("clinical_prose")
            prose_hits = prose_retrieve(req.query, top_k=req.top_k)
            if prose_hits:
                # tag each prose record with its signal so the fusion step can
                # boost candidates that ALSO appear in the SNOMED graph set.
                for h in prose_hits:
                    h["_signal"] = "prose"
                q_res = prose_hits
                qdrant_hits = len(prose_hits)
        except Exception as e:
            print(f"[ask] snomed companion prose unavailable: {e}", flush=True)

        # tag each SNOMED graph record with its signal
        for g in g_res:
            if isinstance(g, dict):
                g["_signal"] = "snomed"
            else:
                g = {"text": g, "doc_id": "", "doc_type": "snomed",
                     "chunk_idx": -1, "_signal": "snomed"}

        path = "hybrid" if (graph_hits and qdrant_hits) else (
            "graph" if graph_hits else ("qdrant" if qdrant_hits else "none"))
        # skip the generic Qdrant/Neo4j block below
        _snomed_shortcut = True
    else:
        _snomed_shortcut = False

    # 3. Parallel retrieval: Qdrant + Neo4j
    import threading
    # Resolve collection(s) → list of Qdrant collection names.
    # If no explicit collection given, derive from the domain profile.
    if not _snomed_shortcut:
        if req.collection is None and profile is not None:
            req_collection = profile["collection"]
        else:
            req_collection = req.collection
        collections = _resolve_collections(req_collection)
        q_res, g_res = [], []

        t1 = threading.Thread(
            target=lambda: q_res.extend(rag.qdrant_search_multi(vec, req.query, collections)))
        t2 = threading.Thread(target=lambda: g_res.extend(
            rag.neo4j_subgraph(req.query,
                              label=(profile["neo4j_label"] if profile else None),
                              entry_strategy=(profile.get("entry_strategy", "keyword")
                                              if profile else "keyword"))))
        t1.start(); t2.start(); t1.join(); t2.join()

        qdrant_hits = len(q_res)
        graph_hits = len(g_res)
        path = "hybrid" if (qdrant_hits and graph_hits) else (
            "qdrant" if qdrant_hits else ("graph" if graph_hits else "none")
        )

    # 4. Fuse + rerank in-process
    # q_res = list of Qdrant records {text,doc_id,doc_type,chunk_idx}
    # g_res = list of graph profile strings (no doc_id)
    # Normalise both into records so [n] citations can resolve to a source doc.
    def _as_record(x):
        if isinstance(x, dict):
            return x
        return {"text": x, "doc_id": "", "doc_type": "", "chunk_idx": -1}
    all_records = [_as_record(x) for x in q_res] + [_as_record(x) for x in g_res]
    # Dedupe by (doc_id, text) keeping order; preserve graph (doc_id="")
    seen = set()
    pool = []
    for r in all_records:
        key = (r.get("doc_id", ""), r.get("text", ""))
        if key in seen:
            continue
        seen.add(key)
        pool.append(r)

    # ── Cross-signal (dual-evidence) boost ──────────────────────────────────
    # A candidate is "dual-signal" when BOTH the SNOMED term-match AND the
    # clinical-prose semantic search surface the SAME disease. SNOMED records
    # look like "[DIAGNOSIS CANDIDATE] Measles (disorder) (SNOMED 123)"; prose
    # records like "[CLINICAL PROSE] Measles\n...". We normalise the leading
    # title and treat a candidate as dual when the other signal's normalised
    # name is contained in (or contains) this one. Marked records get a rerank
    # bump so both the score AND the prompt preference align.
    def _norm_name(text: str) -> str:
        t = text.replace("[DIAGNOSIS CANDIDATE]", "").replace("[CLINICAL PROSE]", "")
        t = t.strip()
        # take the first line / the text before " (SNOMED" or newline
        t = t.split("\n")[0].split("(SNOMED")[0].strip().lower()
        return t

    snomed_norm = {_norm_name(r["text"]) for r in all_records
                   if r.get("_signal") == "snomed"}
    prose_norm = {_norm_name(r["text"]) for r in all_records
                  if r.get("_signal") == "prose"}

    def _is_dual(r: dict) -> bool:
        n = _norm_name(r["text"])
        if not n:
            return False
        others = prose_norm if r.get("_signal") == "snomed" else snomed_norm
        return any((n in o or o in n) and o for o in others)

    DUAL_BOOST = 0.15  # additive to rerank score (0..1 scale)
    for r in pool:
        r["_dual_signal"] = _is_dual(r)

    # Optional hard filter: keep only candidates confirmed by BOTH signals.
    # Only meaningful on the SNOMED hybrid path (graph + prose present) and
    # when the caller opts in. Never silently drops everything if nothing is
    # dual — fall back to the full pool so /ask still returns results.
    # mode="differential" implies the same preference (dual-signal candidates
    # are boosted + ranked first), without hard-dropping single-signal ones.
    _force_differential = (req.mode == "differential")
    if (req.dual_signal_only or _force_differential) and path == "hybrid":
        dual_pool = [r for r in pool if r.get("_dual_signal")]
        if req.dual_signal_only and dual_pool:
            # hard filter only when explicitly requested
            pool = dual_pool

    if not pool:
        return {
            "query": req.query, "source": "llm", "path": path,
            "qdrant_hits": qdrant_hits, "graph_hits": graph_hits,
            "n_contexts": 0, "contexts": [], "contexts_numbered": [],
            "contexts_meta": [], "sources": [], "rerank_scores": [],
            "answer": "", "cache_hit": False,
        }

    scores = _reranker.predict([(req.query, r["text"]) for r in pool])
    ranked = []
    for i, s in enumerate(scores):
        sc = float(s)
        if pool[i].get("_dual_signal"):
            sc += DUAL_BOOST
        ranked.append((i, sc))
    ranked.sort(key=lambda x: x[1], reverse=True)
    ranked = ranked[:req.top_k]

    # Confidence label per ranked differential. Computed on the DE-BOOSTED
    # (raw) cross-encoder score so confidence reflects true model relevance,
    # NOT our dual-signal heuristic prior. dual_signal remains a separate
    # corroboration flag (see contexts_meta / diagnoses).
    # Bands calibrated on bge-reranker-v2-m3 over 12 clinical queries
    # (docs/rerank-calibration.md): noise/weak < 0.15, moderate 0.15-0.5,
    # strong >= 0.5.
    def _confidence(raw_score: float) -> str:
        if raw_score >= 0.5:
            return "high"
        if raw_score >= 0.15:
            return "medium"
        return "low"

    contexts = [pool[i]["text"] for i, _ in ranked]
    contexts_meta = [{"doc_id": pool[i].get("doc_id", ""),
                      "doc_type": pool[i].get("doc_type", ""),
                      "chunk_idx": pool[i].get("chunk_idx", -1),
                      "dual_signal": pool[i].get("_dual_signal", False),
                      "confidence": _confidence(
                          float(ranked[idx][1]) - (DUAL_BOOST
                              if pool[i].get("_dual_signal") else 0.0))}
                     for idx, (i, _) in enumerate(ranked)]
    rerank_scores = [s for _, s in ranked]

    # 5. Synthesize with E4B (separate process)
    answer = ""
    if req.synthesize or _force_differential:
        # differential mode forces synthesis so a ranked Dx is always returned
        answer = rag.synthesize(req.query, contexts, profile)

    # Numbered contexts (citation-ready) — present whether or not synthesis
    # runs, so synthesize:false consumers (Hermes rag_query, etc.) get [1]/[2]
    # markers that match what E4B would cite.
    contexts_numbered = [f"[{i+1}] {c}" for i, c in enumerate(contexts)]

    # Bibliography: map each [n] -> source doc. Dedup by doc_id; assign the
    # [n] number of first appearance. Graph-only hits have doc_id="" -> "graph".
    sources = []
    ref_by_doc = {}
    for i, meta in enumerate(contexts_meta):
        did = meta["doc_id"] or "__graph__"
        if did not in ref_by_doc:
            ref_by_doc[did] = len(sources) + 1
            sources.append({
                "ref": len(sources) + 1,
                "doc_id": meta["doc_id"],
                "doc_type": meta["doc_type"],
                "chunk_idx": meta["chunk_idx"],
                "dual_signal": meta.get("dual_signal", False),
                "confidence": meta.get("confidence", "low"),
            })

    # Differential summary: ranked list of candidate diagnoses with confidence,
    # derived from the contexts_meta + rerank_scores (no text parsing needed).
    def _dx_name(text: str) -> str:
        t = text.replace("[DIAGNOSIS CANDIDATE]", "").replace("[CLINICAL PROSE]", "")
        t = t.split("\n")[0].split("(SNOMED")[0].strip()
        return t[:120]
    differential = [{
        "rank": i + 1,
        "candidate": _dx_name(c),
        "confidence": contexts_meta[i].get("confidence", "low"),
        "dual_signal": contexts_meta[i].get("dual_signal", False),
        "rerank_score": round(rerank_scores[i], 4),
    } for i, c in enumerate(contexts)]

    result = {
        "query": req.query,
        "source": "llm",
        "path": path,
        "mode": req.mode,
        "qdrant_hits": qdrant_hits,
        "graph_hits": graph_hits,
        "n_contexts": len(contexts),
        "contexts": contexts,
        "contexts_numbered": contexts_numbered,
        "contexts_meta": contexts_meta,
        "sources": sources,
        "diagnoses": differential,
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
                            "contexts_numbered": contexts_numbered,
                            "contexts_meta": contexts_meta,
                            "sources": sources,
                            "rerank_scores": rerank_scores,
                            "answer": answer,
                        },
                    )],
                    wait=True,
                )
        except Exception:
            pass

    return result


@app.post("/ingest_domain")
def ingest_domain(req: dict):
    """Trigger a domain-specific ingestor (e.g. clinical_prose Wikipedia corpus).

    Body: {"domain": "clinical_prose", "limit": 500, "seed": [...optional...]}
    Runs domains/<domain>/ingest.py::ingest(...) in a background thread and
    returns immediately. The ingestor embeds via the resident daemon, so no
    second model load. Progress is logged to the daemon stdout.
    """
    import threading
    domain = (req or {}).get("domain")
    if not domain:
        raise HTTPException(status_code=400, detail="domain is required")
    try:
        from domains import list_domains
        if domain not in list_domains():
            raise HTTPException(status_code=404,
                                detail=f"no ingestor for domain '{domain}'")
        mod = importlib.import_module(f"domains.{domain}.ingest")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    limit = int((req or {}).get("limit", 500))
    seed = (req or {}).get("seed")

    def _run():
        try:
            mod.ingest(limit=limit, seed=seed)
        except Exception as e:
            print(f"[ingest_domain] {domain} failed: {e}", flush=True)

    threading.Thread(target=_run, daemon=True).start()
    return {"accepted": True, "domain": domain, "limit": limit}


@app.post("/ingest")
def ingest(req: IngestReq, background_tasks: BackgroundTasks, response: Response):
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

    # Derive collection from domain config when not explicitly given.
    target_collection = req.collection
    if target_collection is None and req.domain:
        try:
            import domain_loader
            target_collection = domain_loader.get_domain(req.domain).get("collection")
        except Exception:
            target_collection = None
    target_collection = target_collection or C.COLL_CHUNKS

    # ── Synchronous incremental guard: skip if unchanged ──
    # Returns 304 immediately (no background task, <10ms) when the caller's
    # checksum matches the stored Qdrant payload. This keeps the API contract
    # from the spec: same doc → 304, zero extraction cost.
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

    task_id = _uuid.uuid4().hex
    with _ingest_tasks_lock:
        _ingest_tasks[task_id] = {
            "doc_id": req.doc_id, "status": "queued", "result": None,
            "error": None,
        }

    def _run():
        import ingest as ingest_mod
        import config as C
        with _ingest_tasks_lock:
            _ingest_tasks[task_id]["status"] = "running"
        try:
            res = ingest_mod.ingest_text(
                text=req.text,
                doc_id=req.doc_id,
                meta=({"doc_type": req.doc_type, "author": req.author}
                      | (req.metadata or {})),
                extract_graph=req.extract_graph,
                if_checksum=None,  # already handled synchronously above
                collection=target_collection,
                domain=req.domain,
                metadata=req.metadata,
            )
            with _ingest_tasks_lock:
                _ingest_tasks[task_id]["status"] = "done"
                _ingest_tasks[task_id]["result"] = res
        except Exception as e:
            with _ingest_tasks_lock:
                _ingest_tasks[task_id]["status"] = "error"
                _ingest_tasks[task_id]["error"] = str(e)

    background_tasks.add_task(_run)
    response.status_code = 202
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


@app.delete("/ingest/{doc_id:path}")
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
            with_payload=["doc_id", "doc_type", "checksum", "chunk_idx",
                          "metadata"],
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
                    "metadata": pl.get("metadata", {}),
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


@app.get("/domains")
def list_domains_status():
    """Per-domain status: configured vs runnable, collection point counts,
    ingestor availability, and companion-domain wiring.

    A domain is:
      * configured  — declared in domain_config.yaml
      * runnable    — has a domains/<name>/retrieve.py on disk
      * ingestable  — has a domains/<name>/ingest.py on disk
    Plus live Qdrant point counts for any declared collection.
    """
    from domains import list_domains, supported_domains
    import ingest as ingest_mod
    import domain_loader  # config loader (default_domain lives in domain_config.yaml)

    qc = ingest_mod.get_qdrant()
    def _pts(coll):
        if not coll:
            return None
        try:
            return qc.get_collection(coll).points_count or 0
        except Exception:
            return 0

    configured = set(supported_domains())
    runnable = set(list_domains())
    # discover ingestable domains by probing for ingest.py on disk
    domains_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "domains")
    ingestable = set()
    if os.path.isdir(domains_dir):
        for d in os.listdir(domains_dir):
            ip = os.path.join(domains_dir, d, "ingest.py")
            if os.path.isfile(ip):
                ingestable.add(d)

    # merge all known domain names
    all_names = configured | runnable | ingestable
    out = []
    for name in sorted(all_names):
        profile = None
        try:
            from domain_loader import get_domain
            profile = get_domain(name)
        except Exception:
            profile = None
        coll = (profile or {}).get("collection")
        out.append({
            "name": name,
            "configured": name in configured,
            "runnable": name in runnable,
            "ingestable": name in ingestable,
            "retrieval": (profile or {}).get("retrieval"),
            "collection": coll,
            "points": _pts(coll),
            "prose_domain": (profile or {}).get("prose_domain"),
        })
    return {"domains": out, "default_domain": (domain_loader.get_default_domain()
                                                if hasattr(domain_loader, "get_default_domain")
                                                else "engineering")}


@app.get("/profiles")
def list_profiles():
    """List all loaded domain profiles from domain_config.yaml (YAML)."""
    import domain_loader
    out = []
    for name in domain_loader.list_domains():
        d = domain_loader.get_domain(name)
        out.append({
            "domain": name,
            "collection": d.get("collection"),
            "neo4j_label": d.get("neo4j_label"),
            "description": d.get("description", ""),
            "chunking": (d.get("chunking", {}) or {}).get("strategy", "sentence"),
            "entry_strategy": d.get("entry_strategy", "keyword"),
        })
    return {"profiles": out}


# ── V3.0 Admin endpoints (YAML domain config) ────────────────────────────────
@app.get("/admin/domains")
def admin_domains():
    """List domains from domain_config.yaml (V3.0 YAML-driven config).

    Returns each domain's schema summary: entity/relation types, pair
    strategy, min_confidence, collection, and neo4j_label.
    """
    import domain_loader
    out = []
    for name in domain_loader.list_domains():
        d = domain_loader.get_domain(name)
        out.append({
            "domain": name,
            "collection": d.get("collection"),
            "neo4j_label": d.get("neo4j_label"),
            "entity_types": d.get("entity_types", []),
            "relation_types": d.get("relation_types", []),
            "pair_strategy": d.get("pair_strategy"),
            "min_confidence": d.get("min_confidence"),
        })
    return {
        "domains": out,
        "default_domain": domain_loader.get_default_domain(),
        "count": len(out),
    }


@app.post("/admin/reload")
def admin_reload():
    """Reload domain_config.yaml from disk without restarting the daemon."""
    import domain_loader
    try:
        domain_loader.reload_domains()
        return {
            "status": "reloaded",
            "domains": domain_loader.list_domains(),
            "default_domain": domain_loader.get_default_domain(),
        }
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"reload failed: {e}")


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
