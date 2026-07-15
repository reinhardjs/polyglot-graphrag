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
    + E2B (:8082)         ~1.5 GB  (config: EXTRACTION_LLM + SYNTHESIS_LLM)
    + E4B (:8084)         ~3.0 GB  (opt-in via SYNTHESIS_LLM_* env; idle by default)
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
import requests
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

# The Jina embedding model is NOT thread-safe: concurrent .encode() calls share
# the same GPU workspace and collide (CUDA OOM under parallel /ask traffic).
# Serialize all embedding inference behind this lock. The GLiNER lock above is
# a separate model and does NOT cover Jina — both need their own serializer.
_jina_lock = threading.Lock()

# Global ingest serializer. The whole ingest pipeline (delete -> embed via
# /embed_late -> graph extract) touches the shared GPU (Jina) and Neo4j. Even
# with _jina_lock on the embed call, overlapping background ingest tasks push
# resident-model VRAM past 12 GB and intermittently CUDA-OOM. Serializing the
# entire task guarantees zero overlap and a 100% reliable bulk ingest.
_ingest_lock = threading.Lock()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Ingest task store (in-process, for BackgroundTasks polling) ───────────────
# Maps task_id → status dict. Lives for the daemon's lifetime. A production
# deployment would use Redis, but in-process is sufficient for single-daemon.
_ingest_tasks: Dict[str, Dict[str, Any]] = {}
_ingest_tasks_lock = threading.Lock()
import uuid as _uuid

# Domain config loader (domain_config.yaml). Imported at module level so the
# startup seeder and /domains /ask can use it without re-importing.
import domain_loader  # noqa: E402  (kept local elsewhere; promote to module scope


def _safe_seed(mod):
    """Run a domain ingestor, swallowing + logging any failure (non-fatal)."""
    try:
        mod.ingest()
    except Exception as e:
        print(f"[gpu-daemon] seed ingest failed: {e}", flush=True)


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
    # Loaded on RERANK_DEVICE (cpu by default) to keep GPU VRAM free for the
    # Jina embedder + E2B GGUF backend (the only model server; it serves
    # both extraction and synthesis) that share the 12 GB card.
    _reranker = CrossEncoder(C.RERANK_MODEL_NAME, device=C.RERANK_DEVICE)
    if C.RERANK_USE_HALF and C.RERANK_DEVICE != "cpu":
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

    # ── Prebuild SNOMED IDF cache (P1.7) ────────────────────────────────────
    # Done at startup so the FIRST user request doesn't pay the ~20s scan cost.
    # Runs in a background thread so it doesn't block the daemon from serving
    # other domains while the table builds.
    def _prebuild_idf():
        try:
            import domains.snomed.retrieve as S
            print("[gpu-daemon] building SNOMED IDF lookup table ...", flush=True)
            S._ensure_idf_cache()
            print(f"[gpu-daemon] SNOMED IDF cache ready "
                  f"({len(S._IDF_CACHE)} words)", flush=True)
        except Exception as e:
            print(f"[gpu-daemon] IDF prebuild failed (lazy on first use): {e}",
                  flush=True)
    threading.Thread(target=_prebuild_idf, daemon=True).start()


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
    domain: Optional[str | List[str]] = None        # v2.6.0 profile (engineering/medical/legal/...) or ["a","b"] for cross-domain
    crag: bool = False                               # Phase 3: Corrective RAG + adaptive routing
    dual_signal_only: bool = False                  # hybrid: keep only candidates confirmed by
                                                    # BOTH signals (SNOMED term-match + clinical prose)
    mode: Optional[str] = None                      # "differential" -> ranked Dx w/ confidence,
                                                    # dual-signal preference, synthesis forced on
    min_confidence: Optional[str] = None            # "low"|"medium"|"high" -> drop candidates
                                                    # below this confidence from `diagnoses`


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
    import embed_late as EL
    with _jina_lock:
        chunks_out = EL.embed_late_chunks(
            _jina, req.text, strategy=req.strategy,
            chunk_size=req.chunk_size, overlap=req.overlap,
            header_prefix=req.header_prefix, task=C.EMBED_TASK_PASSAGE)
    return {"doc_id": req.doc_id, "chunks": chunks_out}


@app.post("/embed_query")
def embed_query(req: EmbedQueryReq):
    # Phase 1: domain-agnostic query modulation (symbolic pre-filtering) —
    # expands known aliases before embedding so Jina v3 grounds slang correctly.
    from query_modulator import QueryModulator
    text = QueryModulator.moderate(req.text, domain=req.domain)
    encode_kw = {"convert_to_numpy": True, "show_progress_bar": False}
    if C.EMBED_TASK_QUERY:
        encode_kw["task"] = C.EMBED_TASK_QUERY
    with _jina_lock:
        vec = _jina.encode([text], **encode_kw)[0]
    return {"vector": vec.tolist(), "text_used": text}


# OpenAI-compatible embeddings endpoint so external tools (ragas, etc.) can
# use the daemon's already-loaded Jina model without a second embeddings
# server. Accepts {"input": str | list[str], "model": "..."} and returns the
# OpenAI-shaped {"data": [{"embedding": [...]}, ...], "model": "..."}.
class OpenAIEmbedReq(BaseModel):
    input: str | list[str]
    model: Optional[str] = None


@app.post("/v1/embeddings")
def openai_embeddings(req: OpenAIEmbedReq):
    texts = req.input if isinstance(req.input, list) else [req.input]
    from query_modulator import QueryModulator
    mod = [QueryModulator.moderate(t, domain=None) for t in texts]
    encode_kw = {"convert_to_numpy": True, "show_progress_bar": False}
    if C.EMBED_TASK_QUERY:
        encode_kw["task"] = C.EMBED_TASK_QUERY
    with _jina_lock:
        vecs = _jina.encode(mod, **encode_kw)
    return {
        "object": "list",
        "data": [{"object": "embedding", "index": i,
                  "embedding": vecs[i].tolist()} for i in range(len(vecs))],
        "model": req.model or C.EMBED_MODEL_NAME,
    }


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


# ── Collection resolver moved to ask.resolve_collections (shared by daemon) ──

def _persist_differential(req: AskReq, result: dict):
    """Persist a ranked differential to Neo4j for later audit / graph review.

    Creates (idempotent MERGE):
      (:Differential {query, mode, min_confidence, path, created}) -[:HAS_CANDIDATE]->
        (:DxCandidate {rank, candidate, confidence, dual_signal, rerank_score})
    and links each candidate to a matching (:SnomedDescription) if the name
    resolves, via -[:MATCHES_CONCEPT]->. Only runs for mode='differential' and when there is
    at least one diagnosis. Failures are swallowed (diagnostic, never fatal).
    """
    if req.mode != "differential" or not result.get("diagnoses"):
        return
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            C.NEO4J_URI, auth=(C.NEO4J_USER, C.NEO4J_PASSWORD))
        diff_id = abs(hash((req.query, req.mode, req.min_confidence or ""))) % (2 ** 63)
        with driver.session() as s:
            s.run(
                "MERGE (d:Differential {id:$id}) "
                "SET d.query=$q, d.mode=$mode, d.min_confidence=$mc, "
                "    d.path=$path, d.created=datetime()",
                id=diff_id, q=req.query, mode=req.mode,
                mc=req.min_confidence, path=result.get("path"))
            # clear stale candidates then re-add (idempotent on re-run)
            s.run(
                "MATCH (d:Differential {id:$id})-[r:HAS_CANDIDATE]->(c:DxCandidate) "
                "DELETE r, c", id=diff_id)
            for dx in result["diagnoses"]:
                s.run(
                    "MATCH (d:Differential {id:$id}) "
                    "CREATE (c:DxCandidate {rank:$rank, candidate:$name, "
                    "  confidence:$conf, dual_signal:$dual, rerank_score:$score}) "
                    "MERGE (d)-[:HAS_CANDIDATE]->(c) "
                    "WITH c "
                    "MATCH (sd:SnomedDescription) WHERE sd.term CONTAINS $name "
                    "MERGE (c)-[:MATCHES_CONCEPT]->(sd)",
                    id=diff_id, rank=dx["rank"], name=dx["candidate"],
                    conf=dx["confidence"], dual=dx["dual_signal"],
                    score=dx["rerank_score"])
        driver.close()
    except Exception as e:
        print(f"[ask] differential persistence failed: {e}", flush=True)


@app.post("/ask")
def ask(req: AskReq):
    """Full RAG answer in ONE call with semantic cache + route metadata."""
    import time as _time
    import uuid as _uuid_mod
    _req_id = _uuid_mod.uuid4().hex[:12]
    _t0 = _time.time()
    import ask as rag
    import numpy as np
    
    # Truncate each context chunk before synthesis: the generator only needs a
    # capped excerpt to summarize + cite; this slashes prompt prefill time and
    # keeps synthesis under the latency target (E2B ~1s vs ~5.7s untruncated).
    def _synth_contexts(ctxs):
        # cap the number of contexts sent to synthesis (LLM only needs top few)
        n = C.MAX_SYNTH_CONTEXTS
        if n and n > 0:
            ctxs = ctxs[:n]
        cap = C.MAX_SYNTH_CONTEXT_CHARS
        if not cap or cap <= 0:
            return ctxs
        return [c[:cap] for c in ctxs]

    from qdrant_client.models import PointStruct

    # 0. Domain profile (v0.x YAML) — resolves collection/label/entry strategy.
    # Only meaningful for single-domain queries; cross-domain (list / "all")
    # resolves each concrete domain inside the fan-out below.
    import domain_loader
    # Unknown-domain guard (P3.12): return 200 with error + available list,
    # never a 500. Valid targets = any concrete domain, alias, or "all".
    if isinstance(req.domain, str) and req.domain.lower() != "all":
        # Resolve alias to concrete name, then check membership among configured
        # domains (skip aliases so we validate the concrete target).
        concrete = domain_loader._resolve_alias(req.domain)
        configured = [d for d in domain_loader.list_domains()
                      if not isinstance(domain_loader._DOMAINS.get(d), dict)
                      or "alias" not in domain_loader._DOMAINS.get(d, {})]
        if concrete not in configured:
            _available = domain_loader.list_domains()
            return {
                "error": "unknown_domain",
                "available": _available,
                "degraded": True,
                "notice": f"unknown domain '{req.domain}'; available: {', '.join(_available)}",
                "query": req.query,
                "n_contexts": 0,
                "contexts": [],
                "contexts_meta": [],
                "request_id": _req_id,
            }
    profile = domain_loader.get_domain(req.domain) if isinstance(req.domain, str) else None

    # 1. Embed query in-process (resident model on GPU)
    #    Phase 1: modulate first (expand domain aliases) so embedding is grounded.
    from query_modulator import QueryModulator
    query_text = QueryModulator.moderate(
        req.query, domain=req.domain if isinstance(req.domain, str) else None)
    encode_kw = {"convert_to_numpy": True, "show_progress_bar": False}
    if C.EMBED_TASK_QUERY:
        encode_kw["task"] = C.EMBED_TASK_QUERY
    with _jina_lock:
        vec = _jina.encode([query_text], **encode_kw)[0].tolist()

    # 2. SEMANTIC CACHE (P1.5: enabled for ALL queries, not just synthesize)
    # Cache entries are scoped by a variant key covering every flag that changes
    # the OUTPUT (synthesize, domain, mode, min_confidence, top_k). Without this
    # a synthesize=false entry (answer="") would be wrongly served to a later
    # synthesize=true request, and a domain=snomed result to a domain=all one.
    def _cache_scope() -> str:
        dom = req.domain
        if isinstance(dom, list):
            dom = "all:" + ",".join(sorted(str(d).lower() for d in dom))
        else:
            dom = str(dom or "").lower()
        synth = bool(req.synthesize or (req.mode == "differential"))
        return "|".join([
            f"synth={int(synth)}",
            f"domain={dom}",
            f"mode={req.mode or ''}",
            f"minconf={req.min_confidence or ''}",
            f"topk={req.top_k}",
        ])

    _scope = _cache_scope()
    if not req.skip_cache:
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            qc = QdrantClient(url=C.QDRANT_URL, prefer_grpc=False)
            if qc.collection_exists(C.COLL_CACHE):
                hits = qc.query_points(
                    C.COLL_CACHE, query=vec, limit=1,
                    score_threshold=C.CACHE_THRESHOLD, with_payload=True,
                    query_filter=Filter(must=[FieldCondition(
                        key="scope", match=MatchValue(value=_scope))]),
                ).points
                if hits:
                    p = hits[0].payload
                    _record_metric("ok", _time.time() - _t0)
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
                        "diagnoses": p.get("diagnoses", []),
                        "cache_hit": True,
                        "notice": "",
                        "request_id": _req_id,
                    }
        except Exception:
            pass

    # 2b. CRAG mode (Phase 3): adaptive routing + corrective retrieval.
    # CRAG is gated behind config.ENABLE_CRAG (default false).
    # Enable to opt into experimental query rewriting + corrective retrieval.
    if C.ENABLE_CRAG and getattr(req, "crag", False):
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
            "notice": "",
        }
    # 2b-. Federated retrieval assembly (domain-independent).
    # The Mediator no longer hardcodes `if domain == "snomed"`. Composition is
    # driven by domain_config.yaml via federated.assemble: the primary domain
    # (unless it's a pure dense_prose corpus) plus its `companions:` list.
    # Every fetched record is tagged with `_signal` = its domain name, so the
    # fusion step (cross-signal dual-evidence) can reference signals generically.
    import federated as F
    import json as _json

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

    # ── Cross-domain fan-out (P2.11) ───────────────────────────────────────
    # domain="all" (or an explicit list ["legal","enterprise"]) runs every
    # concrete primary domain's retriever in parallel, tags each record with
    # `_domain`, and merges into one pool for fusion/rerank. Aliases are
    # expanded so `all` covers healthcare/enterprise/legal/fraud.
    def _concrete_domains() -> list:
        names = []
        for d in domain_loader.list_domains():
            cfg = _DOMAINS_SAFE(d)
            if isinstance(cfg, dict) and "alias" in cfg:
                continue  # skip alias entries themselves
            names.append(d)
        return names

    def _DOMAINS_SAFE(name):
        try:
            return domain_loader._DOMAINS.get(name)
        except Exception:
            return None

    _domains_requested = req.domain
    if isinstance(_domains_requested, str) and _domains_requested.lower() == "all":
        _target_domains = _concrete_domains()
    elif isinstance(_domains_requested, list):
        _target_domains = _domains_requested
    else:
        _target_domains = None  # single-domain path

    if _target_domains is not None:
        # Cross-domain: assemble each concrete domain in parallel.
        q_res = []
        g_res = []
        _total_q = 0
        _total_g = 0
        _threads = []
        _per_domain = {}
        _failed = []

        def _run_domain(dom):
            sub = F.assemble(
                F.FederateRequest(
                    domain=dom, query=req.query, top_k=req.top_k,
                    presentation=presentation, synthesize=req.synthesize,
                    mode=req.mode, vec=vec,
                ), vec=vec, rag=rag)
            _per_domain[dom] = sub
            _failed.extend(sub.failed)

        for dom in _target_domains:
            t = threading.Thread(target=_run_domain, args=(dom,))
            _threads.append(t)
        for t in _threads:
            t.start()
        for t in _threads:
            t.join()
        for dom, sub in _per_domain.items():
            for r in sub.q_res:
                r["_domain"] = dom
                q_res.append(r)
            _total_q += sub.qdrant_hits
            _total_g += sub.graph_hits
        qdrant_hits = _total_q
        graph_hits = _total_g
        path = "cross_domain"
    else:
        # Single-domain path (original).
        fed = F.assemble(F.FederateRequest(
            domain=req.domain or "",
            query=req.query, top_k=req.top_k,
            presentation=presentation, synthesize=req.synthesize, mode=req.mode,
            vec=vec,
        ), vec=vec, rag=rag)
        q_res = fed.q_res
        g_res = fed.g_res
        qdrant_hits = fed.qdrant_hits
        graph_hits = fed.graph_hits
        path = fed.path
        _failed = list(fed.failed)

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

    # Cross-signal (dual-evidence) boost, GENERIC over signal names:
    # A record is "dual-signal" when the SAME disease is confirmed by BOTH
    # signals. We match by the authoritative SNOMED concept id when both
    # records carry one (clinical_prose chunks now store concept_id; snomed
    # records carry concept_id). This is exact and language-agnostic. As a
    # fallback (when a record lacks concept_id), we fall back to normalised
    # name containment. Works for any domain with >=2 signals.
    by_signal: Dict[str, set] = {}
    by_signal_cid: Dict[str, set] = {}
    for r in all_records:
        sig = r.get("_signal", "")
        by_signal.setdefault(sig, set()).add(_norm_name(r.get("text", "")))
        cid = r.get("concept_id")
        if cid:
            by_signal_cid.setdefault(sig, set()).add(str(cid))

    def _is_dual(r: dict) -> bool:
        n = _norm_name(r.get("text", ""))
        if not n:
            return False
        sig = r.get("_signal", "")
        others = set().union(*[v for k, v in by_signal.items() if k != sig]) \
            if len(by_signal) > 1 else set()
        # Primary: exact concept-id match across signals.
        rcid = r.get("concept_id")
        if rcid:
            other_cids = set().union(
                *[v for k, v in by_signal_cid.items() if k != sig]) \
                if len(by_signal_cid) > 1 else set()
            if str(rcid) in other_cids:
                return True
        # Fallback: normalised name containment across signals.
        return any((n in o or o in n) and o for o in others)

    DUAL_BOOST = C.DUAL_SIGNAL_BOOST  # additive rerank bump for dual-signal candidates
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
            "answer": "", "cache_hit": False, "notice": "",
        }

    # 4b. Conditional rerank (P1.8): when synthesize=false and pool is small,
    # skip the BGE cross-encoder entirely and order by raw retrieval scores.
    # The reranker's main value is ordering for LLM synthesis; for raw context
    # retrieval the per-signal scores (IDF for graph hits, Qdrant similarity for
    # prose hits) are sufficient and deterministic.
    # For synthesize=TRUE we ALSO skip the rerank: the LLM reads all top-k
    # contexts and synthesizes them itself, so pre-reranking only adds latency
    # (the BGE reranker runs on CPU — ~9.6s for a 10-candidate pool — and that
    # cost is pure overhead when synthesis is happening anyway).
    _skip_rerank = req.synthesize or (
        (not _force_differential) and len(pool) <= 10
    )

    if _skip_rerank:
        # Raw-score ordering: graph hits carry IDF score (already summed in
        # _build_context? no — re-derive from context text). Simpler: sort by
        # a normalized pool score. We stored no per-record raw score, so use a
        # stable order: dual-signal first, then preserve arrival order.
        ranked = []
        for i, r in enumerate(pool):
            sc = 0.0
            if r.get("_dual_signal"):
                sc += DUAL_BOOST
            ranked.append((i, sc))
        ranked.sort(key=lambda x: x[1], reverse=True)
        ranked = ranked[:req.top_k]
        rerank_scores = [float(pool[i].get("_dual_signal", False)) for i, _ in ranked]
    else:
        scores = _reranker.predict([(req.query, r["text"]) for r in pool])
        ranked = []
        for i, s in enumerate(scores):
            sc = float(s)
            if pool[i].get("_dual_signal"):
                sc += DUAL_BOOST
            ranked.append((i, sc))
        ranked.sort(key=lambda x: x[1], reverse=True)
        ranked = ranked[:req.top_k]
        rerank_scores = [s for _, s in ranked]

    # Confidence label per ranked differential. Computed on the DE-BOOSTED
    # (raw) cross-encoder score so confidence reflects true model relevance,
    # NOT our dual-signal heuristic prior. dual_signal remains a separate
    # corroboration flag (see contexts_meta / diagnoses).
    # Bands calibrated on bge-reranker-v2-m3 over 12 clinical queries
    # (docs/rerank-calibration.md): noise/weak < 0.15, moderate 0.15-0.5,
    # strong >= 0.5.
    def _confidence(raw_score: float) -> str:
        if raw_score >= C.CONFIDENCE_HIGH_THRESHOLD:
            return "high"
        if raw_score >= C.CONFIDENCE_MEDIUM_THRESHOLD:
            return "medium"
        return "low"

    contexts = [pool[i]["text"] for i, _ in ranked]
    contexts_meta = [{"doc_id": pool[i].get("doc_id", ""),
                     "doc_type": pool[i].get("doc_type", ""),
                     "chunk_idx": pool[i].get("chunk_idx", -1),
                     "signal": pool[i].get("_signal", ""),
                     "domain": pool[i].get("_domain", pool[i].get("_signal", "")),
                     "concept_id": pool[i].get("concept_id"),
                     "dual_signal": pool[i].get("_dual_signal", False),
                     "confidence": _confidence(
                         float(ranked[idx][1]) - (DUAL_BOOST
                             if pool[i].get("_dual_signal") else 0.0))}
                    for idx, (i, _) in enumerate(ranked)]
    rerank_scores = [s for _, s in ranked]

    # 5. Synthesize (E2B, separate process)
    answer = ""
    degraded = bool(_failed)
    if req.synthesize or _force_differential:
        # differential mode forces synthesis so a ranked Dx is always returned
        try:
            answer = rag.synthesize(req.query, _synth_contexts(contexts), profile)
        except Exception as e:
            # Synthesis backend unavailable (P3.12): degrade gracefully — empty answer, flag.
            print(f"[ask] synthesis failed (degraded): {e}", flush=True)
            degraded = True
            answer = ""

    # Numbered contexts (citation-ready) — present whether or not synthesis
    # runs, so synthesize:false consumers (Hermes rag_query, etc.) get [1]/[2]
    # markers that match what the synthesis backend would cite.
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
        "dual_signal": contexts_meta[i].get("_dual_signal", False),
        "rerank_score": round(rerank_scores[i], 4),
    } for i, c in enumerate(contexts)]

    # Optional confidence floor: drop candidates below min_confidence from the
    # diagnoses list. Always keep at least the top result so /ask never returns
    # an empty differential when nothing clears the bar.
    _CONF_RANK = {"low": 0, "medium": 1, "high": 2}
    if req.min_confidence and req.min_confidence in _CONF_RANK:
        floor = _CONF_RANK[req.min_confidence]
        kept = [d for d in differential if _CONF_RANK.get(d["confidence"], 0) >= floor]
        if kept:
            # re-rank indices and trim the parallel context lists to match
            keep_idx = [d["rank"] - 1 for d in kept]
            differential = [{**d, "rank": j + 1} for j, d in enumerate(kept)]
            contexts = [contexts[i] for i in keep_idx]
            contexts_numbered = [f"[{j+1}] {contexts[j]}" for j in range(len(contexts))]
            contexts_meta = [contexts_meta[i] for i in keep_idx]
            rerank_scores = [rerank_scores[i] for i in keep_idx]
            # rebuild sources to reflect kept docs
            sources = []
            ref_by_doc = {}
            for m in contexts_meta:
                did = m["doc_id"] or "__graph__"
                if did not in ref_by_doc:
                    ref_by_doc[did] = len(sources) + 1
                    sources.append({
                        "ref": len(sources) + 1, "doc_id": m["doc_id"],
                        "doc_type": m["doc_type"], "chunk_idx": m["chunk_idx"],
                        "dual_signal": m.get("dual_signal", False),
                        "confidence": m.get("confidence", "low"),
                    })
            if req.synthesize or _force_differential:
                answer = rag.synthesize(req.query, _synth_contexts(contexts), profile)

    # Compose human-readable notice for degraded responses.
    _notice_parts = []
    if _failed:
        _dom_list = ", ".join(_failed)
        _notice_parts.append(
            f"domain(s) '{_dom_list}' failed; results exclude {'it' if len(_failed)==1 else 'them'}")
    if degraded and answer == "" and (req.synthesize or _force_differential):
        _notice_parts.append(
            "synthesis unavailable (backend unreachable); returned retrieval-only results")
    _notice = "; ".join(_notice_parts) if _notice_parts else ""

    result = {
        "query": req.query,
        "source": "llm",
        "path": path,
        "mode": req.mode,
        "min_confidence": req.min_confidence,
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
        "degraded": degraded,
        "failed_domains": _failed,
        "notice": _notice,
        "request_id": _req_id,
    }

    _record_metric("degraded" if degraded else "ok", _time.time() - _t0)

    # 6. Store in cache (P1.5: enabled for ALL queries).
    # Do NOT cache degraded results, or a synthesis that was requested but came
    # back empty (synthesis backend down mid-request) — caching that empty answer would poison
    # every repeat. Retrieval-only calls (synthesize=false) legitimately have an
    # empty answer and MUST still be cached.
    _synth_requested = bool(req.synthesize or _force_differential)
    _synth_failed = _synth_requested and answer == ""
    _cache_worthy = (not degraded) and (not _synth_failed)
    if not req.skip_cache and _cache_worthy:
        try:
            qc = QdrantClient(url=C.QDRANT_URL, prefer_grpc=False)
            if qc.collection_exists(C.COLL_CACHE):
                qc.upsert(
                    C.COLL_CACHE,
                    [PointStruct(
                        id=abs(hash((req.query, _scope))) % (2**63),
                        vector=vec,
                        payload={
                            "query": req.query,
                            "scope": _scope,
                            "path": path,
                            "qdrant_hits": qdrant_hits,
                            "graph_hits": graph_hits,
                            "contexts": contexts,
                            "contexts_numbered": contexts_numbered,
                            "contexts_meta": contexts_meta,
                            "sources": sources,
                            "rerank_scores": rerank_scores,
                            "answer": answer,
                            "diagnoses": differential,
                        },
                    )],
                    wait=True,
                )
        except Exception:
            pass

    # 7. Persist structured differential to Neo4j (audit / graph review).
    persisted = False
    try:
        _persist_differential(req, result)
        persisted = req.mode == "differential" and bool(result.get("diagnoses"))
    except Exception:
        pass
    result["differential_persisted"] = persisted

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
        with _ingest_lock:
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


@app.get("/config")
def show_config():
    """Active runtime calibration + model knobs (read from config.py).

    Lets ops confirm the live confidence bands and dual-signal boost without
    reading source. Mirrors the tunable constants; editing config.py + restart
    updates these.
    """
    import domain_loader  # default_domain lives in domain_config.yaml
    return {
        "reranker": {
            "model": C.RERANK_MODEL_NAME,
            "use_half": C.RERANK_USE_HALF,
        },
        "confidence_bands": {
            "high_threshold": C.CONFIDENCE_HIGH_THRESHOLD,
            "medium_threshold": C.CONFIDENCE_MEDIUM_THRESHOLD,
            "computed_on": "raw (de-boosted) cross-encoder score",
        },
        "dual_signal_boost": C.DUAL_SIGNAL_BOOST,
        "default_domain": (domain_loader.get_default_domain()
                           if hasattr(domain_loader, "get_default_domain")
                           else "engineering"),
        "note": "confidence = model relevance, NOT clinical probability; "
                "dual_signal is a separate corroboration flag. "
                "See docs/rerank-calibration.md.",
    }


@app.get("/differentials")
def list_differentials(query: Optional[str] = None, limit: int = 20):
    """List persisted differentials (audit). Optional ?query= filters by the
    exact query string; ?limit= caps results. Returns the Differential node
    plus its ranked DxCandidate children (with MATCHES_CONCEPT links).
    """
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        C.NEO4J_URI, auth=(C.NEO4J_USER, C.NEO4J_PASSWORD))
    try:
        with driver.session() as s:
            if query:
                diff_id = abs(hash((query, "differential", ""))) % (2 ** 63)
                rec = s.run(
                    "MATCH (d:Differential {id:$id})-[:HAS_CANDIDATE]->(c:DxCandidate) "
                    "RETURN d, collect(c) AS cands", id=diff_id).single()
                if not rec:
                    return {"differentials": []}
                d = rec["d"]
                return {"differentials": [{
                    "id": d["id"], "query": d.get("query"), "mode": d.get("mode"),
                    "min_confidence": d.get("min_confidence"),
                    "path": d.get("path"), "created": str(d.get("created")),
                    "candidates": [dict(c) for c in rec["cands"]],
                }]}
            rows = s.run(
                "MATCH (d:Differential)-[:HAS_CANDIDATE]->(c:DxCandidate) "
                "WITH d, collect(c) AS cands "
                "RETURN d, cands ORDER BY d.created DESC LIMIT $lim",
                lim=limit).data()
            return {"differentials": [{
                "id": r["d"]["id"], "query": r["d"].get("query"),
                "mode": r["d"].get("mode"),
                "min_confidence": r["d"].get("min_confidence"),
                "path": r["d"].get("path"), "created": str(r["d"].get("created")),
                "candidates": [dict(c) for c in r["cands"]],
            } for r in rows]}
    except Exception as e:
        return {"differentials": [], "error": str(e)}
    finally:
        driver.close()


@app.delete("/differential")
def delete_differential(query: str, mode: str = "differential",
                        min_confidence: Optional[str] = None):
    """Delete a persisted differential (audit cleanup).

    Matches the same id key used at write time: hash(query, mode, min_confidence).
    Removes the Differential node and its HAS_CANDIDATE DxCandidate children
    (and any MATCHES_CONCEPT edges). Returns {deleted: n, id: ...}.
    """
    from neo4j import GraphDatabase
    diff_id = abs(hash((query, mode, min_confidence or ""))) % (2 ** 63)
    driver = GraphDatabase.driver(
        C.NEO4J_URI, auth=(C.NEO4J_USER, C.NEO4J_PASSWORD))
    try:
        with driver.session() as s:
            rec = s.run(
                "MATCH (d:Differential {id:$id})-[:HAS_CANDIDATE]->(c:DxCandidate) "
                "DETACH DELETE d, c RETURN count(c) AS n", id=diff_id).single()
            n = rec["n"] if rec else 0
        return {"deleted": n, "id": diff_id, "query": query}
    except Exception as e:
        return {"deleted": 0, "id": diff_id, "error": str(e)}
    finally:
        driver.close()



@app.get("/domains")
def list_domains_status():
    """Per-domain status: configured vs runnable, collection point counts,
    ingestor/retriever plugin availability, and companion-domain wiring.

    A domain is:
      * configured  — declared in domain_config.yaml (always True here; every
                      listed domain is queryable via federated.assemble)
      * runnable    — has a retrieval surface: a primary Qdrant collection,
                      a Neo4j graph label, a retrieval strategy, or a companion
                      with a collection. (Generic domains like `engineering`
                      are runnable even without a domains/<name>/retrieve.py.)
      * ingestable  — has a domains/<name>/ingest.py plugin OR is ingestible
                      via the generic /ingest endpoint (has a Qdrant collection).
                      Graph-only domains loaded externally (e.g. snomed via
                      load_snomed.py) have collection=null and are not ingestable.
      * has_retriever_plugin / has_ingest_plugin — whether the on-disk plugin
                      folder exists (informational; runnable/ingestable do not
                      require it).
    Plus live Qdrant point counts for any declared collection.
    """
    from domains import get_retriever  # noqa: F401 (registry import side-effect)
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

    domains_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "domains")

    # configured = declared in domain_config.yaml (every configured domain is
    # queryable via federated.assemble). Aliases resolve to a concrete profile.
    configured_names = set(domain_loader.list_domains())

    out = []
    for name in sorted(configured_names):
        # Resolve alias (e.g. `default` -> `engineering`) so capability checks
        # use the concrete profile, not the alias stub.
        profile = domain_loader.get_domain(name)
        coll = profile.get("collection")
        has_retriever_plugin = os.path.isfile(
            os.path.join(domains_dir, name, "retrieve.py"))
        has_ingest_plugin = os.path.isfile(
            os.path.join(domains_dir, name, "ingest.py"))
        # runnable: the domain has SOME retrieval surface (primary collection,
        # graph label, a retrieval strategy, or a companion with a collection).
        has_surface = bool(
            coll or profile.get("neo4j_label")
            or profile.get("retrieval")
            or profile.get("companions"))
        # ingestable: has an ingest.py plugin, OR is ingestible via the generic
        # /ingest endpoint (has a Qdrant collection). Graph-only domains loaded
        # externally (e.g. snomed via load_snomed.py) have collection=null and
        # are therefore not ingestable here.
        ingestable = bool(has_ingest_plugin or coll)
        out.append({
            "name": name,
            "configured": True,
            "runnable": has_surface,
            "ingestable": ingestable,
            "has_retriever_plugin": has_retriever_plugin,
            "has_ingest_plugin": has_ingest_plugin,
            "retrieval": profile.get("retrieval"),
            "collection": coll,
            "points": _pts(coll),
            "prose_domain": profile.get("prose_domain"),
            "companions": profile.get("companions", []),
        })
    return {"domains": out, "default_domain": domain_loader.get_default_domain()}


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


@app.get("/doc/{collection}")
def get_full_document(collection: str, doc_id: str,
                      include_chunks: bool = True):
    """Reconstruct a FULL document from its Qdrant chunks by `doc_id`.

    Retrieval (including companion retrievers) returns individual CHUNKS, each
    tagged with `doc_id` + `chunk_idx`. This endpoint scrolls every point in
    `collection` sharing that `doc_id` and concatenates them in `chunk_idx`
    order to return the complete source document — useful for citation / full
    context when `synthesize:false` returns only the matching chunk(s).

    `doc_id` is passed as a query param (not a path segment) because doc_ids
    may contain slashes (e.g. `eng/your-doc.md`).

    Response:
        {collection, doc_id, source, chunk_count, text, chunks[]}
    """
    import ingest as ingest_mod
    qc = ingest_mod.get_qdrant()
    if not qc.collection_exists(collection):
        from fastapi import HTTPException
        raise HTTPException(status_code=404,
                            detail=f"collection '{collection}' not found")
    pts, _ = qc.scroll(
        collection,
        scroll_filter={"must": [{"key": "doc_id",
                                 "match": {"value": doc_id}}]},
        limit=500, with_payload=True, with_vectors=False)
    if not pts:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404,
            detail=f"doc_id '{doc_id}' not found in '{collection}'")
    source = pts[0].payload.get("metadata", {}).get("source", "")
    ordered = sorted(pts, key=lambda p: p.payload.get("chunk_idx", 0))
    full = "\n\n".join(p.payload["text"] for p in ordered)
    chunks = [{"chunk_idx": p.payload.get("chunk_idx", -1),
               "text": p.payload["text"]} for p in ordered] \
        if include_chunks else []
    return {
        "collection": collection,
        "doc_id": doc_id,
        "source": source,
        "chunk_count": len(ordered),
        "char_count": len(full),
        "text": full,
        "chunks": chunks,
    }


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
    """Extended health (P3.13): per-domain status + backend connectivity + VRAM."""
    import domain_loader
    # Backend connectivity probes (non-fatal; report status, never 500).
    backends = {}
    # Qdrant
    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(url=C.QDRANT_URL, prefer_grpc=False, timeout=3)
        qc.count(C.COLL_CACHE)
        backends["qdrant"] = "ok"
    except Exception as e:
        backends["qdrant"] = f"down:{type(e).__name__}"
    # Neo4j
    try:
        from neo4j import GraphDatabase
        d = GraphDatabase.driver(C.NEO4J_URI, auth=(C.NEO4J_USER, C.NEO4J_PASSWORD),
                                 connection_acquisition_timeout=3)
        with d.session() as s:
            s.run("RETURN 1").single()
        d.close()
        backends["neo4j"] = "ok"
    except Exception as e:
        backends["neo4j"] = f"down:{type(e).__name__}"
    # Synthesis backend (reachable?)
    try:
        import requests
        r = requests.get(f"{C.SYNTHESIS_LLM_BASE_URL.replace('/v1','')}/health",
                         timeout=3)
        backends["synthesis"] = "ok" if r.status_code < 500 else f"down:{r.status_code}"
    except Exception as e:
        backends["synthesis"] = f"down:{type(e).__name__}"

    # Per-domain runnable status (does the domain resolve + have a retriever?)
    domains = {}
    for name in domain_loader.list_domains():
        try:
            prof = domain_loader.get_domain(name)
            domains[name] = {
                "configured": True,
                "retrieval": prof.get("retrieval"),
                "collection": prof.get("collection"),
            }
        except Exception as e:
            domains[name] = {"configured": False, "error": str(e)}

    status = "ok"
    if any(v.startswith("down") for v in backends.values()):
        status = "degraded"

    return {
        "status": status,
        "device": DEVICE,
        "cuda_alloc_gb": round(torch.cuda.memory_allocated()/1e9, 1),
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory/1e9, 1)
        if torch.cuda.is_available() else 0.0,
        "backends": backends,
        "domains": domains,
        "default_domain": domain_loader.get_default_domain(),
    }


@app.get("/metrics")
def metrics():
    """Lightweight Prometheus-format metrics (P3.14) without external deps.

    Exposes:
      rag_requests_total            — cumulative /ask calls (by status)
      rag_request_latency_seconds    — histogram bucketed by latency
      rag_requests_in_flight        — gauge (approx, via counter delta)
    This is sufficient for `scripts/bench_ask.py` and a Prometheus scrape.
    """
    return Response(content=_render_metrics(), media_type="text/plain")


# ── P3.14: in-process metrics (no external dep) ──────────────────────────────
_METRIC_LOCK = threading.Lock()
_REQ_TOTAL = {"ok": 0, "degraded": 0, "error": 0}
_LAT_BUCKETS = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, float("inf")]
_LAT_COUNTS = {b: 0 for b in _LAT_BUCKETS}
_LAT_SUM = 0.0


def _record_metric(status: str, latency: float) -> None:
    global _LAT_SUM
    with _METRIC_LOCK:
        _REQ_TOTAL[status] = _REQ_TOTAL.get(status, 0) + 1
        _LAT_SUM += latency
        for b in _LAT_BUCKETS:
            if latency <= b:
                _LAT_COUNTS[b] += 1
                break


def _render_metrics() -> str:
    with _METRIC_LOCK:
        lines = []
        lines.append("# HELP rag_requests_total Total /ask requests by status.")
        lines.append("# TYPE rag_requests_total counter")
        for k, v in _REQ_TOTAL.items():
            lines.append(f'rag_requests_total{{status="{k}"}} {v}')
        lines.append("# HELP rag_request_latency_seconds Request latency.")
        lines.append("# TYPE rag_request_latency_seconds histogram")
        cum = 0
        for b in _LAT_BUCKETS:
            cum += _LAT_COUNTS[b]
            label = "+Inf" if b == float("inf") else str(b)
            lines.append(
                f'rag_request_latency_seconds_bucket{{le="{label}"}} {cum}')
        lines.append(f"rag_request_latency_seconds_sum {round(_LAT_SUM, 3)}")
        lines.append(f"rag_request_latency_seconds_count {cum}")
        return "\n".join(lines) + "\n"


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
            # TEXT indexes on name/id make the keyword entry-point lookup in
            # neo4j_subgraph index-backed (CONTAINS). Without these, every /ask
            # does a full :Entity label scan (~250-300ms on a populated graph);
            # with them the entry match is ~10-50ms.
            s.run("CREATE TEXT INDEX entity_name_idx IF NOT EXISTS "
                  "FOR (n:Entity) ON (n.name)")
            s.run("CREATE TEXT INDEX entity_id_idx IF NOT EXISTS "
                  "FOR (n:Entity) ON (n.id)")
            print(f"[gpu-daemon] Neo4j vector index '{C.ENTITY_VECTOR_INDEX}' "
                  f"ready (dim={C.VECTOR_DIM}, cosine); TEXT indexes "
                  f"entity_name_idx/entity_id_idx ready", flush=True)
    except Exception as e:
        print(f"[gpu-daemon] Neo4j vector index init skipped: {e}", flush=True)

    # Auto-seed corpora so a non-technical user can query AS-IS.
    # For each domain in config.SEED_ON_STARTUP, if its Qdrant collection is
    # empty, run its ingestor in a background thread (non-blocking).
    for domain in getattr(C, "SEED_ON_STARTUP", []) or []:
        try:
            from domains import list_domains
            if domain not in list_domains():
                continue
            prof = domain_loader.get_domain(domain)
            coll = prof.get("collection")
            if not coll:
                continue
            from qdrant_client import QdrantClient
            qc = QdrantClient(url=C.QDRANT_URL, prefer_grpc=False)
            if qc.collection_exists(coll) and qc.count(coll).count > 0:
                print(f"[gpu-daemon] seed '{domain}': collection '{coll}' "
                      f"non-empty, skip", flush=True)
                continue
            mod = importlib.import_module(f"domains.{domain}.ingest")
            print(f"[gpu-daemon] seeding demo domain '{domain}' "
                  f"-> {coll} ...", flush=True)
            threading.Thread(target=lambda m=mod: _safe_seed(m),
                              daemon=True).start()
        except Exception as e:
            print(f"[gpu-daemon] seed '{domain}' skipped: {e}", flush=True)


if __name__ == "__main__":
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="Seed ALL demo domains idempotently before starting.")
    args = ap.parse_args()

    if args.demo:
        print("[gpu-daemon] --demo: seeding all demo domains ...", flush=True)
        for seed_fn in [
            ("enterprise", "domains.enterprise.seed"),
            ("legal", "domains.legal.seed"),
            ("fraud", "domains.fraud.seed"),
        ]:
            try:
                mod = importlib.import_module(seed_fn[1])
                print(f"  [demo] {seed_fn[0]}: {mod.seed()}", flush=True)
            except Exception as e:
                print(f"  [demo] {seed_fn[0]}: SKIP ({e})", flush=True)
        # snomed + clinical_prose are already populated (purge kept them);
        # re-running is idempotent (collections already non-empty → skipped).
        print("[gpu-daemon] --demo: done. Starting daemon.", flush=True)

    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)
