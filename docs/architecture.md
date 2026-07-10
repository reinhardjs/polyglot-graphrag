# Architecture — GraphRAG v2

## Overview

Production-grade, 100% local GraphRAG for the engineering knowledge base. All
models occupy a single RTX 3060 (12 GB) simultaneously — no swap, no CPU
offload, no external API calls.

```
┌─────────────────────────────────────────────────────────────────┐
│                        RTX 3060 (12 GB)                         │
├──────────┬──────────┬──────────────────────────────────────────┤
│ E2B :8082│ E4B :8084│  serve_gpu.py :8000 (Jina+BGE          │
│ ~1.5 GB  │ ~3.0 GB  │  + GLiNER lazy) ~4.0 GB shared           │
│ extract  │ synth    │  embed / rerank / extract_graph / ask     │
├──────────┴──────────┴──────────────────────────────────────────┤
│               TOTAL: ~10.6 GB / 12 GB                           │
└─────────────────────────────────────────────────────────────────┘
         ▲                        ▲
    systemd services         FastAPI daemon
    (Restart=no)             (uvicorn, 1 worker)

                    ┌───────┴───────┐
                    │   Docker      │
                    │ Qdrant :6333  │
                    │ Neo4j  :7687  │
                    └───────────────┘
```

## Component responsibility

### Models (GPU)
| Model | Service | Port | VRAM | Role |
|-------|---------|------|------|------|
| Gemma 4 E2B QAT Q4_0 | systemd gemma-4-e2b | :8082 | ~1.5 GB | Entity/edge extraction (JSON, primary) |
| Gemma 4 E4B QAT Q4_0 | systemd gemma-4-e4b | :8084 | ~3.0 GB | Answer synthesis |
| jina-embeddings-v3 fp16 | serve_gpu.py | :8000 | ~3.0 GB | Query & doc embedding (late-chunking) |
| bge-reranker-v2-m3 fp16 | serve_gpu.py | :8000 | ~1.0 GB | Cross-encoder reranking |
| gliner_multi-v2.1 fp16 | serve_gpu.py | :8000 | ~1.6 GB | NER graph extractor (lazy, fallback) |

### Code modules
| File | Responsibility |
|------|---------------|
| `serve_gpu.py` | **Primary daemon.** Preloads aux models on GPU (identities from config). Exposes `/ask`, `/embed_query`, `/embed_late`, `/rerank`, `/extract_graph`, `/models`, `/health`. |
| `serve_cpu.py` | **Fallback daemon.** Same API, runs models on CPU. Used if CUDA torch unavailable. |
| `ingest.py` | Document→Qdrant+Neo4j. Late-chunk embeds, LLM extraction (E2B) or GLiNER fallback, vector-driven entity resolution (Jina v3 → Neo4j index). |
| `ask.py` | CLI client + shared library. `parallel_retrieve()` (Qdrant+Neo4j threaded), `condense()` (rerank), `synthesize()` (E4B stream). |
| `config.py` | All constants: ports, credentials, token limits, entity resolution threshold, vector index config. No YAML parsing. |
| `run.sh` | Orchestrator: `serve`, `ingest`, `ask`, `retrieve`, `health`, `stop`. |
| `bench_rag.py` | Per-stage latency benchmark (3 queries × 3 iterations). |
| `retrieve_json.py` | Headless retrieval → JSON. Used by Hermes plugin (legacy; now prefers /ask). |

### Data stores
| Store | Collection/DB | Contents |
|-------|--------------|----------|
| Qdrant | `engineering_chunks` | Document chunks: dense 1024-d vectors, hash-based sparse vectors, metadata |
| Neo4j | `neo4j` | Entity graph: nodes (id, name, type, name_vector, aliases[], source_docs[], profile), edges (ASSOCIATED_WITH, etc.) |

## Data flow

### Ingest (`ingest.py`)
```
doc.md
  ├─[1] late-chunk Jina embed → Qdrant (dense + sparse vectors)
  └─[2] LLM extract (E2B :8082 → JSON nodes+edges, VERBATIM names)
        │  └─ fallback: GLiNER daemon :8000/extract_graph
        ├─ vector-resolve: embed each name via Jina v3 → query Neo4j
        │  entity_vector_idx (>0.88 cosine → merge aliases, else new)
        ├─ build profiles (source-text context windows)
        └─ write to Neo4j (MERGE nodes + edges, shared source_docs)
```

### Query (`/ask` or `ask.py`)
```
user query
  ├─[1] embed query (Jina, in-process on GPU)
  ├─[2] SEMANTIC CACHE — check Qdrant query_cache at >0.95 cosine
  │     └─ hit → return cached answer (<0.01s)
  ├─[3] PARALLEL:
  │     Thread-A: Qdrant hybrid (dense + hash-based sparse, RRF fusion)
  │     Thread-B: Neo4j k-hop (keyword-overlap entry → 2-hop traversal)
  ├─[4] fuse + dedupe → rerank (BGE, in-process on GPU)
  ├─[5] synthesize (E4B :8084, streamed)  [if synthesize=true]
  │     └─ reasoning trace (stdout only) → clean answer (API field)
  ├─[6] store in query_cache for future hits
  └─[7] return {source, path, qdrant_hits, graph_hits, rerank_scores, contexts, answer}
```

## VRAM budget

```
Model                    | VRAM (GB) | Notes
-------------------------|-----------|------
Gemma E2B QAT Q4_0       | 1.5       | systemd, :8082
Gemma E4B QAT Q4_0       | 3.0       | systemd, :8084
jina-embeddings-v3 fp16  | 3.0       | daemon, shared
bge-reranker-v2-m3 fp16  | 1.0       | daemon, shared
GLiNER fp16 (lazy)       | 1.6       | daemon, loaded only on extract_graph
CUDA context + overhead  | 0.5       |                          |
|-------------------------|-----------|------|
| TOTAL                    | 10.6      | / 12 GB |
```
> GLiNER not in this total — only loaded if `/extract_graph` is called (ingest fallback), adding ~1.6 GB.

## Performance (RTX 3060, all models on GPU, v2.4.1)

| Stage | GPU (serve_gpu.py) | CPU (serve_cpu.py, standalone) |
|-------|-------------------|-------------------------------|
| embed (Jina v3) | 0.056s | 0.776s |
| Qdrant search | 0.039s | 0.039s |
| Neo4j subgraph | 0.006s | 0.006s |
| rerank (BGE v2-m3) | 0.061s (27 docs) | 1.376s (15 docs, capped) |
| **retrieval total** | **0.162s** | **2.197s** |
| synthesis (E4B) | 3.7–8.6s | same (same E4B endpoint) |
| **full pipeline** | **3.9–8.8s** | **5.9–11.4s** |

Cache hit (synthesize=true): <0.01s — 50× faster than cold.

## Key design decisions

1. **Hash-based sparse vectors** — `hash(token) % 65536` ensures consistent term indices across all chunks and queries. Replaced per-chunk `enumerate(Counter)` which produced incompatible vocabularies.

2. **Neo4j node profiles** — each Entity stores a 1-3 sentence context window from the source document. This makes the graph leg of retrieval useful (nodes carry actual text, not just labels).

3. **Keyword-overlap entry selection** — finding the Neo4j entry node by word overlap is fast (~1ms) and effective for 95% of queries (entity IDs like `bug-204`, `pr-482`, `adr-014` match naturally). Vector-similarity fallback only triggers when overlap finds zero candidates.

4. **One-call `/ask` API** — a single HTTP POST runs the full pipeline. Embed + rerank run IN-PROCESS on the resident GPU models (no HTTP self-loop). Only E4B synthesis is a cross-process call.

5. **Portable Hermes integration** — the rag plugin calls `POST /ask` over HTTP. `RAG_DAEMON_URL` env override allows any Hermes agent on any machine to use the KB.

6. **Delete-before-reingest** — each `ingest_file` call deletes existing Qdrant points + Neo4j nodes for that doc_id first, then re-inserts. No stale content from edits/deletions.

7. **Semantic cache** — identical/similar queries (>0.95 cosine) return cached answer in <0.01s with 0 LLM tokens. Only caches synthesize=true responses. `skip_cache: true` bypasses.

8. **Observable retrieval** — every response includes `source` (llm|cache), `path` (qdrant|graph|hybrid), per-leg hit counts, and per-context rerank scores. Full debugging transparency without extra tooling.

9. **Clean synthesis output** — E4B reasoning trace (chain-of-thought) is routed to stdout only. The `answer` API field contains only the final answer text.

10. **No zero-shot routing (MiniLM removed)** — benchmarked at 50% accuracy on real queries. Parallel retrieval (always run both legs) is safer (wrong route loses context), costs the same wall-clock time (~0.04s since legs run concurrently), and frees 0.1 GB VRAM.

11. **Vector-driven entity resolution** — replaced hardcoded CANON_MAP with Jina v3's cross-lingual embeddings stored in Neo4j's native vector index (5.x). Entities are extracted VERBATIM by E2B (no forced translation). Each entity name is embedded, and `db.index.vector.queryNodes()` finds the closest existing entity. Above ENTITY_RESOLUTION_THRESHOLD (0.88 cosine) → reuse existing node + append alias. Below → new canonical entity. "Basis Data" (ID), "Database" (EN), and "Base de Datos" (ES) converge automatically. Aliases preserved for audit trail. `source_docs[]` list tracks multi-document provenance safely.

12. **Dual-architecture reranking** — same BGE v2-m3 model on both GPU and CPU daemons. GPU path: full 27-doc pool at 61ms (CUDA tensor cores). CPU path: pool capped to 15 docs at 1.4s (config: `RERANK_CPU_POOL_CAP`). Same model, same quality — cap delivers speed without model downgrade. Pool cap value proven via 5→27 doc sweep (cap=15 hits 4.5/5 overlap with full reference).

13. **Fully modular config** — every model-specific behavior is a config flag, not hardcoded. `EMBED_TRUST_REMOTE`, `EMBED_USE_HALF`, `EMBED_TASK_PASSAGE`, `EMBED_TASK_QUERY`, `RERANK_USE_HALF`, `EXTRACTION_READS_REASONING`. Swap any component by editing config.py — no code changes. `GET /models` endpoint returns current identity + flags for verification.

## Known gaps

- Edge types are almost all `ASSOCIATED_WITH` (E2B JSON tends to omit relationship semantics — GLiNER fallback co-occurrence is the safety net).
- Entity resolution threshold (0.88) is a tunable constant — higher reduces false merges, lower connects more variants.
- No multi-turn conversation support (each query is stateless).
- **No incremental ingest** — currently batch-only (ingest.py rebuilds all). `/ingest` endpoint planned (see `docs/ingest-endpoint-plan.md`) for single-doc updates. Frequent engineering doc changes need this + optional file-watcher.
- **Single-domain only** — Qdrant collection is hardcoded to `engineering_chunks`. Multi-domain support (legal, hospitality, accounting, etc.) is planned via namespace-based collections and Neo4j domain labels (see `docs/multi-domain-plan.md`).
