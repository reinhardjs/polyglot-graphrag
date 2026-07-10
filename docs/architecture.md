# Architecture — GraphRAG v2

## Overview

Production-grade, 100% local GraphRAG for the engineering knowledge base. All
models occupy a single RTX 3060 (12 GB) simultaneously — no swap, no CPU
offload, no external API calls.

```
┌─────────────────────────────────────────────────────────────────┐
│                        RTX 3060 (12 GB)                         │
├──────────┬──────────┬──────────────────────────────────────────┤
│ E2B :8082│ E4B :8084│  serve_gpu.py :8000 (Jina+MiniLM+BGE     │
│ ~1.5 GB  │ ~3.0 GB  │  + GLiNER lazy) ~4.1 GB shared           │
│ extract  │ synth    │  embed / rerank / extract_graph / ask     │
├──────────┴──────────┴──────────────────────────────────────────┤
│               TOTAL: ~10.7 GB / 12 GB                           │
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
| all-MiniLM-L6-v2 fp16 | serve_gpu.py | :8000 | ~0.1 GB | Routing (unused currently, available) |
| bge-reranker-v2-m3 fp16 | serve_gpu.py | :8000 | ~1.0 GB | Cross-encoder reranking |
| gliner_multi-v2.1 fp16 | serve_gpu.py | :8000 | ~1.6 GB | NER graph extractor (lazy, fallback) |

### Code modules
| File | Responsibility |
|------|---------------|
| `serve_gpu.py` | **Primary daemon.** Preloads aux models on GPU. Exposes `/ask` (one-call RAG), `/embed_query`, `/embed_late`, `/rerank`, `/extract_graph`, `/health`. |
| `serve_cpu.py` | **Fallback daemon.** Same API, runs models on CPU. Used if CUDA torch unavailable. |
| `ingest.py` | Document→Qdrant+Neo4j. Late-chunk embeds, LLM extraction (E2B) or GLiNER fallback, canonicalise, write. |
| `ask.py` | CLI client + shared library. `parallel_retrieve()` (Qdrant+Neo4j threaded), `condense()` (rerank), `synthesize()` (E4B stream). |
| `config.py` | All constants: ports, credentials, token limits, CANON_MAP. No YAML parsing. |
| `run.sh` | Orchestrator: `serve`, `ingest`, `ask`, `retrieve`, `health`, `stop`. |
| `bench_rag.py` | Per-stage latency benchmark (3 queries × 3 iterations). |
| `retrieve_json.py` | Headless retrieval → JSON. Used by Hermes plugin (legacy; now prefers /ask). |

### Data stores
| Store | Collection/DB | Contents |
|-------|--------------|----------|
| Qdrant | `engineering_chunks` | Document chunks: dense 1024-d vectors, hash-based sparse vectors, metadata |
| Neo4j | `neo4j` | Entity graph: nodes (id, name, type, profile, source_doc), edges (ASSOCIATED_WITH) |

## Data flow

### Ingest (`ingest.py`)
```
doc.md
  ├─[1] late-chunk Jina embed → Qdrant (dense + sparse vectors)
  └─[2] LLM extract (E2B :8082 → JSON nodes+edges)
        │  └─ fallback: GLiNER daemon :8000/extract_graph
        ├─ canonicalise (CANON_MAP: ID→EN)
        ├─ build profiles (source-text context windows)
        └─ write to Neo4j (MERGE nodes + edges)
```

### Query (`/ask` or `ask.py`)
```
user query
  ├─[1] embed query (Jina, in-process on GPU)
  ├─[2] PARALLEL:
  │     Thread-A: Qdrant hybrid (dense + hash-based sparse, RRF fusion)
  │     Thread-B: Neo4j k-hop (keyword-overlap entry → 2-hop traversal)
  ├─[3] fuse + dedupe → rerank (BGE, in-process on GPU)
  └─[4] synthesize (E4B :8084, streamed)  [if synthesize=true]
       → answer with citations
```

## VRAM budget

```
Model                    | VRAM (GB) | Notes
-------------------------|-----------|------
Gemma E2B QAT Q4_0       | 1.5       | systemd, :8082
Gemma E4B QAT Q4_0       | 3.0       | systemd, :8084
jina-embeddings-v3 fp16  | 3.0       | daemon, shared
bge-reranker-v2-m3 fp16  | 1.0       | daemon, shared
all-MiniLM-L6-v2 fp16    | 0.1       | daemon, shared
GLiNER fp16 (lazy)       | 1.6       | daemon, loaded only on extract_graph
CUDA context + overhead  | 0.5       |
-------------------------|-----------|------
TOTAL                    | 10.7      | / 12 GB
```

## Key design decisions

1. **Hash-based sparse vectors** — `hash(token) % 65536` ensures consistent term indices across all chunks and queries. Replaced per-chunk `enumerate(Counter)` which produced incompatible vocabularies.

2. **Neo4j node profiles** — each Entity stores a 1-3 sentence context window from the source document. This makes the graph leg of retrieval useful (nodes carry actual text, not just labels).

3. **Keyword-overlap entry selection** — finding the Neo4j entry node by word overlap is fast (~1ms) and effective for 95% of queries (entity IDs like `bug-204`, `pr-482`, `adr-014` match naturally). Vector-similarity fallback only triggers when overlap finds zero candidates.

4. **One-call `/ask` API** — a single HTTP POST runs the full pipeline. Embed + rerank run IN-PROCESS on the resident GPU models (no HTTP self-loop). Only E4B synthesis is a cross-process call.

5. **Portable Hermes integration** — the rag plugin calls `POST /ask` over HTTP. `RAG_DAEMON_URL` env override allows any Hermes agent on any machine to use the KB.

6. **Delete-before-reingest** — each `ingest_file` call deletes existing Qdrant points + Neo4j nodes for that doc_id first, then re-inserts. No stale content from edits/deletions.

## Known gaps

- Edge types are almost all `ASSOCIATED_WITH` (E2B JSON tends to omit relationship semantics — GLiNER fallback co-occurrence is the safety net).
- Canonicalisation is exact-match only; multi-word entity merging not implemented.
- No semantic cache (identical questions re-run the full pipeline).
- No conversation/multi-turn support (each query is stateless).
- E4B synthesis includes the reasoning trace (bullet-point chain) before the answer.
