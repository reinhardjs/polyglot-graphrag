# GraphRAG v2 — Elite local RAG system (ALL models on GPU)

Production-grade, 100% local GraphRAG for the engineering knowledge base.
Built under `rag-system/v2/` — the original v1 pipeline is untouched in the
parent directory.

## Architecture (RTX 3060, 12 GB) — verified 2026-07-10

| Component | Model | Port | VRAM | Device |
|-----------|-------|------|------|--------|
| Extraction LLM | Gemma 4 E2B QAT Q4_0 | :8082 | ~1.5 GB | GPU (systemd) |
| Synthesis LLM | Gemma 4 E4B QAT Q4_0 | :8084 | ~3.0 GB | GPU (systemd) |
| Embeddings | jina-embeddings-v3 (fp16) | :8000 | ~4.1 GB shared | GPU (daemon) |
| Router | all-MiniLM-L6-v2 (fp16) | :8000 | (shared) | GPU |
| Reranker | bge-reranker-v2-m3 (fp16) | :8000 | (shared) | GPU |
| Graph extractor | gliner_multi-v2.1 (fp16) | :8000 | lazy-loaded | GPU |
| **Full RAG** | `POST /ask` | :8000 | — | GPU |

**Total GPU: ~10.7 GB / 12 GB** (1.5 GB headroom). GLiNER is lazy-loaded on
first `/extract_graph` call (E2B LLM is primary; GLiNER is fallback). Query-only
workloads (embed + rerank + synthesize) stay ~1.6 GB lighter.

Prerequisite: `rag-env` must have CUDA torch:
```bash
/mnt/data-970-plus/rag-env/bin/pip install torch==2.3.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```
(The default rag-env ships `torch 2.3.1+cpu`.)

## Files

| File | Purpose |
|------|---------|
| `config.py` | Ports, DB creds, GLiNER labels, CANON_MAP, token budgets, URLs |
| `serve_gpu.py` | **Primary daemon.** Preloads Jina/MiniLM/BGE on GPU at startup. Exposes `/ask` (one-call RAG), `/embed_late`, `/embed_query`, `/rerank`, `/extract_graph`. |
| `serve_cpu.py` | **CPU fallback.** Same API as `serve_gpu.py` but runs models on CPU (no CUDA torch). |
| `ingest.py` | Late-embed → Qdrant, LLM extract (E2B :8082) → canonicalize → Neo4j. |
| `ask.py` | CLI client: embed → Qdrant‖Neo4j → rerank → E4B synthesis. Thin; all model work delegated to daemon/LLMs. |
| `retrieve_json.py` | Headless retrieval: returns contexts as JSON. Used by Hermes rag plugin. |
| `run.sh` | Orchestrator: serve/ingest/ask/retrieve/health/stop. |
| `bench_rag.py` | Pure latency benchmark for each pipeline stage. |
| `sample_data/` | 5 engineering docs: 2 ADRs, 1 bug, 1 PR, 1 wiki. |

## API endpoints (served by `serve_gpu.py` on :8000)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ask` | **One-call full RAG.** `{"query":"...","synthesize":true/false}` → `{"contexts":[...],"answer":"..."}` |
| `POST` | `/embed_query` | Single query vector → `{"vector":[0.036...,...]}` |
| `POST` | `/embed_late` | Full-doc late-chunked vectors → `{"doc_id":"..","chunks":[...]}` |
| `POST` | `/rerank` | BGE rerank query vs docs → `{"ranked":[[0,0.9],[1,0.8]]}` |
| `POST` | `/extract_graph` | GLiNER NER + co-occurrence edges (lazy-loads on first call) |
| `GET` | `/health` | `{"status":"ok","device":"cuda","cuda_alloc_gb":2.3}` |

The LLMs speak OpenAI format on separate ports:
- E2B extraction: `POST http://localhost:8082/v1/chat/completions`
- E4B synthesis: `POST http://localhost:8084/v1/chat/completions`

## Usage

```bash
cd /mnt/data-970-plus/rag-system/v2
sudo systemctl start gemma-4-e2b.service gemma-4-e4b.service
bash run.sh serve              # start GPU daemon on :8000
bash run.sh ingest sample_data # populate Qdrant + Neo4j
bash run.sh ask "who reported BUG-204?"
bash run.sh retrieve "who reported BUG-204?"  # retrieval-only, no synthesis
bash run.sh health             # check all services + VRAM
bash run.sh stop               # stop daemon + LLMs
```

Pure API (no `bash run.sh`, no `ask.py` client needed):
```bash
# Full answer with E4B synthesis
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"who reported BUG-204?","synthesize":true}'

# Retrieval-only (contexts, ~0.2s)
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"who reported BUG-204?","synthesize":false}'
```

## Benchmark (RTX 3060, all models on GPU)

| Stage | Mean | Min | Max |
|-------|------|-----|-----|
| embed (Jina, GPU) | 0.08 s | 0.08 | 0.13 |
| qdrant search | 0.03 s | 0.03 | 0.04 |
| neo4j subgraph | 0.01 s | 0.01 | 0.02 |
| rerank (BGE, GPU) | 0.05 s | 0.04 | 0.07 |
| **retrieval total** | **0.22 s** | 0.19 | 0.29 |
| synth (E4B, GPU) | 6.38 s | 5.30 | 7.86 |
| **full pipeline** | **6.61 s** | 5.52 | 8.14 |

> 97% of latency is E4B generation. Retrieval is sub-250ms. Run `bench_rag.py` to re-measure.

## Hermes integration

The `rag_query` Hermes tool calls `POST /ask` over HTTP. The plugin lives at
`~/.hermes/plugins/rag/`. Set `RAG_DAEMON_URL` to point to the daemon host.

```bash
# Enable the plugin
hermes plugins enable rag  (keep allow_tool_override=false)

# In chat — Hermes auto-invokes rag_query for engineering questions
hermes
# > how does checkout relate to billing per our ADRs?
```

## Sample queries (guaranteed answers)

- "who reported BUG-204 and what severity was it?" → bob, SEV-2
- "what PR implemented ADR-014 and who reviewed it?" → PR-482, carol
- "how does checkout relate to billing per our ADRs?" → billing depends on checkout; ADR-014 event-driven migration
- "basis data apa yang digunakan ADR-021?" (ID) → PostgreSQL

## Known gaps

- Edges mostly `ASSOCIATED_WITH` (E2B JSON tends to drop edge types → GLiNER fallback).
- Canonicalization is exact-match only; multi-word entities not merged.
- Re-ingesting a doc overwrites vectors (upsert by doc_id:chunk_idx) and merges nodes (MERGE), but stale chunks/nodes from *deleted* content linger.
- For incremental updates, re-ingest the full folder. A per-doc `update` command (delete-then-reingest) is a planned improvement.

## Verified (2026-07-10)

- CUDA torch 2.3.1+cu121 installed in rag-env (cuda: True 12.1)
- serve_gpu.py loads Jina/MiniLM/BGE on GPU at startup; GLiNER lazy-loaded
- E2B (:8082) extraction returns valid JSON; E4B (:8084) synthesis streams answers
- `/ask` endpoint works: retrieval-only ~0.2s, full synth ~6.6s
- Hermes plugin auto-invokes rag_query → sourced answers (bob/SEV-2, PR-482/carol, checkout↔billing)
- Multilingual: Indonesian doc (adr-021) ingested; CANON_MAP maps ID→EN
- Benchmark captured (3 queries × 3 iterations)
