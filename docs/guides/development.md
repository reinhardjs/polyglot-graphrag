# Development — GraphRAG v2

## Prerequisites

- RTX 3060 or equivalent with ≥12 GB VRAM
- Docker (Qdrant + Neo4j)
- Python 3.11+ (rag-env with CUDA torch)
- systemd services: `gemma-4-e2b` (:8082), `gemma-4-e4b` (:8084)

## Environment

```bash
# Python env (prefix-based — conda activate is broken on this box)
export RAG_ENV=/mnt/data-970-plus/rag-env
export HF_HOME=/mnt/data-970-plus/hf_cache

# Install CUDA torch (once, after env creation)
$RAG_ENV/bin/pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Key deps
$RAG_ENV/bin/pip install fastapi uvicorn qdrant-client neo4j openai \
    sentence-transformers gliner einops requests numpy
# Pin transformers to 4.49.0 (5.x breaks torch 2.3)
$RAG_ENV/bin/pip install transformers==4.49.0
```

## Project layout

```
rag-system/
├── v1/                          # Original CPU-daemon architecture
│   ├── ingest.py, ask.py, config.py, router.py, retrieve.py
│   └── ...                      # (superseded, kept for reference)
├── v2/                          # CURRENT baseline — all-GPU
│   ├── config.py                # All constants
│   ├── serve_gpu.py             # Primary GPU daemon (:8000)
│   ├── serve_cpu.py             # CPU fallback daemon
│   ├── ingest.py                # Doc ingestion
│   ├── ask.py                   # CLI client + shared library
│   ├── retrieve_json.py         # Headless retrieval bridge
│   ├── bench_rag.py             # Latency benchmark
│   ├── run.sh                   # Orchestrator
│   ├── README.md                # Project README
│   └── sample_data/             # 5 engineering docs
├── docs/
│   ├── README.md                   # Docs index
│   ├── architecture/               # architecture.md, full-architecture.md, hermes-integration.md
│   ├── benchmarks/                 # cpu-vs-gpu-benchmark.md, full-benchmark-history.md
│   ├── guides/                     # development.md (this file)
│   └── roadmap/                    # version-requirements.md, multi-domain-plan.md,
│                                   # ingest-endpoint-plan.md, general-purpose-rag-plan.md,
│                                   # agent-learning-system.md
├── v2/
│   ├── MODEL_SWITCHING.md          # Ready-made config snippets for 5 model families
│   └── ...
├── logs/                        # Runtime logs (gitignored)
├── CHANGELOG.md                 # Version history
└── docker-compose.yml           # Qdrant + Neo4j
```

## Running the dev stack

```bash
cd /mnt/data-970-plus/rag-system/v2

# 1. Start DBs
docker compose -f ../docker-compose.yml up -d

# 2. Start LLMs (systemd, boot-only)
sudo systemctl start gemma-4-e2b.service gemma-4-e4b.service

# 3. Start daemon (GPU)
/mnt/data-970-plus/rag-env/bin/python serve_gpu.py

# 4. Ingest sample data
/mnt/data-970-plus/rag-env/bin/python ingest.py sample_data

# 5. Test
/mnt/data-970-plus/rag-env/bin/python ask.py "who reported BUG-204?"
```

Or use `run.sh`:
```bash
bash run.sh serve          # start GPU daemon
bash run.sh ingest sample_data
bash run.sh ask "who reported BUG-204?"
bash run.sh health         # check all services + VRAM
bash run.sh stop           # stop daemon + LLMs
```

## Testing

### Smoke tests
```bash
# Direct API (full synth, caches on first run)
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"who reported BUG-204?","synthesize":true}'
# → {"source":"llm","cache_hit":false,"path":"hybrid","answer":"..."}

# Same query (cache hit — ~0.13s)
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"who reported BUG-204?","synthesize":true}'
# → {"source":"cache","cache_hit":true,"answer":"..."}

# Retrieval-only (never cached)
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"who reported BUG-204?","synthesize":false}'

# Check cache stats
curl -s "http://localhost:6333/collections/query_cache" | python -c "import sys,json; d=json.load(sys.stdin); print(f'points: {d[\"result\"][\"points_count\"]}')"
```

### Ingest API (v2.5.0)
```bash
# Single-doc ingest (non-blocking → returns task_id)
curl -s -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" \
  -d '{"text":"...doc text...","doc_id":"my-doc","doc_type":"Incident","collection":"engineering_chunks"}'
# → {"task_id":"...","status":"accepted","doc_id":"my-doc"}

# Poll ingest status
curl -s http://127.0.0.1:8000/ingest/status/<task_id>
# → {"status":"done","result":{"chunks":4,"entities":8,"timing":{...}}}

# List docs in a collection
curl -s "http://127.0.0.1:8000/ingest?collection=engineering_chunks"

# List all Qdrant collections
curl -s http://127.0.0.1:8000/collections

# Delete a doc (Qdrant + Neo4j)
curl -s -X DELETE "http://127.0.0.1:8000/ingest/my-doc?collection=engineering_chunks"
# → {"status":"deleted","vectors_deleted":4,"nodes_cleaned":8}

# Multi-domain query (cross-domain / "all")
curl -s -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"query":"...","collection":["engineering_chunks","legal_chunks"],"synthesize":false}'
```

### Benchmark
```bash
/mnt/data-970-plus/rag-env/bin/python bench_rag.py
```

### Hermes integration test
```bash
hermes chat --yolo -q "who reported BUG-204?"
# Hermes auto-invokes rag_query → returns sourced answer
```

## Making changes

1. **Code**: edit files in `v2/`. The daemon auto-reloads Python modules on each request (imports are inside endpoint functions).

2. **Re-ingest after ingest.py changes**: `bash run.sh ingest sample_data` (delete-before-reingest is automatic).

3. **Verify**: run smoke tests above. Check `bash run.sh health`.

4. **Document**: update CHANGELOG.md + relevant docs/ file.

## Common issues

| Issue | Fix |
|-------|-----|
| `transformers` import crash (nn not defined) | Pin `transformers==4.49.0` |
| Daemon starts in CPU mode | Install CUDA torch: `pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121` |
| Qdrant point count mismatch | Check `docker compose exec -T qdrant ...` |
| Neo4j profiles empty | Re-ingest: `bash run.sh ingest sample_data` |
| GLiNER first-load timeout | Raise client timeout to 300s; it downloads ~1.6 GB |
| Port :8000 already in use | `fuser -k 8000/tcp` |
| Cache always misses | Check `query_cache` collection exists and has points; daemon log for errors |
| Neo4j container OOM | Check docker-compose heap settings (1-2 GB) |
