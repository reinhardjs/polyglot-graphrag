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
│   ├── architecture.md          # Architecture decisions + VRAM budget
│   ├── hermes-integration.md    # Hermes plugin docs
│   └── development.md           # This file
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
# Direct API
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"who reported BUG-204?","synthesize":false}'
# → {"n_contexts":5, ...}

# Full answer
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"who reported BUG-204 and what severity?","synthesize":true}'
# → {"answer":"bob [1]. Severity: SEV-2 [1]", ...}
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
| Neo4j container OOM | Check docker-compose heap settings (1-2 GB) |
