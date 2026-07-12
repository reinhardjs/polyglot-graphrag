# GraphRAG v3 — Neuro-Symbolic Knowledge Graph Pipeline

Production-grade, 100% local GraphRAG with domain-agnostic extraction.
All models on GPU (RTX 3060, 12 GB). Python 3.11.

## Architecture (verified 2026-07-12)

| Component | Model | Port | VRAM | Role |
|-----------|-------|------|------|------|
| Embeddings | jina-embeddings-v3 (fp16) | :8000 | ~3.9 GB | Dense vectors |
| Reranker | bge-reranker-v2-m3 (fp16) | :8000 | shared | Result ranking |
| Entity NER | gliner_multi-v2.1 (fp16) | :8000 | lazy-loaded | Zero-shot entity detection |
| Extraction LLM | gemma-4-E2B Q4_0 | :8082 | ~2.3 GB | Relation classification |
| Neo4j | — | :7687 | — | Entity graph |
| Qdrant | — | :6333 | — | Vector store |

**Total GPU: ~8.0 GB / 12 GB** (4 GB headroom). All models run on the GPU daemon
except the extraction LLM which runs as a separate llama-server process.

---

## Extraction Pipeline

| Mode | How | Precision | Latency | Use |
|------|-----|-----------|----------|-----|
| `hybrid` | GLiNER entities → E2B relation classify | 100% | 10-15s | Default (≤4K tokens) |
| `sliding_window` | Sentence-chunk + coref summaries | 100% (short) / high-recall | 21-323s | Long documents |
| `llm` | E2B full-doc single-pass | 89% | 11.8s | Fallback |
| `index_routing` | GLiNER → Qwen 1.5B | ~20% | 1.2s | ❌ Deprecated |

Supported domains: `engineering` (default), `journal`, `legal`, `medical`,
`accounting`, `hospitality`. Schema in `domain_config.yaml`.

---

## Quick Start

```bash
# Start services
sudo systemctl start neo4j qdrant rag-gpu-daemon

# Start E2B extraction model (Option B flags)
LD_LIBRARY_PATH=.../llama.cpp-linux-x86_64-nvidia-cuda-avx2-2.23.1 \
  /home/reinhard/.lmstudio/.../llama-server \
  -m /mnt/data-970-plus/models/gemma-4-E2B_q4_0-it.gguf \
  --host 0.0.0.0 --port 8082 --ctx-size 8192 --cache-type-k f16 \
  --n-gpu-layers 99 --parallel 1 --cont-batching &

# Ingest sample engineering docs
cd /mnt/data-970-plus/rag-system/v2
/mnt/data-970-plus/rag-env/bin/python -c "
import ingest, config as C
C.EXTRACTION_MODE='hybrid'
text=open('sample_data/pr-482-checkout-events.md').read()
r=ingest.ingest_text(text,'pr-482',domain='engineering',extract_graph=True)
print('entities:',r['entities'])
"

# Query via API
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"who reported BUG-204?","synthesize":false}'

# Query via Hermes (auto-invokes rag_query tool)
hermes
> who authored PR-482?
```

---

## Benchmark (RTX 3060, all models on GPU)

| Stage | Time |
|-------|------|
| Embed (Jina v3, GPU) | 58 ms |
| Qdrant search | 67 ms |
| Neo4j subgraph | 82 ms |
| Rerank (BGE, GPU) | 1 ms |
| **Retrieval total** | **127 ms** |

Full pipeline latency depends on extraction mode (see table above).
Run `python bench_rag.py` for per-stage breakdown.

---

## Sample Queries (guaranteed answers)

- "who reported BUG-204 and what severity?" → bob, SEV-2
- "what PR implemented ADR-014 and who reviewed it?" → PR-482, carol
- "how does checkout relate to billing per our ADRs?" → billing depends on checkout
- "basis data apa yang digunakan ADR-021?" (ID) → PostgreSQL

---

## Files

| File | Purpose |
|------|---------|
| `config.py` | All ports, URIs, model names, extraction mode |
| `serve_gpu.py` | Primary daemon — embed, rerank, GLiNER, ingest, /ask |
| `ingest.py` | Late-chunk vectors → Qdrant, graph extraction → Neo4j |
| `hybrid_extraction.py` | GLiNER+E2B hybrid extraction (primary path) |
| `sliding_window.py` | Long-doc sentence-chunk extraction |
| `domain_config.yaml` | Domain schemas (entity/relation types, chunking) |
| `domain_loader.py` | Config loader with validation |
| `ask.py` | CLI client for /ask |
| `bench_rag.py` | Stage-by-stage latency benchmark |
| `sample_data/` | Engineering docs (PR, ADR, bug, wiki) |
| `plans/` | Architecture proposals |

---

## API Endpoints (serve_gpu.py :8000)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ask` | Full RAG retrieval |
| `POST` | `/embed_query` | Single query vector |
| `POST` | `/embed_late` | Full-doc late-chunk vectors |
| `POST` | `/rerank` | BGE rerank |
| `POST` | `/extract_entities` | GLiNER zero-shot NER |
| `POST` | `/ingest` | Single-doc ingest (non-blocking) |
| `GET` | `/ingest` | List ingested documents |
| `DELETE` | `/ingest/{doc_id}` | Remove a document |
| `GET` | `/collections` | List Qdrant collections |
| `GET` | `/health` | GPU status + VRAM |
| `POST` | `/admin/reload` | Hot-reload domain config |

---

## Multi-Domain

Choose domain on ingest:

```bash
# Ingest a journal paper
ingest.ingest_text(text, doc_id, domain="journal", extract_graph=True)
```

Vectors go to `{domain}_collection` (Qdrant). Nodes carry a secondary label
(`:Journal`, `:Engineering`) for scoped graph queries. Config lives in
`domain_config.yaml` — hot-reload via `POST /admin/reload`.

---

## Hermes Integration

The `rag_query` tool calls `POST /ask`. Plugin at `~/.hermes/plugins/rag/`.

```bash
hermes plugins enable rag
hermes
> how does checkout relate to billing per our ADRs?
```

---

## Documentation

| Document | Covers |
|----------|--------|
| `ARCHITECTURE.md` | Full architecture, extraction modes, design decisions |
| `BENCHMARKS.md` | All benchmarks (WS1-4, journal, bugfix impact) |
| `NEXT_PHASE_PLAN.md` | Completed work + dynamic-label roadmap |
| `MODEL_SWITCHING.md` | How to swap any model component |
| `plans/dynamic-label-injection.md` | Plan for fixing GLiNER entity drift |

---

## Version

**v3.1.0** — production. Key features:

- GLiNER+E2B hybrid extraction (100% precision, engineering domain)
- Sliding window for arbitrary-length documents
- Journal domain (academic literature extraction)
- 5 production bug fixes (Qdrant batch cap, sparse collision, label routing,
  domain passthrough, collection isolation)
- **Dynamic Label Injection** — `label_provider.py` auto-expands GLiNER's
  vocabulary with entity names E2B discovers that GLiNER missed. Candidates are
  promoted after `promotion_threshold` documents (default 3) and evicted via LRU
  when `max_per_domain` (default 20) is exceeded. Zero-latency in-memory merge
  on the hot path. State persists to `~/.hermes/labels/{domain}_dynamic.json`.
  See `plans/dynamic-label-injection.md` for design + validation.

  Enable/configure in `domain_config.yaml`:
  ```yaml
  dynamic_labels:
    enabled: true
    max_per_domain: 20
    promotion_threshold: 3
    ttl_docs: 50
  ```

  Audit dropped entities to `logs/dropped_entities.jsonl` (per-doc JSON).
  Iterate with `scripts/bench_drift.py` (baseline gate) and
  `scripts/validate_dynamic_labels.py` (A/B/C recall harness).
