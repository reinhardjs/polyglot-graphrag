# Polyglot GraphRAG v2

**100% local, multi-domain GraphRAG** — engineering bugs, medical records, legal
documents, accounting ledgers, and hospitality SOPs, all ingested and queried by
the same codebase. No external APIs, no cloud, no per-token bills. Everything runs
on a single **RTX 3060 (12 GB)** alongside Qdrant (vectors) + Neo4j (graph).

> Graph + vector hybrid retrieval · cross-encoder reranking · LLM extraction +
> synthesis · **domain profiles** (one TOML per domain) · **citation traceability**
> on every answer · sub-second retrieval, ~0.2 s cold retrieval / ~6 s full answer.

---

## Why this exists

Most RAG is a flat vector search over chunks. Polyglot GraphRAG adds a **knowledge
graph** on top: entities are extracted from documents, resolved across languages
and spellings via cross-lingual embeddings, and connected. A question like
*"how does checkout relate to billing?"* walks the graph (2-hop) **and** does
vector search — both legs run in parallel, then a reranker fuses them. The result
is answers that cite their sources and trace multi-hop relationships a pure vector
store misses.

**v2.6.0 adds general-purpose, drop-in domains.** The 11 domain-specific
assumptions (chunking strategy, entity vocabulary, prompts, metadata) are now
declared in `v2/domains/*.toml`. Switch from engineering to medical to legal by
passing `domain=` — no code changes.

---

## Features

| | |
|---|---|
| **Hybrid retrieval** | Parallel Qdrant vector search + Neo4j k-hop graph traversal, fused by BGE reranker |
| **Vector-driven entity resolution** | Jina v3 cross-lingual embeddings in a Neo4j vector index merge `Basis Data` (ID) ↔ `Database` (EN) ↔ `Base de Datos` (ES) automatically — no hardcoded translation maps |
| **LLM graph extraction** | Gemma 4 E2B extracts entities/edges as JSON (verbatim names); GLiNER is the zero-shot NER fallback |
| **Domain profiles** | `engineering` · `medical` · `legal` · `accounting` · `hospitality` — each with its own chunking, prompts, graph schema, metadata, entry strategy |
| **Pluggable chunking** | `sentence` · `paragraph` · `section` · `fixed` strategies (per profile) |
| **Hybrid graph entry** | `keyword` (fast token overlap) · `vector` (cosine) · `hybrid` — selected per profile |
| **Citation traceability** | Every `/ask` returns `contexts_numbered`, `contexts_meta`, and a `sources` bibliography mapping `[n]` → source doc — zero extra DB cost |
| **Semantic cache** | Identical/similar queries (>0.95 cosine) return the cached answer in <0.01 s |
| **Incremental ingest** | `POST /ingest` is non-blocking (task_id + polling), with checksum-based skip; `DELETE /ingest/{doc_id}` for edits |
| **CPU fallback** | `serve_cpu.py` mirrors the full GPU API for CUDA-less machines (fp32 enforced) |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         RTX 3060 (12 GB)                          │
├──────────────┬──────────────┬────────────────────────────────────┤
│ E2B  :8082   │ E4B  :8084   │  serve_gpu.py :8000                │
│ ~1.5 GB      │ ~3.0 GB      │  Jina v3 + BGE v2-m3 (+GLiNER lazy)│
│ extraction   │ synthesis     │  ~4.0 GB shared                   │
├──────────────┴──────────────┴────────────────────────────────────┤
│                TOTAL: ~10.6 GB / 12 GB  (1.4 GB headroom)        │
└──────────────────────────────────────────────────────────────────┘
         ▲ systemd (Restart=no)              ▲ FastAPI daemon
   ┌─────────────┴──────────────┐
   │  Docker: Qdrant :6333 · Neo4j :7687  │
   └──────────────────────────────────────┘
```

**Ingest** (`ingest.py`): late-chunk embed → Qdrant · LLM extract (E2B) →
vector-resolve entities (Jina v3 → Neo4j index) → write graph.
**Query** (`/ask`): embed query → semantic cache check → parallel Qdrant +
Neo4j → rerank (BGE) → synthesize (E4B) → `{contexts, sources, answer}`.

See [docs/architecture/architecture.md](docs/architecture/architecture.md) for the
full design, VRAM budget, and per-stage latency.

---

## Quick start

```bash
# 1. Databases (Qdrant + Neo4j)
docker compose up -d

# 2. GPU LLMs (systemd, boot-only) + the RAG daemon
sudo systemctl start gemma-4-e2b.service gemma-4-e4b.service rag-gpu-daemon.service
#   — or, without systemd:  cd v2 && bash run.sh serve

# 3. Ingest the sample engineering docs
cd v2 && bash run.sh ingest sample_data
#   — or any doc/domain:
curl -s -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" \
  -d '{"text":"...","doc_id":"enc-1","domain":"medical",
       "metadata":{"patient_id":"P-42","title":"Encounter note"}}'

# 4. Ask
bash run.sh ask "who reported BUG-204?"
curl -s -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"query":"what treats hypertension?","domain":"medical","synthesize":true}'
```

Prerequisites: Python 3.11+ (`rag-env` with CUDA torch), Docker, RTX 3060 (or
equivalent ≥12 GB VRAM). Full setup in [docs/guides/development.md](docs/guides/development.md).

---

## API (served by `serve_gpu.py` on `:8000`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ask` | One-call RAG. `domain` or `collection` (str/list/`"all"`), `synthesize`, `skip_cache`. Returns `contexts_numbered`, `contexts_meta`, `sources`, `answer`. |
| `POST` | `/ingest` | Non-blocking single-doc ingest. `domain`, `metadata`, `collection`, `if_checksum`. → `task_id`. |
| `GET` | `/ingest/status/{task_id}` | Poll ingest task. |
| `DELETE` | `/ingest/{doc_id}?collection=` | Remove doc from Qdrant + Neo4j. |
| `GET` | `/ingest?collection=` | List docs (incl. per-doc `metadata`). |
| `POST` | `/embed_query` · `/embed_late` · `/rerank` · `/extract_graph` | Model primitives. |
| `GET` | `/collections` | List Qdrant collections + point counts. |
| `GET` | `/profiles` | List loaded domain profiles. |
| `GET` | `/health` | Daemon status + VRAM. |

The LLMs speak OpenAI format on separate ports (`E2B :8082`, `E4B :8084`). The
Hermes `rag_query` plugin calls `POST /ask` over HTTP — set `RAG_DAEMON_URL` to
point any Hermes agent at the KB.

---

## Domain profiles

Point the system at a knowledge domain with one TOML file — no code changes:

```bash
curl -s -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" \
  -d '{"text":"Patient has chest pain. Lisinopril treats hypertension.",
       "doc_id":"enc-1","domain":"medical"}'
curl -s -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"query":"what treats hypertension?","domain":"medical","synthesize":true}'
```

Available: `engineering` (sentence, keyword) · `medical` (section, hybrid) ·
`legal` (section, hybrid) · `accounting` (paragraph, keyword) · `hospitality`
(section, hybrid). Add your own in `v2/domains/` — see
[docs/domains/README.md](docs/domains/README.md) for the full schema and a
copy-paste template.

---

## Tests

```bash
cd v2
/mnt/data-970-plus/rag-env/bin/python -m pytest tests/ -q
# 33 passed — chunking, extraction/synthesis prompts, metadata, condense,
# neo4j entry strategies (unit) + cross-domain routing (e2e, needs :8000 live)
```

---

## Documentation map

| Doc | What's in it |
|-----|--------------|
| [docs/README.md](docs/README.md) | Docs index |
| [docs/architecture/architecture.md](docs/architecture/architecture.md) | System design, VRAM, data flow, decisions |
| [docs/architecture/full-architecture.md](docs/architecture/full-architecture.md) | Every component, every config knob |
| [docs/architecture/hermes-integration.md](docs/architecture/hermes-integration.md) | Hermes `rag_query` plugin |
| [docs/domains/README.md](docs/domains/README.md) | Domain profile TOML schema + how to add a domain |
| [docs/guides/development.md](docs/guides/development.md) | Environment, workflows, testing |
| [docs/roadmap/version-requirements.md](docs/roadmap/version-requirements.md) | Master spec (v2.5.0 + v2.6.0) |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

> **Legacy v1 docs.** The root `ARCHITECTURE.md`, `SYSTEM_CONTEXT.md`, and
> `EXTERNAL_ACCESS.md` describe the superseded v1 pipeline (regex extraction,
> Gemma 4 12B on `:8083`). They are retained for historical rationale only;
> the live system is `v2/`.

---

## License

MIT — see repository for details.
