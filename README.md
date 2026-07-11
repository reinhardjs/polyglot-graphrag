# Polyglot GraphRAG v2

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/GPU-RTX%203060%2012GB-orange.svg" alt="RTX 3060 12GB">
  <img src="https://img.shields.io/badge/tests-33%20passed-brightgreen.svg" alt="Tests: 33 passed">
</p>

<p align="center">
  <b>100% local, multi-domain GraphRAG.</b> Engineering bugs. Medical records.
  Legal documents. Accounting ledgers. Hospitality SOPs. One codebase,
  zero cloud APIs, zero per-token bills — just an RTX 3060.
</p>

<p align="center">
  hybrid retrieval · cross-encoder reranking · LLM extraction + synthesis ·
  domain profiles · citation traceability · ~0.2 s retrieval / ~6 s full answer
</p>

---

## Table of Contents

- [What is this?](#what-is-this)
- [Features](#features)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [API](#api)
- [Domain profiles](#domain-profiles)
- [Tests](#tests)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## What is this?

Most RAG is flat vector search over chunks. You get the chunk, you guess the
rest. Polyglot GraphRAG adds a **knowledge graph** on top: entities are
extracted from documents, resolved across languages and spellings via
cross-lingual embeddings, and connected. A question like *"how does checkout
relate to billing?"* walks the graph **and** does vector search — both legs
run in parallel, then a reranker fuses them. You get answers that cite their
sources and trace relationships a pure vector store misses.

**Domain profiles** (v2.6.0) make it general-purpose. The 11 domain-specific
assumptions — chunking strategy, entity vocabulary, prompts, metadata — live in
`v2/domains/*.toml`. Switch from engineering to medical to legal by passing
`domain=` — no code changes, no restart.

All from a single RTX 3060 (12 GB): extraction (E2B, ~1.5 GB), synthesis (E4B,
~3 GB), embedding + reranking (Jina v3 + BGE v2-m3, ~4 GB shared), and GLiNER
(lazy-loaded NER fallback). ~10.6 GB total with 1.4 GB headroom.

---

## Features

| Feature | Description |
|---------|-------------|
| **Hybrid retrieval** | Parallel Qdrant vector search + Neo4j k-hop graph traversal, fused by BGE cross-encoder reranker |
| **Vector entity resolution** | Jina v3 cross-lingual embeddings in Neo4j's native vector index merge `Basis Data` (ID) ↔ `Database` (EN) ↔ `Base de Datos` (ES) — no hardcoded translation maps |
| **LLM graph extraction** | Gemma 4 E2B extracts entities/edges as JSON (verbatim names); GLiNER multi-v2.1 is the zero-shot NER fallback |
| **Domain profiles** | `engineering` · `medical` · `legal` · `accounting` · `hospitality` — each controls chunking, prompts, graph schema, metadata, entry strategy |
| **Pluggable chunking** | `sentence` · `paragraph` · `section` · `fixed` — selected per profile |
| **Hybrid graph entry** | `keyword` (fast token overlap) · `vector` (cosine similarity) · `hybrid` (both in parallel) — selected per profile |
| **Citation traceability** | Every `/ask` response carries `contexts_numbered`, `contexts_meta`, and a `sources` bibliography mapping `[n]` → source doc |
| **Semantic cache** | Similar queries (>0.95 cosine) return cached answers in ~0.07 s |
| **Incremental ingest** | `POST /ingest` is non-blocking (task_id + polling); checksum-based skip on re-ingest; `DELETE` for edits |
| **CPU fallback** | `serve_cpu.py` mirrors the full GPU API for CUDA-less machines (fp32 enforced) |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         RTX 3060 (12 GB)                          │
├──────────────┬──────────────┬────────────────────────────────────┤
│ E2B  :8082   │ E4B  :8084   │  serve_gpu.py :8000                │
│ ~1.5 GB      │ ~3.0 GB      │  Jina v3 + BGE v2-m3 (+GLiNER)    │
│ extraction   │ synthesis    │  ~4.0 GB shared                    │
├──────────────┴──────────────┴────────────────────────────────────┤
│                TOTAL: ~10.6 GB / 12 GB                            │
└──────────────────────────────────────────────────────────────────┘
         │                              │
    systemd (Restart=no)          FastAPI daemon
         │                              │
   ┌─────┴──────────────────────────────┴─────┐
   │  Docker: Qdrant :6333  ·  Neo4j :7687    │
   └──────────────────────────────────────────┘
```

**Ingest pipeline** — document → late-chunk embed (Jina v3, strategy from
profile) → Qdrant · LLM extract (E2B) or GLiNER fallback → vector-resolve
entities (Jina v3 → Neo4j vector index) → write graph + node profiles.

**Query pipeline** — embed query → semantic cache check → parallel Qdrant +
Neo4j → rerank (BGE) → synthesize (E4B) → `{contexts, sources, answer}`.

See [docs/architecture/architecture.md](docs/architecture/architecture.md) for
the full design, VRAM budget, and per-stage latency numbers.

---

## Quick start

```bash
# 1. Databases
docker compose up -d

# 2. GPU LLMs + RAG daemon
sudo systemctl start gemma-4-e2b.service gemma-4-e4b.service rag-gpu-daemon.service
#   — or without systemd:  cd v2 && bash run.sh serve

# 3. Ingest the sample engineering docs
cd v2 && bash run.sh ingest sample_data

# 4. Ask
bash run.sh ask "who reported BUG-204?"
```

Or via pure HTTP — no CLI needed:

```bash
# Ingest a doc into a domain
curl -s -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text":"...","doc_id":"enc-1","domain":"medical",
       "metadata":{"patient_id":"P-42","title":"Encounter note"}}'

# Query with synthesis
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"what treats hypertension?","domain":"medical","synthesize":true}'

# Retrieval-only (no LLM, ~0.17 s)
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"who reported BUG-204?","synthesize":false}'
```

**Prerequisites:** Python 3.11+ (`rag-env` with CUDA torch), Docker, RTX 3060
or equivalent ≥12 GB VRAM. Full setup guide in
[docs/guides/development.md](docs/guides/development.md).

---

## API

All endpoints served by `serve_gpu.py` on `:8000`. `serve_cpu.py` is a drop-in
replacement for CUDA-less machines (identical API, fp32 models).

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ask` | One-call RAG. Accepts `domain` or `collection` (str, list, or `"all"`), `synthesize`, `skip_cache`. Returns `contexts_numbered`, `contexts_meta`, `sources`, `answer`. |
| `POST` | `/ingest` | Non-blocking single-doc ingest. Accepts `domain`, `metadata`, `collection`, `if_checksum`. Returns `task_id`. |
| `GET` | `/ingest/status/{task_id}` | Poll a running ingest task. |
| `DELETE` | `/ingest/{doc_id}?collection=` | Remove a document from Qdrant + Neo4j. |
| `GET` | `/ingest?collection=` | List ingested docs (includes per-doc `metadata`). |
| `POST` | `/embed_query` `/embed_late` `/rerank` `/extract_graph` | Model primitives (embedding, reranking, NER extraction). |
| `GET` | `/collections` | List Qdrant collections and point counts. |
| `GET` | `/profiles` | List loaded domain profiles (name, collection, label, chunking strategy, entry strategy). |
| `GET` | `/health` | Daemon status, device, VRAM used. |

The LLMs speak the OpenAI chat-completions format on separate ports:
- Extraction: `POST http://localhost:8082/v1/chat/completions`
- Synthesis: `POST http://localhost:8084/v1/chat/completions`

Hermes integration: the `rag_query` Hermes tool calls `POST /ask` over HTTP.
Set `RAG_DAEMON_URL` to point any Hermes agent at the KB.

---

## Domain profiles

One TOML file per domain. No code changes — the same codebase handles
engineering bugs, medical records, and legal documents.

```bash
# Ingest with domain routing
curl -s -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" \
  -d '{"text":"Patient has chest pain. Lisinopril treats hypertension.",
       "doc_id":"enc-1","domain":"medical"}'

# Query with domain-aware synthesis (clinical tone)
curl -s -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"query":"what treats hypertension?","domain":"medical","synthesize":true}'
```

| Domain | Collection | Chunking | Entry | Use case |
|--------|-----------|----------|-------|----------|
| `engineering` | `engineering_chunks` | sentence | keyword | ADRs, bugs, PRs, wikis |
| `medical` | `medical_chunks` | section | hybrid | Clinical notes, prescriptions, lab results |
| `legal` | `legal_chunks` | section | hybrid | Statutes, cases, regulations |
| `accounting` | `accounting_chunks` | paragraph | hybrid | Financial statements, ledgers, audit trails |
| `hospitality` | `hospitality_chunks` | section | hybrid | SOPs, recipes, inventory, service standards |

To add a new domain, copy any `.toml` from `v2/domains/` and tune the fields.
Full schema and field reference in
[docs/domains/README.md](docs/domains/README.md).

---

## Tests

```bash
cd v2
/mnt/data-970-plus/rag-env/bin/python -m pytest tests/ -q
# 33 passed — chunking, extraction/synthesis prompts, metadata validation,
# condense dedup, neo4j entry strategies (unit) + cross-domain routing (e2e)
```

---

## Documentation

| Document | Contents |
|----------|----------|
| [docs/README.md](docs/README.md) | Docs index |
| [docs/architecture/architecture.md](docs/architecture/architecture.md) | System design, VRAM budget, data flow, design decisions |
| [docs/architecture/full-architecture.md](docs/architecture/full-architecture.md) | Every component, every config knob |
| [docs/architecture/hermes-integration.md](docs/architecture/hermes-integration.md) | Hermes `rag_query` plugin setup |
| [docs/domains/README.md](docs/domains/README.md) | TOML profile schema + how to add a domain |
| [docs/guides/development.md](docs/guides/development.md) | Environment, workflows, testing |
| [docs/roadmap/version-requirements.md](docs/roadmap/version-requirements.md) | Master specification (v2.5.0 + v2.6.0) |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

Legacy v1 docs (`ARCHITECTURE.md`, `SYSTEM_CONTEXT.md`, `EXTERNAL_ACCESS.md`)
are kept at the repo root for historical reference. The live system is `v2/`.

---

## Contributing

Bug reports, feature requests, and pull requests are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the contributor guide.

---

## License

[MIT](LICENSE) © 2026 Reinhard
