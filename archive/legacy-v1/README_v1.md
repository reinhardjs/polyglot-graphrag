# Polyglot GraphRAG v2

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/GPU-RTX%203060%2012GB-orange.svg" alt="RTX 3060 12GB">
  <img src="https://img.shields.io/badge/tests-76%20passed-brightgreen.svg" alt="Tests: 76 passed">
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
| **Neuro-Symbolic (v2.7.0)** | Phase 1 query modulation (alias expansion before embed) · Phase 2 subgraph pruning (Top-N centrality) · Phase 3 CRAG (adaptive routing + corrective fallback) · Phase 4 evaluation harness (faithfulness / relevancy / precision / recall) |

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
# ── Basic ask ──────────────────────────────────────────────────────────
# Retrieval-only (no LLM, ~0.17 s)
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"who reported BUG-204?","synthesize":false}'

# Full answer with synthesis + citation traceability
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"why did checkout stop calling the payment gateway?","domain":"engineering"}'

# ── Domain-agnostic (switch domain, no code change) ────────────────────
# Ask a medical question against the medical profile
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"what are the symptoms of gastritis?","domain":"medical","synthesize":true}'

# ── Neuro-Symbolic (v2.7.0) ────────────────────────────────────────────
# CRAG mode — adaptive routing + corrective retrieval (self-correcting)
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"who reported BUG-204?","domain":"engineering","crag":true}'
# Response includes: crag_route (graph|hybrid), crag_grade (CORRECT|INCORRECT),
# crag_corrected (true if corrective pass fired), crag_trace (full FSM path)
# Standard /ask fields (path, sources, contexts, answer) still present.

# Query modulation — see domain aliases expanded before embedding
curl -s -X POST http://127.0.0.1:8000/embed_query \
  -H "Content-Type: application/json" \
  -d '{"text":"patient has N/V","domain":"medical"}'
# Response: {"vector": [...], "text_used": "patient has N/V (nausea)"}
# The modulator expanded "N/V" → "nausea" before embedding.

# ── Evaluate quality ──────────────────────────────────────────────────
# Run the evaluation harness against golden datasets (live, via daemon)
cd v2
python evaluate_pipeline.py sample_data/golden/engineering.json --live --domain engineering
# Output: {"faithfulness": 0.87, "answer_relevancy": 0.71,
#          "context_precision": 1.0, "context_recall": 0.57, ...}

# Evaluate in CRAG mode to compare
python evaluate_pipeline.py sample_data/golden/engineering.json --live --domain engineering --crag

# ── Ingest into any domain ─────────────────────────────────────────────
curl -s -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text":"...","doc_id":"enc-1","domain":"medical",
       "metadata":{"patient_id":"P-42","title":"Encounter note"}}'

# ── Run tests ──────────────────────────────────────────────────────────
bash run_tests.sh unit    # fast, no daemon needed (51 tests)
bash run_tests.sh all     # full suite (76 tests) + eval smoke
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
| `POST` | `/ask` | One-call RAG. Accepts `domain`, `collection`, `synthesize`, `skip_cache`, `crag` (v2.7.0). Returns `contexts_numbered`, `contexts_meta`, `sources`, `answer`. CRAG mode adds `crag_route`, `crag_grade`, `crag_corrected`, `crag_trace`. |
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
<legacy-venv>/bin/python -m pytest tests/ -q
# 76 passed — all phases (unit) + cross-domain routing / CRAG (e2e, auto-skip w/o daemon)

# Or use the one-shot runner:
bash run_tests.sh unit   # fast, no daemon (51 tests)
bash run_tests.sh e2e    # live daemon tests
bash run_tests.sh eval   # Phase 4 eval smoke on golden datasets
bash run_tests.sh phase N  # one phase (1|2|3|4)
bash run_tests.sh all    # full suite (76) + eval smoke
```

---

## Documentation

| Document | Contents |
|----------|----------|
| [docs/README.md](docs/README.md) | Docs index |
| [docs/architecture/neuro-symbolic-v2.7.0.md](docs/architecture/neuro-symbolic-v2.7.0.md) | Complete v2.7.0 neuro-symbolic architecture + quality improvement roadmap |
| [docs/architecture/architecture.md](docs/architecture/architecture.md) | System design, VRAM budget, data flow, design decisions |
| [docs/architecture/full-architecture.md](docs/architecture/full-architecture.md) | Every component, every config knob |
| [docs/architecture/hermes-integration.md](docs/architecture/hermes-integration.md) | Hermes `rag_query` plugin setup |
| [docs/domains/README.md](docs/domains/README.md) | TOML profile schema + how to add a domain |
| [docs/guides/model-startup.md](docs/guides/model-startup.md) | Every model, every flag, why — llama-server parameters, VRAM allocation, cold-start sequence |
| [docs/guides/development.md](docs/guides/development.md) | Environment, workflows, testing |
| [docs/roadmap/neuro-symbolic-plan.md](docs/roadmap/neuro-symbolic-plan.md) | Neuro-Symbolic GraphRAG upgrade (all 4 phases shipped) |
| [docs/roadmap/agent-learning-system.md](docs/roadmap/agent-learning-system.md) | v3.0.0 self-improving loop (planned) |
| [docs/comparison-greycat.md](docs/comparison-greycat.md) | How we compare to GreyCat's unified engine |
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
