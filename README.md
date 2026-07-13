# GraphRAG — Neuro-Symbolic Knowledge Graph Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![release-please](https://img.shields.io/badge/version-0.1.0--beta.1-orange)](.release-please-manifest.json)

Production-grade, 100% local GraphRAG with domain-agnostic extraction.
All models on GPU (RTX 3060, 12 GB). Python 3.11.

> **Status:** Beta pre-release toward v1.0.0 stable. See [Releases & Versioning](#releases--versioning-release-please--conventional-commits).

## Architecture (verified 2026-07-13)

| Component | Model | Port | VRAM | Role |
|-----------|-------|------|------|------|
| Embeddings | jina-embeddings-v3 (fp16) | :8000 | ~3.9 GB | Dense vectors + entity resolution |
| Reranker | bge-reranker-v2-m3 (fp16) | :8000 | shared | Result ranking |
| Entity NER | gliner_multi-v2.1 (fp16) | :8000 | ~0.6 GB | Zero-shot entity detection |
| Extraction LLM | gemma-4-E2B-it-QAT Q4_0 (`--reasoning off`) | :8082 | ~2.3 GB | Relation classification (per window) |
| Synthesis LLM | gemma-4-E4B-it-QAT Q4_0 (`--reasoning off`) | :8084 | ~3.6 GB | Answer writing (/ask) |
| Neo4j | — | :7687 | — | Entity graph |
| Qdrant | — | :6333 | — | Vector store |

**~11 GB / 12 GB with all three GPU models** (E2B + daemon + E4B).
Synthesis-only users can skip E4B → ~7.5 GB.

---

## Extraction Pipeline

| Mode | How | Precision | Latency | Use |
|------|-----|-----------|----------|-----|
| `sliding_window` | GLiNER entities + E2B relations, parallel windows | high-recall | 21-323s | **Default** — all docs, arbitrary length |
| `hybrid` | GLiNER entities → E2B relation classify | 100% | 10-15s | Fast single-pass (≤4K tokens) |
| `llm` | E2B full-doc single-pass | 89% | 11.8s | Fallback |

Supported domains: `engineering` (default), `journal`, `legal`, `medical`,
`accounting`, `hospitality`. Schema in `domain_config.yaml`.

---

## Prerequisites

- **NVIDIA GPU** (RTX 3060 12 GB tested). ~8 GB VRAM used by default; E4B
  synthesis adds ~3 GB. See [RUN.md → Requirements](RUN.md#1-requirements-read-before-you-start).
- **Docker + docker compose** (runs Neo4j + Qdrant — the only stateful deps).
- **Python 3.11** venv for the FastAPI daemon + ingest scripts.
- **Two GGUF model files you download yourself** (not in repo): Gemma 4 E2B
  (extraction) + Gemma 4 E4B (optional synthesis). See [RUN.md §1](RUN.md#1-requirements-read-before-you-start).

> **New here?** Read [RUN.md](RUN.md) first — it explains Gemma/E2B/llama.cpp in
> plain terms and gives the all-in-one startup. For how to evolve the data model
> without wiping Neo4j/Qdrant, read [MIGRATION.md](MIGRATION.md).

### Setup

```bash
git clone https://github.com/reinhardjs/polyglot-graphrag.git
cd polyglot-graphrag
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Start supporting services (Neo4j :7687/:7474, Qdrant :6333) via Docker
docker compose up -d
```

> **Full all-in-one walkthrough** (model download, systemd units, data flow,
> limitations) is in **[RUN.md](RUN.md)**. The short version:

```bash
# 1) Start GPU LLMs (Gemma E2B :8082, E4B :8084) + FastAPI daemon (:8000)
bash run.sh serve

# 2) Ingest bundled sample docs (engineering domain)
bash run.sh ingest sample_data

# 3) Ask (retrieval + synthesis — needs E4B; use synthesize:false if E4B down)
bash run.sh ask "who reported BUG-204?"
```

Or the legacy systemd path (deprecated; prefer `run.sh serve`):

```bash
sudo systemctl start gemma-4-e2b.service      # extraction LLM (:8082)
sudo systemctl start gemma-4-e4b.service      # synthesis LLM (:8084, optional)
sudo systemctl start rag-gpu-daemon           # pipeline daemon (:8000)
```

---

## Data Flow (how ingest & ask work)

**Ingest** — `POST /ingest` (async, poll `/ingest/status/{task_id}`):

```
text → chunk (per-domain strategy)
     → embed (Jina v3, GPU) ───────────────► Qdrant  <domain>_chunks  (vectors)
     → extract graph (Gemma E2B :8082) ───► Neo4j   :Entity nodes + edges
           (entity name embedded + vector-matched ≥0.88 → merge, not duplicate)
```

**Ask** — `POST /ask`:

```
query → embed (Jina) → Qdrant (top-K vectors)  ┐
                     → Neo4j (entity subgraph)  ├─► BGE rerank ─► (Gemma E4B :8084) answer
                                                ┘   synth:true  (synthesize:false → contexts only)
```

Re-ingesting the same `doc_id` **replaces** that doc's data cleanly (old chunks +
that doc's Neo4j nodes/edges removed, new written). Entities shared across
multiple documents are merged via vector resolution (≥0.88 cosine) and never
duplicated. See [MIGRATION.md](MIGRATION.md) for the full forward-compatibility
contract.

---

## Running Tests

```bash
# Unit tests
cd tests && pytest -v

# End-to-end (requires running daemon)
cd tests && pytest -v test_e2e_*

# A/B validation harness for Dynamic Labels
cd scripts && python validate_dynamic_labels.py
```

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for:

- Conventional Commit prefixes (`feat:`, `fix:`, `feat!:`, `docs:`, etc.)
- Release workflow via release-please
- Semantic versioning rules (pre-1.0 beta → feat bumps minor)

**Quick checklist for a contribution:**

1. Fork the repo, create a feature branch
2. Follow Conventional Commits (this drives versioning)
3. Add/update tests — run `pytest -v tests/`
4. If changing extraction logic, benchmark with `scripts/bench_drift.py`
5. Open a PR to `main`

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

## Sample Queries (guaranteed answers with bundled sample_data/)

Ingest the bundled sample docs first with `sync_docs.py sample_data/` (or `bash
run.sh ingest sample_data`) before running these:

- "who reported BUG-204 and what severity?" → bob, SEV-2
- "what PR implemented ADR-014 and who reviewed it?" → PR-482, carol
- "how does checkout relate to billing per our ADRs?" → billing depends on checkout
- "basis data apa yang digunakan ADR-021?" (ID) → PostgreSQL

---

## Files

| File | Purpose |
|------|---------|
| `config.py` | All ports, URIs, model names, extraction mode |
| `sync_docs.py` | Git-style file→graph sync client (SHA256 change detection) |
| `SYNC.md` | sync_docs.py usage guide |
| `QUICKSTART.md` | 5-minute copy-paste path for first-time users |
| `sliding_window.py` | Parallel sliding-window extraction (default ingest mode) |
| `serve_gpu.py` | Primary daemon — embed, rerank, GLiNER, ingest, /ask |
| `ingest.py` | Late-chunk vectors → Qdrant, graph extraction → Neo4j |
| `hybrid_extraction.py` | GLiNER+E2B hybrid extraction (fast path) |
| `label_provider.py` | Dynamic Label Injection — auto-expand GLiNER vocab |
| `domain_config.yaml` | Domain schemas (entity/relation types, chunking) |
| `domain_loader.py` | Config loader with validation |
| `query_modulator.py` | Domain-aware query expansion |
| `serve_cpu.py` | CPU-only daemon for benchmarking |
| `crag_pipeline.py` | Corrective RAG (adaptive routing) |
| `ask.py` | CLI client for /ask |
| `bench_rag.py`, `bench_ws2.py` | Stage-by-stage latency benchmarks |
| `sample_data/` | Engineering docs (PR, ADR, bug, wiki) |
| `tests/` | Unit + e2e test suite (pytest) |
| `plans/` | Architecture proposals |
| `scripts/` | Benchmarks / validation |
| `docs/` | Design & planning docs |
| `archive/legacy-v1/` | Dead v1-era code + superseded docs (history only) |
| `release-please-config.json` / `.release-please-manifest.json` | Release automation config |
| `.github/workflows/release-please.yml` | Automated release PRs from `main` |

---

## Repository Layout

```
polyglot-graphrag/            (git repo root = latest code on `main`)
├── config.py  serve_gpu.py  ingest.py  sync_docs.py  hybrid_extraction.py  ...
├── domain_config.yaml  run.sh  run_tests.sh
├── QUICKSTART.md  RUN.md  SYNC.md  MIGRATION.md  ARCHITECTURE.md  ...
├── sample_data/  tests/  plans/  scripts/
├── archive/
│   ├── legacy-v1/       historical v1 code (not used)
│   └── domains/         old TOML profiles (superseded by domain_config.yaml)
├── docs/                    design & planning docs
├── labels/  logs/           runtime state (gitignored)
└── release-please-config.json  .release-please-manifest.json  .github/
```

Versions are **no longer folder-based** (`v1/`, `v2/`, `v3/` removed). Versioning
is driven by **Git tags + release-please** (see below).

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

The `rag_query` tool calls `POST /ask`. Enable it in your Hermes profile:

```bash
hermes plugins enable rag
hermes
> how does checkout relate to billing per our ADRs?
```

---

## Documentation

| Document | Covers |
|----------|--------|
| `QUICKSTART.md` | **Fastest path** — 5-minute copy-paste from clone to query |
| `RUN.md` | **Start here** — all-in-one run guide, Gemma/E2B explained, requirements, limits |
| `SYNC.md` | File-sync daemon (`sync_docs.py`) — change tracking, --watch, config |
| `MIGRATION.md` | Forward-compat contract: enhance without resetting Neo4j/Qdrant |
| `ARCHITECTURE.md` | Full architecture, extraction modes, design decisions |
| `BENCHMARKS.md` | All benchmarks (WS1-4, journal, bugfix impact) |
| `NEXT_PHASE_PLAN.md` | Completed work + dynamic-label roadmap |
| `MODEL_SWITCHING.md` | How to swap any model component |
| `benchmark_report.md` | Lighter E2B (mobile QAT) evaluation — verdict: not a replacement |
| `plans/dynamic-label-injection.md` | Plan for fixing GLiNER entity drift |

---

## Version

**v0.1.0-beta.1** — current development (beta). The earlier v3.1.x line
(neuro-symbolic pipeline) has been **reset to 0.x.x** to signal the
beginning of a structured release process. All future releases are
beta pre-releases toward a final **v1.0.0** stable.

- GLiNER+E2B hybrid extraction (100% precision, engineering domain)
- Sliding window for arbitrary-length documents
- Journal domain (academic literature extraction)
- **Dynamic Label Injection** — `label_provider.py` auto-expands GLiNER's
  vocabulary with entity names E2B discovers that GLiNER missed. Candidates are
  promoted after `promotion_threshold` documents (default 3) and evicted via LRU
  when `max_per_domain` (default 20) is exceeded. Zero-latency in-memory merge
  on the hot path. State persists to `<project>/labels/{domain}_dynamic.json`
  (override location via `LABEL_STATE_DIR` env var).
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

- **Strategy 3: LLM Fallback NER** (opt-in, `llm_fallback.enabled: true` in
  `domain_config.yaml`). When E2B references an entity GLiNER missed, a single
  E2B call classifies the dropped name into a domain *semantic type*. The
  inferred type is promoted (not the raw name) → clean taxonomy + generalization
  (e.g. `Kubernetes`→`Component` rather than `type="Kubernetes"`). Recovered
  entities become typed graph nodes; `second_pass: true` re-extracts the
  dropped edges. Triggers only when drops occur (zero cost otherwise).

> Pre-v3 history (v1/v2 eras) is archived in `archive/legacy-v1/CHANGELOG_v1.md`.

---

## Releases & Versioning (release-please + Conventional Commits)

This repo uses **[release-please](https://github.com/googleapis/release-please)**
driven by **[Conventional Commits](https://www.conventionalcommits.org/)**.
There are **no versioned folders** — `main` always holds the latest code, and
every release is a **Git tag** cut from `main`.

**Workflow:**
1. Commit to `main` using Conventional Commit prefixes:
   - `feat:` → minor bump · `fix:` → patch · `feat!:` / `fix!:` / `BREAKING CHANGE:` → major
   - `docs:`, `chore:`, `refactor:`, `test:`, `ci:` → no release (or patch, configurable)
2. On push to `main`, `.github/workflows/release-please.yml` opens/updates a
   **release PR** that bumps the version in `.release-please-manifest.json` and
   regenerates the changelog.
3. Merge the release PR → release-please creates the **Git tag** (e.g. `v0.2.0-beta.1`)
   and the **GitHub Release**. The eventual **stable** will be `v1.0.0`.

**Current version** is seeded in `.release-please-manifest.json` (`"0.1.0-beta.1"`).
The project is currently on a **beta channel** (`prerelease-type: beta`), so
release-please tags pre-releases like `v0.1.0-beta.1`, `v0.1.0-beta.2`, … and
the GitHub Release is marked **Pre-release**. To cut the first **stable**
release, remove `prerelease-type` from `release-please-config.json` (or set the
next version via a `release-as:` commit) and merge the resulting release PR —
the tag becomes `v1.0.0` (non-prerelease).

> To force a specific version once: merge a commit with `release-as: X.Y.Z` in
> its body, or set `release-as` in `release-please-config.json`.

### Changelog (v3.0 → v3.1.5 — historical; version reset to 0.x.x)

| Version | Date | Highlights |
|---------|------|-----------|
| v3.0.8 | 2026-07-12 | Neuro-symbolic base: GLiNER+E2B hybrid, sliding window, journal domain |
| v3.1.0 | 2026-07-12 | Dynamic Label Injection (`label_provider.py`) — auto-expand GLiNER vocab |
| v3.1.1 | 2026-07-12 | Dynamic-label promotion/eviction/LRU/TTL hardening |
| v3.1.2 | 2026-07-13 | Strategy 3: LLM Fallback NER (semantic typing of dropped entities) |
| v3.1.3 | 2026-07-13 | Fix: forward `domain_name` in `sliding_window._parse_and_validate` |
| v3.1.4 | 2026-07-13 | Strategy 3 traceability: `:Discovered` label, audit `inferred_type`, provenance |
| v3.1.5 | 2026-07-13 | Default dynamic-label state to `<project>/labels/` (was `~/.hermes/labels`) — **end of v3.x line; version reset to 0.x.x beta begins** |

> Pre-v3 history (v1/v2 eras) is archived in `archive/legacy-v1/CHANGELOG_v1.md`.
