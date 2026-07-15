# Architecture — polyglot-graphrag (v0.1.0, experimental)

> **Status: EXPERIMENTAL.** This describes the system as it actually runs at
> `0.1.0`. The config/API contract may still change. See [VERSIONING.md](VERSIONING.md).

---

## 1. Big picture

```
                ┌─────────────────────────── INGEST ───────────────────────────┐
 doc(.md/.pdf) ─┤ chunk ─► embed (Jina v3, :8000) ─► Qdrant (vectors)            │
                │        └► extract entities/relations (Gemma E2B, :8082) ─► Neo4j│
                └───────────────────────────────────────────────────────────────┘

                ┌─────────────────────────── ASK ──────────────────────────────┐
 query ─► embed ─► [Qdrant vectors] + [Neo4j graph] + [companion collections]
                │        (federated.assemble runs all in parallel, tagged _signal)
                └► rerank (BGE) ► fuse + dual-evidence boost ► E2B writes answer
```

Two stores, searched together:
- **Qdrant** (`:6333`) — vector store. Finds similar *text* by meaning.
- **Neo4j** (`:7687`) — graph store. Finds *related entities*
  ("bob → reported → BUG-204").

The daemon on `:8000` (`serve_gpu.py`) is the single HTTP surface: it embeds,
reranks, runs GLiNER (lazy), and proxies extraction/synthesis to the two
llama.cpp servers.

---

## 2. Components & ports

| Component | Model | Port | VRAM (approx) | Role |
|-----------|-------|------|------|------|
| GPU daemon | Jina v3 (embed) + BGE reranker-v2-m3 + GLiNER multi-v2.1 (lazy) | `:8000` | ~4.0 GB | embed / rerank / NER |
| Extraction LLM | Gemma-4-E2B Q4_0 (`--reasoning off`) | `:8082` | ~2.3 GB | entities + relations |
| Synthesis LLM | Gemma-4-E4B Q4_0 (`--reasoning off`) | `:8084` | ~3.6 GB | writes the `/ask` answer |
| Neo4j | — | `:7687` | — | entity graph |
| Qdrant | — | `:6333` | — | vector store |

**~10.6 GB / 12 GB** with all three GPU models. Synthesis-only users can skip
E4B → ~7 GB. Everything runs locally; nothing leaves the machine.

> **E4B is REQUIRED for synthesis.** `POST /ask` with `synthesize:true` needs
> E4B (`:8084`) up; otherwise use `synthesize:false` for retrieval-only. This is
> by design, not a bug.

---

## 3. Extraction pipeline

| Mode | How | Use |
|------|-----|-----|
| `sliding_window` | GLiNER entities + E2B relations, parallel windows (coref summaries) | **Default** — long docs, high recall |
| `hybrid` | GLiNER entities → E2B relation classify (one pass) | Fast single doc (≤4K tokens) |
| `llm` | E2B full-doc single-pass | Fallback |

Entities/relations are written to Neo4j via batched `UNWIND`. Entity resolution
is vector-driven (Jina v3 name embeddings, cosine ≥ 0.88 → merge) — no hardcoded
canonicalization map, so "Basis Data" ≡ "Database" converge automatically.

---

## 4. Domains & retrieval (Federated Factory)

All schema lives in `domain_config.yaml` (single source of truth). Retrieval is
built by `federated.assemble(domain)` — **no hardcoded domain branches**. A
domain declares:
- `collection` — its primary Qdrant collection (or `null` for a graph-only domain).
- `neo4j_label` — its graph node label (or `null` for a prose-only corpus).
- `companions: [ … ]` — extra Qdrant collections searched **in parallel** and
  tagged with `_signal`. Candidates confirmed by ≥2 signals get a rerank boost
  (dual-evidence).

### Domains with REAL ingested data

| Domain | Primary | Graph | Companions | Notes |
|--------|---------|-------|------------|-------|
| `snomed` | (graph-only term-match) | `SnomedConcept` | `clinical_prose` | terminology graph + Wikipedia prose companion |
| `default` | **alias → `snomed`** | — | — | stable public name; unqualified `/ask` uses it |

`clinical_prose` (Qdrant) is the semantic companion that supplies the
free-text disease descriptions SNOMED's structured graph cannot express.

### Schema stubs (configured, NOT yet seeded)

The active domains are `snomed` (graph-only) + `clinical_prose` companion,
`enterprise` (the ingested engineering knowledge base), `legal`, and `fraud`.
To add a new corpus, copy `domains/snomed/` or `domains/clinical_prose/` and
register it in `domain_config.yaml`.

### Companion — what it is

A companion is a **second Qdrant collection** attached to a domain. The
`snomed` primary is a structured terminology graph that misses the natural-language
presentations of diseases; the `clinical_prose` companion (Wikipedia medicine
articles) supplies that prose. See
[docs/domains/README.md](docs/domains/README.md).

---

## 5. API endpoints (serve_gpu.py :8000)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ask` | Full RAG (retrieval + optional synthesis). `domain` optional → `default`. |
| `POST` | `/ask/differential` | Differential diagnosis mode (clinical). |
| `POST` | `/embed_query` | Single query vector. |
| `POST` | `/embed_late` | Full-doc late-chunked vectors (companion ingest contract). |
| `POST` | `/rerank` | BGE rerank. |
| `POST` | `/extract_entities` | GLiNER zero-shot NER. |
| `POST` | `/ingest` | Single-doc ingest (non-blocking). |
| `POST` | `/ingest_domain` | Ingest a domain/companion corpus (runs its `domains/<name>/ingest.py`). |
| `GET`  | `/ingest/status/{id}` | Poll ingest task. |
| `DELETE` | `/ingest/{doc_id}` | Remove a doc. |
| `GET`  | `/domains` | List domains + default. |
| `GET`  | `/collections` | List Qdrant collections. |
| `GET`  | `/config` | Runtime config (reranker, dual-signal boost, default_domain…). |
| `GET`  | `/health` | GPU status + VRAM. |
| `POST` | `/admin/reload` | Hot-reload `domain_config.yaml`. |

The `/ask` response includes `contexts_meta[]` with a `signal` field so callers
can see which domain/companion each context came from.

---

## 6. Key design decisions

1. **Hybrid graph + vector.** The graph catches relationships vectors miss;
   vectors catch phrasing the graph's term-match misses. Fusion + dual-evidence
   boost prefers candidates confirmed by both.
2. **Entity resolution by embedding.** No hardcoded `CANON_MAP`; cross-lingual
   by construction.
3. **Domain-agnostic by construction.** Adding a domain = a folder +
   a `domain_config.yaml` block; `federated.assemble` does the rest.
4. **Companions as a first-class concept**, not a special case — generic
   `_signal` tagging means the dual-evidence boost works on any domain.
5. **100% local.** No external API; models served via llama.cpp.

---

## 7. Known gaps (experimental)

- `synthesize:true` works whenever E2B (:8082) is up (it serves both extraction and synthesis); else use `synthesize:false`.
- VRAM is tight on 12 GB — avoid concurrent heavy ingest + many parallel queries.
- Non-`snomed` domains are untested stubs unless you add and seed them.
- `/ask` is stateless (no multi-turn memory).
- GLiNER drops entity types not in the YAML `entity_types` list (dynamic-label
  work is tracked under `docs/roadmap/`).

---

## 8. Doc index

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Try-first entry point |
| [RUN.md](RUN.md) | Full run-from-zero guide |
| [QUICKSTART.md](QUICKSTART.md) | 5-min ingest→ask |
| [docs/domains/README.md](docs/domains/README.md) | Add a domain / companion |
| [VERSIONING.md](VERSIONING.md) / [CHANGELOG.md](CHANGELOG.md) | Versioning + history |
| [SYNC.md](SYNC.md) / [MIGRATION.md](MIGRATION.md) | Doc↔store sync, safe schema evolution |
| [docs/](docs/) | Architecture deep-dives, benchmarks, roadmaps |
