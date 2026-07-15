# API Reference — GraphRAG GPU Daemon (`serve_gpu.py`)

All endpoints are served by the daemon on **`http://localhost:8000`** (configurable
via `DAEMON_URL`). The daemon hosts the auxiliary models (Jina embed, BGE rerank,
GLiNER, E2B extraction + synthesis via llama.cpp) and the retrieval/ingest API.

> Base URL: `http://localhost:8000`. All request/response bodies are JSON
> (`Content-Type: application/json`) unless noted. This is the **authoritative**
> endpoint list — if a roadmap/guide doc disagrees with it, this file wins.

## Retrieval

### `POST /ask` — hybrid retrieval (+ optional synthesis)
The main query endpoint. Searches Qdrant (vector + sparse) and the Neo4j graph
(in parallel, via `federated.assemble`), reranks, and optionally synthesizes an
answer with E2B (the same model that performs extraction).

```json
{ "query": "who reported BUG-204?",
  "domain": "engineering",        // optional; None → default_domain
  "top_k": 5,                     // default 5
  "synthesize": false,            // false → return raw contexts only (no synthesis)
  "skip_cache": false,
  "collection": null,             // optional Qdrant collection override
  "crag": false,                  // Corrective-RAG adaptive routing
  "dual_signal_only": false,      // keep only candidates confirmed by ≥2 signals
  "mode": null,                   // "differential" → ranked Dx (SNOMED)
  "min_confidence": null }        // "low"|"medium"|"high"
```
```bash
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"query":"who reported BUG-204?","domain":"engineering","synthesize":false}'
```
Response includes `contexts[]`, `contexts_meta[]` (each tagged with `_signal`,
`doc_id`, `doc_type`, `chunk_idx`, `dual_signal`), `path`, `n_contexts`, and
`answer` (when `synthesize:true`).

### `GET /doc/{collection}` — reconstruct a full document from chunks
Scrolls every Qdrant point in `collection` sharing `doc_id` and concatenates
them in `chunk_idx` order. `doc_id` is a **query param** (doc_ids may contain
slashes). See `GET /doc` example below.

```bash
curl "localhost:8000/doc/engineering_docs?doc_id=docs:e707efd2b0b0&include_chunks=false"
# → {collection, doc_id, source, chunk_count, char_count, text, chunks[]}
```

### `POST /embed_query` — embed a query string
Domain-agnostic query embedding (applies alias modulation first).

```json
{ "text": "total VRAM budget on the RTX 3060", "domain": "engineering" }
```
```bash
curl -s -X POST localhost:8000/embed_query -H 'Content-Type: application/json' \
  -d '{"text":"what is the VRAM budget?"}'   # → {"vector":[...], "text_used": "..."}
```

### `POST /rerank` — rerank candidates with BGE
```json
{ "query": "VRAM budget", "docs": ["doc A text", "doc B text"] }
```
```bash
curl -s -X POST localhost:8000/rerank -H 'Content-Type: application/json' \
  -d '{"query":"VRAM budget","docs":["E2B uses 1.5GB","unrelated text"]}'
# → {"ranked": [[idx, score], ...]}  (sorted desc by score)
```

### `POST /extract_entities` — GLiNER zero-shot NER
Returns entities with character spans.
```json
{ "text": "BUG-204 impacted the checkout service", "labels": ["Bug","Service"] }
```
```bash
curl -s -X POST localhost:8000/extract_entities -H 'Content-Type: application/json' \
  -d '{"text":"BUG-204 impacted checkout","labels":["Bug","Service"]}'
# → {"entities":[{"text":"BUG-204","label":"Bug","start":0,"end":6}, ...]}
```

## Ingestion

### `POST /ingest` — ingest a single document (async)
Embeds + extracts graph (unless `extract_graph:false`) and writes to Qdrant +
Neo4j. Returns a `task_id`; poll `GET /ingest/status/{task_id}`. If
`if_checksum` matches the stored payload, returns `304` (skipped).

```json
{ "doc_id": "adr-014",
  "text": "ADR-014 introduced the checkout EDA...",
  "doc_type": "eng",            // default "eng"
  "author": "unknown",
  "extract_graph": true,
  "if_checksum": null,          // pass stored checksum to skip-if-unchanged
  "collection": null,          // None → domain default (engineering_chunks)
  "domain": "engineering",
  "metadata": null }            // domain metadata schema
```
```bash
TID=$(curl -s -X POST localhost:8000/ingest -H 'Content-Type: application/json' \
  -d '{"doc_id":"x","text":"BUG-204 impacted checkout","domain":"engineering"}' \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['task_id'])")
curl -s localhost:8000/ingest/status/$TID
```

### `GET /ingest` — list ingested documents
Returns the current ingestion state (doc_ids + statuses).

### `GET /ingest/status/{task_id}` — poll an ingest task
```bash
curl localhost:8000/ingest/status/$TID
```

### `DELETE /ingest/{doc_id:path}` — delete a document
Removes the document (and its graph, if any) from both Qdrant and Neo4j.
`doc_id` may contain slashes, so it is a path segment.
```bash
curl -s -X DELETE "localhost:8000/ingest/eng/your-doc.md"
```

### `POST /ingest_domain` — trigger a domain-specific ingestor
Runs `domains/<domain>/ingest.py::ingest(...)` in a background thread (e.g. to
seed `clinical_prose`, `engineering_docs`, `example_companion`).
```json
{ "domain": "engineering_docs", "limit": 500, "seed": [ "optional doc list" ] }
```
```bash
curl -s -X POST localhost:8000/ingest_domain -H 'Content-Type: application/json' \
  -d '{"domain":"engineering_docs"}'
```

### `POST /extract_graph` — extract graph only (no chunk embed)
GLiNER zero-shot NER + co-occurrence edges (E2B-free fallback path).
```json
{ "text": "BUG-204 impacted checkout", "domain": "engineering", "extract_graph": true }
```

### `POST /embed_late` — late-chunking embed (ingest-time)
Chunks text per the requested strategy and embeds each chunk. Used internally
by ingest; exposed for testing/custom pipelines.
```json
{ "text": "long document...", "doc_id": "doc",
  "strategy": "sentence",       // sentence|paragraph|section|fixed
  "chunk_size": 512, "overlap": 64, "header_prefix": "##" }
```

## Status / introspection

### `GET /health` — liveness
```bash
curl localhost:8000/health
```

### `GET /models` — loaded model info (embedder dim, reranker, GLiNER)
```bash
curl localhost:8000/models
```

### `GET /config` — active daemon configuration (non-secret)
```bash
curl localhost:8000/config
```

### `GET /collections` — Qdrant collections + point counts
```bash
curl localhost:8000/collections
```

### `GET /profiles` — domain profiles from `domain_config.yaml`
Returns each domain's `collection`, `neo4j_label`, `description`, `chunking`
strategy, `entry_strategy`.
```bash
curl localhost:8000/profiles
```

### `GET /domains` — per-domain runnable/ingestable status
Returns `configured`, `runnable`, `ingestable`, `has_retriever_plugin`,
`has_ingest_plugin`, `collection`, `points`, `companions` for every domain,
plus `default_domain`.
```bash
curl localhost:8000/domains
```

## Admin

### `GET /admin/domains` — domain schema summary
Entity/relation types, pair strategy, min_confidence, collection, neo4j_label
per domain (from `domain_config.yaml`).

### `POST /admin/reload` — reload `domain_config.yaml` without restart
```bash
curl -s -X POST localhost:8000/admin/reload
# → {"status":"reloaded","domains":[...],"default_domain":"default"}
```

## Differentials (SNOMED / clinical, optional)

### `GET /differentials` — list persisted differentials (audit)
Optional `?query=<exact query>&limit=<n>`. Returns Differential nodes + ranked
`DxCandidate` children.

### `DELETE /differential` — delete a persisted differential
```bash
curl -s -X DELETE "localhost:8000/differential?query=<query>"
```

---

## Endpoint summary (22 routes)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/ask` | Hybrid retrieval (+ E2B synthesis) |
| GET | `/doc/{collection}` | Reconstruct full doc from chunks by `doc_id` |
| POST | `/embed_query` | Embed a query string |
| POST | `/rerank` | BGE rerank over candidates |
| POST | `/extract_entities` | GLiNER zero-shot NER |
| POST | `/ingest` | Ingest one document (async) |
| GET | `/ingest` | List ingested documents |
| GET | `/ingest/status/{task_id}` | Poll ingest task |
| DELETE | `/ingest/{doc_id:path}` | Delete a document |
| POST | `/ingest_domain` | Trigger domain ingestor |
| POST | `/extract_graph` | Graph-only extraction (E2B-free) |
| POST | `/embed_late` | Late-chunking embed |
| GET | `/health` | Liveness |
| GET | `/models` | Loaded model info |
| GET | `/config` | Active config |
| GET | `/collections` | Qdrant collections + counts |
| GET | `/profiles` | Domain profiles |
| GET | `/domains` | Per-domain runnable/ingestable status |
| GET | `/admin/domains` | Domain schema summary |
| POST | `/admin/reload` | Reload `domain_config.yaml` |
| GET | `/differentials` | List persisted differentials |
| DELETE | `/differential` | Delete a differential |
