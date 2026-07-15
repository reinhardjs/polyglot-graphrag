# Domain Configuration — `domain_config.yaml`

Domain profiles declare how a domain is ingested and queried. They live in
**`domain_config.yaml`** (root of the repo) as a `domains:` mapping — one block
per domain/companion. No code changes are needed to add a domain; the config is
read at runtime by `config.load_domain_profile()` and `domain_loader.get_domain()`.

> Note: older docs referenced `v3/domains/*.toml` — that layout is gone. The
> single `domain_config.yaml` is the source of truth. The current version is
> **0.1.0 (Experimental)**.

## What a profile controls

```yaml
domains:
  engineering:                       # domain key used by `domain=` on API calls
    collection: engineering_chunks   # PRIMARY Qdrant corpus (real docs)
    neo4j_label: Engineering         # :Entity:Engineering graph label
    chunking:
      strategy: sentence             # sentence | paragraph | section | fixed
      chunk_size: 512
      overlap: 64
    companions: [engineering_docs]   # SECONDARY corpora (searched in parallel)
    # entry_strategy, entity_types, relation_types, synthesis, metadata_schema ...
```

## Selecting a domain

Pass `domain` on any API call. The profile drives collection routing, chunking,
extraction prompt, synthesis tone, graph labels, and entry strategy.

```bash
# Ingest into the engineering domain (routes to engineering_chunks)
curl -s -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" \
  -d '{"text":"...","doc_id":"enc-1","domain":"engineering"}'

# Query the engineering domain (primary graph + companion, fused)
curl -s -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"query":"who reported BUG-204?","domain":"engineering","synthesize":true}'
```

If `domain` is omitted, the system defaults to `enterprise` (the primary,
non-confidential corpus — `enterprise` Qdrant collection, dense prose
retrieval). SNOMED is available via `domain: snomed` / `domain: healthcare` but
is NOT the implicit default (its data is confidential / access-controlled).

## Primary corpus vs companion (secondary) corpus

* **Primary corpus** — the domain's own collection (`engineering_chunks`) plus
  its Neo4j graph. This is the domain's main knowledge. Ingested via
  `sync_docs.py` (git-style) or `POST /ingest`.
* **Companion (secondary) corpus** — an extra collection attached to a domain,
  declared via `companions: [name]`. It fills gaps the primary graph can't see
  (prose/design knowledge). Ingested via `POST /ingest_domain {"domain":"name"}`.
  Example: `engineering_docs` (this repo's `docs/`) is the companion of
  `engineering`; `clinical_prose` is the companion of `snomed`.

At query time, `/ask` runs the primary retriever AND every companion retriever
in parallel, tags each hit with `_signal` (= domain or companion name), then
fuses them. Hits appearing in ≥2 signals get a dual-evidence boost.

## Profile schema (key fields)

| Field | Meaning |
|-------|---------|
| `collection` | Qdrant collection for this domain's PRIMARY chunks. |
| `neo4j_label` | Neo4j `:Entity:<Label>` label — isolates graph traversal per domain. Omit for prose-only companions. |
| `chunking.strategy` | `sentence` (1 chunk/sentence), `paragraph` (1/para), `section` (split on `section_header_prefix`), `fixed` (sliding window). |
| `chunking.chunk_size` / `overlap` | Token budget / sliding-window overlap. |
| `companions` | List of companion corpus domain names searched in parallel. |
| `entity_types` / `relation_types` | **Single source of truth for the extraction vocabulary.** Flows into the LLM extraction prompt (via `{entity_types}`/`{relation_types}` tokens rendered by `ingest.py`), the structured/GLiNER extractors, and validation. Edit the vocab HERE — do NOT also hardcode it in an extraction prompt; the prompt is rendered from these values. |
| `entry_strategy` | Graph entry: `keyword` (token overlap, falls back to vector), `vector`, `hybrid`. |
| `metadata_schema.fields` | Declared metadata keys (unknown keys are warned, not dropped). |

## Available profiles

| Domain | Collection | Chunking | Companions | Notes |
|--------|-----------|----------|------------|-------|
| `enterprise` | `enterprise` | fixed | — | **Default** primary domain (non-confidential). ADRs, runbooks, postmortems, RFCs. |
| `snomed` | (graph-only) | — | `clinical_prose` | Term-match diagnosis over SNOMED + prose. Confidential / access-controlled. |
| `healthcare` | (alias → snomed) | — | `clinical_prose` | Friendly public name for the SNOMED clinical domain. |
| `clinical_prose` | `clinical_prose` | fixed | — | Companion: Wikipedia medicine prose (semantic symptom→disease). |
| `legal` | `legal` | fixed | — | GDPR/SOC2/HIPAA/contracts. Retrieval-only (neo4j_label: null). |
| `fraud` | `fraud` (graph) | none | `fraud` | Transaction graph (Neo4j) + narrative companion. |

## Adding a new domain

1. Add a `domains.<name>` block to `domain_config.yaml` (`collection`,
   `neo4j_label`, `chunking`, `companions:`, etc.).
2. (Optional) add `domains/<name>/retrieve.py` and/or `ingest.py` for custom
   behavior. Omit both to use the **default** retriever/ingestor — the primary
   needs no custom code at all.
3. Ingest (primary via `sync_docs.py`, companion via `POST /ingest_domain`).
   No restart required.

## API

```
GET /profiles      # all loaded domain profiles (name, collection, label, description)
```
