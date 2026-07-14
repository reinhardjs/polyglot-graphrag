# domains/ — per-domain ingest + retrieve plugins

Every knowledge domain owns its OWN ingest and retrieve logic here. The rest
of the system never hardcodes a domain's internals — it asks the registry:

```python
from domains import get_retriever, list_domains
retrieve = get_retriever("snomed")          # free-text query
ctxs = retrieve("fever and rash", top_k=5)
# graduated presentation:
retrieve_t = get_retriever("snomed", temporal=True)
ctxs = retrieve_t({"day1": ["fever"], "day2": ["rash"]}, top_k=5)
```

## Layout

```
domains/
  __init__.py            # registry: get_retriever(name, temporal=False)
  snomed/
    retrieve.py          # term-match + IDF + temporal (the diagnosis retriever)
  clinical_prose/
    ingest.py            # Wikipedia medicine -> Qdrant (semantic corpus)
    retrieve.py          # dense + sparse vector search over the prose
```

## Contract

Each domain folder is a **package** (directory with code). To be runnable it
MUST contain `retrieve.py` exposing:

* `retrieve(query: str, top_k: int = 5) -> list[dict]`
  each dict: `{"text", "doc_id", "doc_type", "chunk_idx"}`
* optionally `retrieve_temporal(presentation: dict, top_k=5) -> list[dict]`
  for graduated / day-by-day presentations.

Optionally `ingest.py` exposing `ingest(source=None, **kw)` that loads data
into that domain's Qdrant collection / Neo4j subgraph.

The domain's Qdrant collection name and Neo4j label are declared in
`domain_config.yaml` (`collection` / `neo4j_label`) — **never hardcoded** in
the retrieve/ingest code. `config.py.QDRANT_COLLECTIONS` mirrors that map for
callers that need it.

## Adding a new domain (no core changes)

1. `mkdir domains/<name>`
2. write `retrieve.py` with `retrieve(query, top_k)` (+ optional
   `retrieve_temporal`)
3. write `ingest.py` with `ingest(...)` (if it needs data loaded)
4. add a block to `domain_config.yaml` under `domains:` with `collection`,
   `neo4j_label`, `retrieval`, `entity_types`, `relation_types`,
   `pair_strategy`, `min_confidence`, `chunking`, `metadata_schema`
5. (optional) trigger ingest via `POST /ingest_domain {"domain":"<name>"}`

That's it — `/ask`, the reranker, and synthesis pick it up automatically.

## SNOMED + clinical_prose (hybrid diagnosis)

`snomed` alone can't describe disease presentations (concept names only). The
`clinical_prose` domain supplies free-text disease descriptions (Wikipedia
medicine articles) embedded into Qdrant. When `/ask` runs with
`domain=snomed`, it calls BOTH retrievers and merges their contexts before
rerank + synthesis, so a query like "fever, cough, rash" matches the Measles
*article* (semantic) even though the SNOMED "Measles (disorder)" name contains
none of those words (term-match would miss it).

The `snomed` domain config has `prose_domain: clinical_prose` to declare the
companion corpus; the `/ask` SNOMED branch reads that and merges.

## Registry semantics

* `get_retriever(name)` imports `domains/<name>/retrieve.py` on first use and
  caches the module.
* If the module defines `retrieve_temporal`, `get_retriever(name, temporal=True)`
  returns a callable that dispatches to it.
* `list_domains()` returns domains with a `retrieve.py` on disk (runnable set).
* `supported_domains()` returns domains declared in `domain_config.yaml`.
